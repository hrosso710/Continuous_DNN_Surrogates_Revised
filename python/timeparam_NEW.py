import torch
import torch.nn as nn
import torch.nn.functional as F


def sanitize_key(key: str) -> str:
    """Replace '.' with '__' to avoid conflicts in ParameterDict keys."""
    return key.replace('.', '__')


def unsanitize_key(skey: str) -> str:
    """Revert the key sanitization so we can load into net via load_state_dict."""
    return skey.replace('__', '.')


class TimeParameterizedNet(nn.Module):
    """
    Wrap a PyTorch module (net) so that its weights become time-dependent
    via a polynomial expansion of degree d-1 in time.

    - Each original parameter net.state_dict()[name] is replaced by a new parameter
      of shape (d, *original_param.shape).
    - In the forward pass, we evaluate the polynomial basis [1, t, t^2, ..., t^(d-1)]
      (or the analogous Legendre basis) and form a weighted sum over those (d, ...)
      parameters to reconstruct the parameter at time t. Then we load that into net
      and do a forward pass.

    BUG FIX (time normalization): both the monomial basis (t**i) and the Legendre
    basis (which is only orthogonal/well-behaved on [-1, 1]) assume the incoming
    `t` already lies in [0, 1]. Previously `tspan` was stored but never used, so
    if this module was wrapped around dynamics evaluated over an absolute time
    range other than [0, 1] (e.g. t in [0, 25]), the basis functions were evaluated
    far outside their intended domain, causing blow-up (monomial) or loss of
    orthogonality/explosive extrapolation (Legendre). We now always rescale the
    incoming absolute time into [0, 1] via `tspan` before building the basis, and
    clamp defensively in case an adaptive ODE solver slightly overshoots the
    nominal integration bounds.
    """

    def __init__(self, net: nn.Module, tspan, basis="legendre", d: int = 3):
        """
        Args:
            net   (nn.Module): The original network whose parameters we wrap.
            tspan (list or tuple of float): [t0, t1], the ABSOLUTE time span over
                         which this net will be evaluated (e.g. [0, 25], not
                         necessarily [0, 1]). Incoming times are rescaled into
                         [0, 1] using this span before the basis is evaluated.
            d     (int): The degree of the polynomial expansion, i.e. we learn
                         coefficients for [1, t, ..., t^(d-1)].
            basis (str): Either "monomial" or "legendre".
        """
        super().__init__()
        self.net = net
        self.tspan = tspan
        self.t0 = float(tspan[0])
        self.t1 = float(tspan[1])
        if self.t1 <= self.t0:
            raise ValueError(f"tspan must satisfy t1 > t0, got tspan={tspan}")
        self.d = d
        self.basis = basis.lower()

        # BUG FIX (dead parameters): self.net's own original parameters are
        # never used in forward() -- functional_call() always substitutes the
        # full current_params dict built from time_params, for every named
        # parameter. So self.net's raw parameters were vestigial (kept only
        # as the shape/initial-value template copied into time_params[0]
        # below), yet were left trainable and thus (a) sat in the optimizer's
        # parameter list doing nothing, and (b) got included in
        # `sum(p.numel() for p in func.parameters() if p.requires_grad)`,
        # silently inflating every reported parameter count in Table 4 by
        # this net's raw size (252 for the stationary-ODE architecture).
        # Freezing them here removes both problems with no change to
        # forward() behavior (they were already functionally inert).
        for param in self.net.parameters():
            param.requires_grad = False

        # 2) Extract the original state of net (including all named_parameters).
        original_state = self.net.state_dict()

        # 3) Create a ParameterDict in which each parameter has shape (d, *orig_shape).
        self.time_params = nn.ParameterDict()

        # 4) Keep a mapping from sanitized key -> original key.
        #    We do this so we can safely store them in time_params,
        #    but still load them into net with the original key.
        self.key_mapping = {}

        for name, orig_tensor in original_state.items():
            # Create new parameter with one extra leading dimension = d
            # shape: (d, *orig_tensor.shape)
            p = nn.Parameter(torch.zeros(d, *orig_tensor.shape))

            # Initialize so that p[0,...] = original parameter (like standard basis)
            with torch.no_grad():
                p[0].copy_(orig_tensor)

            # Sanitize the key (replace '.' with '__')
            sname = sanitize_key(name)
            self.time_params[sname] = p
            self.key_mapping[sname] = name  # Remember how to revert the key

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale absolute time t (in [t0, t1]) into [0, 1], clamped defensively."""
        t_norm = (t - self.t0) / (self.t1 - self.t0)
        return t_norm.clamp(0.0, 1.0)

    def _build_basis(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Build the (batch, d) basis matrix from normalized time in [0, 1]."""
        if self.basis == "monomial":
            a = torch.cat([t_norm ** i for i in range(self.d)], dim=1)
        elif self.basis == "legendre":
            x_leg = 2 * t_norm - 1  # rescale [0,1] -> [-1,1]
            a_list = [torch.ones_like(x_leg)]  # P0(x) = 1
            if self.d > 1:
                a_list.append(x_leg)  # P1(x) = x
            for n in range(2, self.d):
                Pn = ((2 * n - 1) * x_leg * a_list[-1] - (n - 1) * a_list[-2]) / n
                a_list.append(Pn)
            a = torch.cat(a_list, dim=1)
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")
        return a

    def gram_matrix(self, device=None, dtype=None) -> torch.Tensor:
        """
        M_de := \\int_0^1 p_d(t) p_e(t) dt, integrated over the SAME normalized
        t_norm in [0, 1] that _build_basis operates on (i.e. this matches what
        the model actually treats as "time" internally, not necessarily the
        absolute tspan duration -- see the regularization comment in
        ode_demo_NEW.py for why that's the right choice here).

        Closed form (no numerical integration needed):
          - monomial:  M_de = 1 / (d + e + 1)                 (dense)
          - legendre:  M_de = delta_de / (2d + 1)              (diagonal,
            using orthogonality of the shifted Legendre polynomials on [0,1];
            equivalent to the manuscript's normalization on [-1, 1] up to the
            constant Jacobian factor from the t_norm -> x_leg = 2 t_norm - 1
            change of variables, which is absorbed into alpha).

        This is Eq. (17)'s M matrix (Section 4.2): dense for monomial (couples
        regularization gradients across degrees), diagonal for Legendre
        (each coefficient regularized independently, scaled by 1/(2d+1)).
        """
        d = self.d
        idx = torch.arange(d, dtype=dtype or torch.float32, device=device)
        if self.basis == "monomial":
            M = 1.0 / (idx.unsqueeze(0) + idx.unsqueeze(1) + 1.0)
        elif self.basis == "legendre":
            M = torch.diag(1.0 / (2.0 * idx + 1.0))
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")
        return M

    def regularization_term(self) -> torch.Tensor:
        """
        R_NODE(theta) := (alpha/2) * \\int_0^1 ||theta_NODE(t)||_2^2 dt, WITHOUT
        the alpha factor (caller multiplies by alpha/2 -- see ode_demo_NEW.py),
        i.e. this returns \\int_0^1 ||theta_NODE(t)||_2^2 dt itself, computed
        via the Gram matrix as sum_{d,e} M_de <theta_d, theta_e> (manuscript
        Eq. 5's second term, expanded via the degree-(D+1) basis expansion of
        Eq. 12/13). This is the SAME quantity whose Frechet derivative w.r.t.
        a single coefficient theta_d is given in Eq. (17); we don't need that
        gradient formula explicitly since autograd differentiates this scalar
        directly, but the diagonal-vs-dense M structure is exactly what makes
        the resulting gradients match Eq. (17) for Legendre vs monomial.
        """
        total = None
        for time_param in self.time_params.values():
            # time_param: (d, *orig_shape) -> flatten to (d, N)
            d = time_param.shape[0]
            P = time_param.reshape(d, -1)
            M = self.gram_matrix(device=P.device, dtype=P.dtype)
            term = torch.sum(P * (M @ P))  # = trace(P^T M P) = sum_de M_de <theta_d, theta_e>
            total = term if total is None else total + term
        return total

    def forward(self, t, x):
        """
        Forward pass:
          1) Rescale absolute time t into [0, 1] using self.tspan.
          2) Construct the polynomial basis vector a = [1, t, t^2, ..., t^(d-1)]
             (or Legendre analogue) from the NORMALIZED time.
          3) For each parameter, sum_{i=0 to d-1} a[i] * time_params[name][i].
          4) Load these computed parameters into self.net.
          5) Call net's forward.

        Args:
            t (float or Tensor): The ABSOLUTE time at which to evaluate the
                weights (i.e. in the same units as tspan, e.g. [0, 25], not
                assumed to already be in [0, 1]).
            x (Tensor): Input features.

        Returns:
            Tensor: Output of the wrapped net at time t.
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)  # shape (1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # shape (batch_size, 1)
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], -1)

        # --- BUG FIX: normalize absolute time into [0, 1] before building basis ---
        t_norm = self._normalize_time(t)
        a = self._build_basis(t_norm)
        # ---------------------------------------------------------------------

        # NOTE (unchanged / not part of this fix): this per-sample Python loop is
        # a separate performance issue (Bug #2) -- each batch element triggers its
        # own functional_call. Left as-is here since this rewrite only addresses
        # the time-normalization bug; consider vectorizing with torch.vmap later.
        outputs = []
        for i in range(x.shape[0]):
            a_i = a[i]  # (d,)
            current_params = {}
            for sname, time_param in self.time_params.items():
                param_t = torch.einsum('d,d...->...', a_i, time_param)
                current_params[self.key_mapping[sname]] = param_t
            out = torch.func.functional_call(self.net, current_params, (t[i], x[i].unsqueeze(0)))
            outputs.append(out)

        return torch.cat(outputs, dim=0)


# Example usage
if __name__ == "__main__":
    import torch.optim as optim

    # A simple base net that takes (t, x) and does something trivial
    class BaseNet(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, t, x):
            out = self.fc1(x)
            out = F.relu(out)
            out = self.fc2(out)
            return out

    base_net = BaseNet(input_dim=2, hidden_dim=4, output_dim=1)

    # Wrap it into a time-parameterized net. tspan is the ABSOLUTE time range
    # this net will be evaluated over -- here it happens to already be [0, 1].
    t0, t1 = 0.0, 1.0
    time_net = TimeParameterizedNet(base_net, tspan=[t0, t1], d=5)

    # Print the time_params and shapes
    for name, param in time_net.time_params.items():
        print(f"{name} : {param.shape}")

    # Example: forward pass at different times
    x = torch.randn(5, 2)  # batch of 5
    y_t0 = time_net(0.0, x)  # output at time = 0.0
    y_t1 = time_net(1.0, x)  # output at time = 1.0
    y_test = base_net(0.0, x)

    print("Output at t=0.0 :", y_t0.squeeze())
    print("Output at t=1.0 :", y_t1.squeeze())
    print("Output base net :", y_test.squeeze())

    # Show difference in the actual parameters at t=0 vs t=1
    sd_0 = {}
    sd_1 = {}
    with torch.no_grad():
        time_net(0.0, x)  # This loads net with the param set for t=0
        sd_0 = {k: p.clone() for k, p in time_net.net.state_dict().items()}

        time_net(1.0, x)  # This loads net with the param set for t=1
        sd_1 = {k: p.clone() for k, p in time_net.net.state_dict().items()}

    print("\nComparing final layer (fc2) weight at t=0 vs t=1:")
    print("fc2.weight at t=0:", sd_0["fc2.weight"])
    print("fc2.weight at t=1:", sd_1["fc2.weight"])

    # Demonstrate training so the network can learn a time-varying mapping
    optimizer = optim.SGD(time_net.parameters(), lr=1e-1)
    for name, param in time_net.time_params.items():
        print(f"{name} : {param}")

    # Suppose we want y(t) = t * sum(x) just as a silly target
    for step in range(100):
        optimizer.zero_grad()

        x_train = torch.randn(10, 2)
        t_train = torch.rand(1).item()  # scalar in [0, 1)
        target = t_train * x_train.sum(dim=1, keepdim=True)

        y_pred = time_net(t_train, x_train)
        loss = F.mse_loss(y_pred, target)
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:3d}, t={t_train:.2f}, Loss={loss.item():.4f}")

    print("\nTraining complete. The time-parameterized network can now "
          "adjust weights as a function of time.")
    for name, param in time_net.time_params.items():
        print(f"{name} : {param}")
