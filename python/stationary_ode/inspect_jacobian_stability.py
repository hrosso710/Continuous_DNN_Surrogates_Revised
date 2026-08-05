"""
Diagnostic: compute the Jacobian J(t) = d f(y,t;theta(t)) / dy at points
along EACH MODEL'S OWN predicted trajectory in the test region [20, 25],
and report its eigenvalues' real parts.

Why the model's own trajectory (not the true trajectory): what governs
error growth during forward integration is the local stability of the
vector field along the path the solver actually follows, which is the
model's own predicted y(t), not the ground truth. A positive real part in
the Jacobian spectrum means nearby trajectories diverge locally (matches
the manuscript's own Section 3.3.1 stability framework); the more positive
the max real part, the faster errors should compound -- which is exactly
what inspect_test_error_profile.py showed happening ~3x faster for
legendre than monomial.
"""
import torch
import sys
from torchdiffeq import odeint

T0, T1 = 0.0, 25.0
true_y0 = torch.tensor([[2., 0.]])
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
DATA_SIZE = 500
t_grid = torch.linspace(T0, T1, DATA_SIZE)


class ODEFunc(torch.nn.Module):
    """Copied standalone from ode_demo_NEW.py -- see inspect_test_error_profile.py
    for why this isn't imported directly."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 2),
        )

    def forward(self, t, y):
        return self.net(y ** 3)


class LambdaLocal(torch.nn.Module):
    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


sys.path.insert(0, '..')
from timeparam_NEW import TimeParameterizedNet  # noqa: E402


def get_true_trajectory():
    with torch.no_grad():
        return odeint(LambdaLocal(), true_y0, t_grid, method='rk4')


def load_func(basis, degree=3, seed=0):
    ckpt_path = f'best_checkpoint_{basis}_d{degree}_seed{seed}.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    d = sd['time_params.net__0__weight'].shape[0]
    base = ODEFunc()
    func = TimeParameterizedNet(base, tspan=[T0, T1], basis=basis, d=d)
    func.load_state_dict(sd)
    func.eval()
    return func


def jacobian_eigs(func, t_scalar, y_vec):
    """d f(y, t) / dy, evaluated at a single (t, y) point. y_vec: shape (2,)."""
    y = y_vec.clone().requires_grad_(True)

    def f(y_in):
        # func expects (t, y) with y of shape (1, 2) given how TimeParameterizedNet
        # loops over the batch dim internally.
        return func(torch.tensor(t_scalar), y_in.unsqueeze(0)).squeeze(0)

    J = torch.autograd.functional.jacobian(f, y)  # (2, 2)
    eigs = torch.linalg.eigvals(J)
    return J.detach(), eigs


def profile(basis, degree=3, seed=0):
    func = load_func(basis, degree, seed)
    true_y = get_true_trajectory()
    val_end_idx = int(0.80 * DATA_SIZE)

    y0_test = true_y[val_end_idx:val_end_idx + 1].squeeze(0)
    t_test = t_grid[val_end_idx:]
    with torch.no_grad():
        pred_test = odeint(func, y0_test, t_test, rtol=1e-7, atol=1e-9).squeeze(1)  # (100, 2)

    print(f"\n=== {basis}: Jacobian eigenvalues along the MODEL'S OWN test-region trajectory ===")
    print(f"{'t':>7} {'y(t) [model]':>22} {'max Re(eig)':>14} {'eigenvalues':>28}")
    check_idx = list(range(0, len(t_test), 10)) + [len(t_test) - 1]
    max_re_over_window = []
    for i in sorted(set(check_idx)):
        t_i = t_test[i].item()
        y_i = pred_test[i]
        J, eigs = jacobian_eigs(func, t_i, y_i)
        max_re = eigs.real.max().item()
        max_re_over_window.append(max_re)
        y_str = f"[{y_i[0].item():7.3f}, {y_i[1].item():7.3f}]"
        eig_str = ", ".join(f"{e.real:.3f}{'+' if e.imag>=0 else '-'}{abs(e.imag):.3f}i" for e in eigs)
        print(f"{t_i:7.2f} {y_str:>22} {max_re:14.4f} {eig_str:>28}")

    print(f"  mean(max Re(eig)) over test window: {sum(max_re_over_window)/len(max_re_over_window):.4f}")
    return max_re_over_window


if __name__ == '__main__':
    results = {}
    for basis in ['monomial', 'legendre']:
        results[basis] = profile(basis)

    print("\n=== SUMMARY: mean max Re(eig) over test-region trajectory ===")
    for basis, vals in results.items():
        print(f"  {basis:10s}: {sum(vals)/len(vals):.4f}")
