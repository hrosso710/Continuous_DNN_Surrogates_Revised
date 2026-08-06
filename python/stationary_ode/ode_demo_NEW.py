# %%
import os
import sys
import argparse
import json
import random
import time
import numpy as np
import matplotlib as plt

# stationary_ode/ is now one level below python/, where the shared modules live
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timeparam_NEW import TimeParameterizedNet
from experiment_runtime import print, makedirs, relative_error, RunningAverageMeter

import torch
import torch.nn as nn
import torch.optim as optim


plt.rcParams.update({
    "text.usetex": False,  # Avoid needing LaTeX installation
    "font.family": "cmr10",  # Use Matplotlib's built-in Computer Modern
    "axes.labelsize": 12,
    "font.size": 12,
    "axes.formatter.use_mathtext": True
})

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--basis', choices=['legendre', 'monomial', 'none'], default='legendre',
                     help='"none" trains a standard, time-INVARIANT neural ODE (ODEFunc '
                          'used directly, no TimeParameterizedNet wrapping) -- the static/'
                          'identity baseline needed for Table 3\'s "static (identity)" row '
                          'and its ΔError column (Referee 2, comment 2.6 / '
                          'Referee 3, comment R3.6).')
parser.add_argument('--degree', type=int, default=3,
                     help='Ignored (forced to 0) when --basis none.')
parser.add_argument('--seed', type=int, default=0,
                     help='Random seed controlling model init (ODEFunc weight init) '
                          'and training-window sampling (get_batch). Set explicitly '
                          'and vary across runs (e.g. 0-4) to report mean/std over '
                          'multiple seeds, as Tables 5/6 require.')
parser.add_argument('--grad_clip', type=float, default=0.0,
                     help='Max gradient norm for clipping (torch.nn.utils.clip_grad_norm_). '
                          '0 disables clipping (default, matches prior behavior).')
parser.add_argument('--lr_decay_every', type=int, default=0,
                     help='Decay the learning rate by --lr_decay_gamma every N iterations. '
                          '0 disables decay (default, matches prior behavior).')
parser.add_argument('--lr_decay_gamma', type=float, default=0.5,
                     help='Multiplicative LR decay factor, applied every --lr_decay_every '
                          'iterations if that is > 0.')
parser.add_argument('--early_stop_patience', type=int, default=0,
                     help='Stop training after this many consecutive VALIDATION checks '
                          '(each --test_freq iterations apart) with no improvement in '
                          'validation relative error. 0 disables early stopping (default). '
                          'This is a pre-registered stopping rule -- the checkpoint at '
                          'which training stops IS the reported result.')
parser.add_argument('--alpha', type=float, default=0.0,
                     help='Regularization strength, matching the manuscript objective '
                          '(Eq. 3/5): loss = data misfit + (alpha/2) * R(theta), where '
                          'for basis in {monomial, legendre} R(theta) is the L2(0,T) norm '
                          'of the time-varying weight trajectory (Eq. 5, computed exactly '
                          'via the closed-form Gram matrix -- dense for monomial, diagonal '
                          'for legendre, matching Eq. 17), and for basis=none R(theta) is '
                          'the standard L2 norm of the (time-invariant) weights, i.e. the '
                          'D=0 degenerate case of the same expression. Default 0.0 recovers '
                          'the previous (unregularized) behavior exactly.')
parser.add_argument('--tspan_mode', type=str, default='full', choices=['full', 'train'],
                     help='"full" (default): normalize time against the entire absolute '
                          'span [0, T_MAX], so val/test evaluate the basis functions past '
                          'their trained sub-interval (continued polynomial extrapolation). '
                          '"train": normalize against [0, train_end_t] only, so the '
                          'internal clamp freezes theta(t) at its train_end_t value for '
                          'all of val/test (zero-order-hold extrapolation instead of '
                          'polynomial extrapolation). Experimental -- see comment above '
                          'train_end_t below.')
parser.add_argument('--results_json', type=str, default=None,
                     help='Path to write a structured JSON summary of this run. Defaults '
                          'to results_{basis}_d{degree}_seed{seed}.json if not given.')
parser.add_argument('--rtol', type=float, default=1e-7,
                     help='Relative tolerance for the dopri5 solver (torchdiffeq default: '
                          '1e-7). NFE (and wall-clock) scales directly with how tight this '
                          'is -- loosening it (e.g. 1e-4) trades some precision for speed.')
parser.add_argument('--atol', type=float, default=1e-9,
                     help='Absolute tolerance for the dopri5 solver (torchdiffeq default: '
                          '1e-9). See --rtol.')
parser.add_argument('--train_end_frac', type=float, default=0.64,
                     help='Fraction of the total time domain [0, T_MAX] used for TRAINING '
                          '(get_batch() only samples windows from this region). Default '
                          '0.64 -> for T_MAX=25, train = t in [0, 16].')
parser.add_argument('--val_end_frac', type=float, default=0.80,
                     help='Cumulative fraction of [0, T_MAX] marking the end of the '
                          'VALIDATION region (val = t in [train_end, val_end]; test = '
                          't in [val_end, T_MAX]). Default 0.80 -> for T_MAX=25, '
                          'val = t in [16, 20], test = t in [20, 25].')
parser.add_argument('--batch_time_max', type=int, default=None,
                     help='If set, --batch_time grows linearly from its initial value up '
                          'to this value over training (a curriculum), to combat exposure '
                          'bias / compounding error from training on short, isolated windows '
                          '-- diagnosed via diagnostic_rollout.py showing low error on '
                          'isolated windows but error growing sharply with continuous '
                          'rollout length. None (default) disables growth, matching prior '
                          'fixed-window behavior.')
parser.add_argument('--batch_time_grow_every', type=int, default=100,
                     help='Grow batch_time by one increment every this many iterations, '
                          'when --batch_time_max is set.')
parser.add_argument('--batch_time_grow_steps', type=int, default=None,
                     help='Number of growth increments to reach --batch_time_max, spaced '
                          '--batch_time_grow_every iterations apart. Growth completes after '
                          'batch_time_grow_steps * batch_time_grow_every iterations, which '
                          'should be well BEFORE --niters so there is real training time '
                          'left at the full window length -- growth completing exactly at '
                          'the final iteration (e.g. by naively setting this to '
                          'niters // grow_every) leaves no time to actually train at the '
                          'target window length. Defaults to niters // (4 * grow_every), '
                          'i.e. growth completes by 25% of the training budget.')
args = parser.parse_args()

if args.batch_time_max is not None and args.batch_time_max > 50 and not args.adjoint:
    print("WARNING: --batch_time_max is large and --adjoint is not set. Backprop through "
          "long integration windows without the adjoint method can be memory- and "
          "compute-intensive. Consider --adjoint. Also note (per the manuscript's own "
          "Section 3.3.1) that adjoint-based backward reconstruction of the forward "
          "trajectory can itself become less accurate over long, dissipative integration "
          "windows -- worth sanity-checking convergence either way.")

# Seed every RNG this script touches BEFORE any model is constructed or any
# batch is sampled, so runs are actually reproducible and distinguishable by
# --seed alone.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(f"The device being used is {device}.")
print(f"Seed: {args.seed} | Basis: {args.basis} | Degree: {args.degree} | Data size: {args.data_size}")

true_y0 = torch.tensor([[2., 0.]], device=device)
t = torch.linspace(0., 25., args.data_size, device=device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

# The absolute time domain the network will actually be evaluated over.
# TimeParameterizedNet rescales whatever absolute time it receives into
# [0, 1] using this span before building its polynomial/Legendre basis.
TSPAN = [float(t[0].item()), float(t[-1].item())]
T_MAX = TSPAN[1]

# ---------------------------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT (Referee 2, comment 2.6.5; agreed "Option A":
# contiguous, non-overlapping time regions, extended to three regions rather
# than two so validation can drive early stopping without ever touching the
# held-out test region).
#
#   TRAIN region: t in [0, train_end_t]         -- get_batch() samples windows
#                                                    from here only.
#   VAL region:   t in [train_end_t, val_end_t]  -- drives early stopping ONLY;
#                                                    never used for gradient
#                                                    steps.
#   TEST region:  t in [val_end_t, T_MAX]        -- touched exactly once, after
#                                                    training stops, for the
#                                                    final reported number.
#
# All three are evaluated the SAME way for reporting purposes: integrate the
# model forward from the true state at the region's left boundary, compute
# relative error against ground truth over that region. This makes train/val/
# test numbers directly comparable in Table 3, addressing comment 2.6.5
# (validation used for tuning/stopping, test reserved for final comparison)
# together with 2.6.4 (relative, not raw, error).
# ---------------------------------------------------------------------------
train_end_t = args.train_end_frac * T_MAX
val_end_t = args.val_end_frac * T_MAX

if args.tspan_mode == 'train':
    # EXPERIMENTAL (see inspect_jacobian_stability.py / weight-trajectory
    # diagnostics): normalize time against the TRAINING window only, not the
    # full span. Since t_norm is clamped to [0,1] inside TimeParameterizedNet,
    # this makes theta(t) FREEZE at its t=train_end_t value for all of
    # val/test, instead of continuing to trace out the polynomial basis's
    # shape (which for Legendre has a genuine interior curvature reversal --
    # P3's critical point -- landing inside the val region). Motivated by the
    # ground truth here being time-invariant, so there's no principled reason
    # theta(t) should keep varying past the point where data stops
    # constraining it.
    TSPAN = [TSPAN[0], train_end_t]

_t_cpu = t.detach().cpu()
train_end_idx = int(torch.searchsorted(_t_cpu, torch.tensor(train_end_t)).item())
val_end_idx = int(torch.searchsorted(_t_cpu, torch.tensor(val_end_t)).item())
# Guard against degenerate splits (e.g. --data_size too small for --batch_time).
train_end_idx = max(args.batch_time + 1, min(train_end_idx, args.data_size - 3))
val_end_idx = max(train_end_idx + 1, min(val_end_idx, args.data_size - 1))

print(f"Split: train=[0, {train_end_t:.2f}] (idx 0:{train_end_idx}) | "
      f"val=[{train_end_t:.2f}, {val_end_t:.2f}] (idx {train_end_idx}:{val_end_idx}) | "
      f"test=[{val_end_t:.2f}, {T_MAX:.2f}] (idx {val_end_idx}:{args.data_size})")


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='rk4')

# Region-specific time points / initial conditions / ground truth, used for
# the final train/val/test relative-error reporting (see main block below).
# Boundary points are shared between adjacent regions (e.g. true_y at
# train_end_idx is both the last train point and the val region's known
# initial condition) -- this is not leakage, it's just the known true state
# at that time, exactly analogous to true_y0 being given at t=0.
t_train_region = t[:train_end_idx + 1]
y0_train_region = true_y0
true_y_train_region = true_y[:train_end_idx + 1]

t_val_region = t[train_end_idx:val_end_idx + 1]
y0_val_region = true_y[train_end_idx:train_end_idx + 1]
true_y_val_region = true_y[train_end_idx:val_end_idx + 1]

t_test_region = t[val_end_idx:]
y0_test_region = true_y[val_end_idx:val_end_idx + 1]
true_y_test_region = true_y[val_end_idx:]


def get_batch(batch_time=None):
    """
    Sample a short training window, starting at a random point WITHIN THE
    TRAIN REGION ONLY (t in [0, train_end_t]).

    `batch_time` can be overridden per-call to support an optional growing
    curriculum (see --batch_time_max): as training progresses, windows can
    grow from --batch_time up to --batch_time_max, so the model is
    increasingly exposed to longer, self-consistent rollouts rather than
    only ever seeing short, isolated windows. Diagnosed as necessary via
    diagnostic_rollout.py, which showed low error on isolated windows but
    error compounding sharply over continuous rollouts.

    Uses the window's true ABSOLUTE times as `batch_t` (not a shared
    relative-time axis starting at 0), since the wrapped network's weights
    depend on absolute time.
    """
    bt = batch_time if batch_time is not None else args.batch_time
    max_start = train_end_idx - bt
    if max_start < 1:
        raise ValueError(
            f"train_end_idx ({train_end_idx}) is too small for batch_time "
            f"({bt}) -- increase --data_size or --train_end_frac, or decrease "
            f"--batch_time / --batch_time_max."
        )
    s = np.random.randint(0, max_start)
    batch_y0 = true_y[s:s + 1]              # (1, D) -- single starting state
    batch_t = t[s:s + bt]                   # (T,) -- TRUE absolute times for this window
    batch_y = true_y[s:s + bt].unsqueeze(1)  # (T, 1, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-', label='true ')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--', label='predicted ')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label='true ')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--', label='predicted ')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


from tqdm import tqdm
import sys


class ProgressWrapperNet(nn.Module):
    def __init__(self, net, desc="Evaluating ODE"):
        super().__init__()
        self.net = net
        self.counter = 0
        self.desc = desc
        tqdm_kwargs = dict(file=sys.stdout, dynamic_ncols=True)
        self.pbar = tqdm(desc=self.desc, unit=" calls", **tqdm_kwargs)

    def forward(self, t, y):
        self.counter += 1
        if self.counter % 100000 == 0:
            elapsed = self.pbar.format_dict["elapsed"]
            rate = self.pbar.format_dict["rate"]
            formatted_time = tqdm.format_interval(elapsed)
            rate_str = f"{rate:.2f}" if rate else "?"
            print(f"[Eval] {self.counter} ODE calls [{formatted_time}, {rate_str} calls/s]", flush=True)
        return self.net(t, y)

    def close(self):
        self.pbar.set_description(f"{self.desc} (done, {self.counter} calls)")
        self.pbar.close()


if __name__ == '__main__':

    func = ODEFunc().to(device)  # stationary ODE vector field

    if args.basis == 'none':
        # Static / identity baseline: standard, time-INVARIANT neural ODE
        # (Chen et al.'s original formulation) -- no TimeParameterizedNet
        # wrapping. This is the reference point for Table 3's "static
        # (identity)" row and its DeltaError column.
        args.degree = 0  # placeholder, matches the MATLAB pipeline's convention for basis='none'
        print("Using STATIC (identity) baseline -- no time-parameterization.")
    else:
        # BUG FIX: tspan matches the TRUE absolute time domain (TSPAN, derived
        # from `t` above) instead of a hardcoded [0, 1]. TimeParameterizedNet
        # uses this to rescale absolute time into [0, 1] before evaluating its
        # polynomial/Legendre basis.
        # BUG FIX (degree off-by-one): TimeParameterizedNet's `d` parameter is
        # documented as the NUMBER OF BASIS FUNCTIONS, producing powers up to
        # t^(d-1) -- so a genuine degree-D polynomial (basis {1,...,t^D}, D+1
        # functions) needs d=D+1. This was previously called with d=args.degree
        # directly, so every "--degree 3" run only ever fit a degree-2
        # (quadratic) trajectory {1, t, t^2}, one degree short of what the
        # flag and every downstream table/figure label claimed. Verified via
        # checkpoint inspection: time_params shape was (3, 50, 2) under the
        # old call, confirming only 3 basis functions were ever used.
        func = TimeParameterizedNet(func, tspan=TSPAN, basis=args.basis, d=args.degree + 1).to(device)

    num_params = sum(p.numel() for p in func.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params}")

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    scheduler = None
    if args.lr_decay_every > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every,
                                               gamma=args.lr_decay_gamma)

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)  # diagnostic only: smoothed per-window train L1 loss

    # Track the best VALIDATION checkpoint (relative error over the val
    # region only). This checkpoint -- not the final-iteration weights -- is
    # what gets used for the final train/val/test reporting below.
    best_val_relerr = float('inf')
    best_state_dict = None
    best_iter = None
    best_val_nfe = None
    patience_counter = 0
    stopped_early = False
    stop_iter = None
    final_iter_reached = args.niters
    val_relerr = None
    nfe = None

    end = time.time()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        if args.batch_time_max is not None:
            grow_steps = args.batch_time_grow_steps
            if grow_steps is None:
                grow_steps = max(1, args.niters // (4 * args.batch_time_grow_every))
            growth_units_done = itr // args.batch_time_grow_every
            frac = min(1.0, growth_units_done / grow_steps)
            current_batch_time = int(round(args.batch_time + frac * (args.batch_time_max - args.batch_time)))
            current_batch_time = max(args.batch_time, min(current_batch_time, train_end_idx - 1))
        else:
            current_batch_time = args.batch_time

        batch_y0, batch_t, batch_y = get_batch(current_batch_time)

        start_time = time.time()

        pred_y = odeint(func, batch_y0, batch_t, rtol=args.rtol, atol=args.atol).to(device)
        data_loss = torch.mean(torch.abs(pred_y - batch_y))  # raw L1 on a single training window
        if args.alpha > 0:
            # R(theta) from Eq. (5): L2(0,T) norm of the time-varying weights for
            # basis in {monomial, legendre} (via TimeParameterizedNet.regularization_term,
            # using the closed-form Gram matrix -- Eq. 17); for basis=none (time-invariant
            # weights) this degenerates to the standard L2 norm of func's own parameters,
            # which is the D=0 case of the same expression (theta_NODE(t) = theta_0 for
            # all t, so \int_0^1 ||theta_NODE(t)||^2 dt = ||theta_0||^2 exactly).
            if args.basis == 'none':
                reg = sum(p.pow(2).sum() for p in func.parameters())
            else:
                reg = func.regularization_term()
            loss = data_loss + 0.5 * args.alpha * reg
        else:
            loss = data_loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(func.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        print(f"[{itr:04d}] Train Time: {(end_time - start_time) * 1000:.2f} ms | "
              f"Train Batch Loss (L1, diagnostic only): {loss.item():.4f}")
        if device.type == 'cuda':
            print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | "
                  f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                func_with_pbar = ProgressWrapperNet(func, desc="Evaluating VAL region")

                eval_start = time.time()
                pred_val = odeint(func_with_pbar.to(device), y0_val_region.to(device),
                                   t_val_region.to(device), rtol=args.rtol, atol=args.atol)
                val_relerr = relative_error(pred_val, true_y_val_region)
                eval_end = time.time()

                nfe = func_with_pbar.counter
                func_with_pbar.close()

                print('Iter {:04d} | Val RelErr: {:.6f} | Eval Time: {:.2f} ms | NFE: {} | '
                      'batch_time: {}'.format(
                          itr, val_relerr, (eval_end - eval_start) * 1000, nfe, current_batch_time))

                if val_relerr < best_val_relerr:
                    best_val_relerr = val_relerr
                    best_iter = itr
                    best_val_nfe = nfe
                    best_state_dict = {k: v.clone() for k, v in func.state_dict().items()}
                    patience_counter = 0
                    print(f'  New best val relerr: {best_val_relerr:.6f} (iter {itr})')
                else:
                    patience_counter += 1
                    if args.early_stop_patience > 0:
                        print(f'  No improvement ({patience_counter}/{args.early_stop_patience} '
                              f'checks since last best)')

                if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                    print(f'\nEarly stopping triggered at iter {itr} '
                          f'(no improvement for {args.early_stop_patience} consecutive '
                          f'val checks). Reporting best checkpoint from iter {best_iter}.')
                    stopped_early = True
                    stop_iter = itr
                    final_iter_reached = itr
                    break

        end = time.time()

    # ---- Load the best-on-validation checkpoint before final reporting ----
    if best_state_dict is not None:
        func.load_state_dict(best_state_dict)
    else:
        # Edge case: --niters < --test_freq, so no validation check ever ran.
        # Fall back to whatever the model currently is, and do one validation
        # check now so best_iter/best_val_relerr aren't left undefined.
        print("WARNING: no validation check occurred during training "
              "(--niters < --test_freq?) -- evaluating current weights once now.")
        with torch.no_grad():
            func_with_pbar = ProgressWrapperNet(func, desc="Evaluating VAL region (fallback)")
            pred_val = odeint(func_with_pbar.to(device), y0_val_region.to(device),
                               t_val_region.to(device), rtol=args.rtol, atol=args.atol)
            best_val_relerr = relative_error(pred_val, true_y_val_region)
            best_val_nfe = func_with_pbar.counter
            func_with_pbar.close()
        best_iter = final_iter_reached
        best_state_dict = {k: v.clone() for k, v in func.state_dict().items()}

    # ---- Final TRAIN and TEST region evaluation, on the SAME (best-on-val)
    #      checkpoint, using the identical relative-error metric -- so train/
    #      val/test are directly comparable, addressing 2.6.4 and 2.6.5 together. ----
    with torch.no_grad():
        train_pbar = ProgressWrapperNet(func, desc="Evaluating TRAIN region")
        pred_train = odeint(train_pbar.to(device), y0_train_region.to(device),
                             t_train_region.to(device), rtol=args.rtol, atol=args.atol)
        train_relerr = relative_error(pred_train, true_y_train_region)
        train_nfe = train_pbar.counter
        train_pbar.close()

        test_pbar = ProgressWrapperNet(func, desc="Evaluating TEST region (held out)")
        pred_test = odeint(test_pbar.to(device), y0_test_region.to(device),
                            t_test_region.to(device), rtol=args.rtol, atol=args.atol)
        test_relerr = relative_error(pred_test, true_y_test_region)
        test_nfe = test_pbar.counter
        test_pbar.close()

    print(f"\nTraining complete (stopped_early={stopped_early}, best_iter={best_iter}).")
    print(f"  TRAIN relerr: {train_relerr:.6f} (NFE: {train_nfe})")
    print(f"  VAL   relerr: {best_val_relerr:.6f} (NFE: {best_val_nfe})  [selection criterion]")
    print(f"  TEST  relerr: {test_relerr:.6f} (NFE: {test_nfe})  [held out, reported once]")

    best_ckpt_path = f"best_checkpoint_{args.basis}_d{args.degree}_seed{args.seed}.pt"
    torch.save({'state_dict': best_state_dict, 'iter': best_iter,
                'val_relerr': best_val_relerr, 'train_relerr': train_relerr,
                'test_relerr': test_relerr}, best_ckpt_path)
    print(f"Saved best checkpoint to {best_ckpt_path}")

    summary = {
        'basis': args.basis,
        'degree': args.degree,
        'seed': args.seed,
        'data_size': args.data_size,
        'niters_requested': args.niters,
        'train_end_frac': args.train_end_frac,
        'val_end_frac': args.val_end_frac,
        'train_end_t': train_end_t,
        'val_end_t': val_end_t,
        'batch_time': args.batch_time,
        'batch_time_max': args.batch_time_max,
        'batch_time_grow_every': args.batch_time_grow_every,
        'final_batch_time': current_batch_time,
        'grad_clip': args.grad_clip,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay_gamma': args.lr_decay_gamma,
        'early_stop_patience': args.early_stop_patience,
        'alpha': args.alpha,
        'rtol': args.rtol,
        'atol': args.atol,
        'stopped_early': stopped_early,
        'stop_iter': stop_iter,
        'best_iter': best_iter,
        'train_relerr': train_relerr,
        'train_nfe': train_nfe,
        'val_relerr': best_val_relerr,
        'val_nfe': best_val_nfe,
        'test_relerr': test_relerr,
        'test_nfe': test_nfe,
        'trainable_params': num_params,
        # Table 3 should use these three directly -- all computed on the
        # same best-on-validation checkpoint, all relative error.
        'reported_train_relerr': train_relerr,
        'reported_val_relerr': best_val_relerr,
        'reported_test_relerr': test_relerr,
        'reported_test_nfe': test_nfe,
        'reported_iter': best_iter,
    }
    results_path = args.results_json or f"results_{args.basis}_d{args.degree}_seed{args.seed}.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {results_path}")

# %%
# %%
