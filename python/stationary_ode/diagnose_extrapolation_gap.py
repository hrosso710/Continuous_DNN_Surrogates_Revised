"""
diagnose_extrapolation_gap.py

Produces a 3-panel figure documenting the investigation into why
Legendre-parameterized neural ODEs show elevated test-region error on the
stationary ODE task (Table 3), relative to monomial parameterization.

This consolidates the three diagnostics developed during that investigation:
  (1) ||theta_NODE(t)|| across train/val/test -- rules out weight blow-up.
  (2) Jacobian eigenvalues (max real part) along each model's own predicted
      test-region trajectory -- rules out chaotic/expanding instability.
  (3) Pointwise relative error growth over the test window, both models
      integrated forward from the TRUE state at t=20 -- shows the elevated
      error is a smooth, compounding directional bias, not an instantaneous
      mismatch or a divergence event.

Requires trained checkpoints named best_checkpoint_{basis}_d{degree}_seed{seed}.pt
in the current directory (produced by ode_demo_NEW.py). Run for a given
seed/degree pair, e.g.:
    python3 diagnose_extrapolation_gap.py --degree 3 --seed 0

Output: diagnostic_figure_d{degree}_seed{seed}.png
"""
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

sys.path.insert(0, '..')
from timeparam_NEW import TimeParameterizedNet  # noqa: E402

T0, T1 = 0.0, 25.0
TRAIN_END_T, VAL_END_T = 16.0, 20.0
DATA_SIZE = 500
true_y0 = torch.tensor([[2., 0.]])
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
t_grid = torch.linspace(T0, T1, DATA_SIZE)


class ODEFunc(torch.nn.Module):
    """Standalone copy of ode_demo_NEW.py's ODEFunc -- NOT imported from that
    module, which has no __main__ guard and runs its full training script as
    an import side effect."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 50), torch.nn.Tanh(), torch.nn.Linear(50, 2),
        )

    def forward(self, t, y):
        return self.net(y ** 3)


class LambdaLocal(torch.nn.Module):
    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


def get_true_trajectory():
    with torch.no_grad():
        return odeint(LambdaLocal(), true_y0, t_grid, method='rk4')


def build_basis(t_norm, basis, d):
    if basis == 'monomial':
        return torch.stack([t_norm ** i for i in range(d)])
    x = 2 * t_norm - 1
    a = [torch.tensor(1.0)]
    if d > 1:
        a.append(x)
    for n in range(2, d):
        a.append(((2 * n - 1) * x * a[-1] - (n - 1) * a[-2]) / n)
    return torch.stack(a)


def load_func(basis, degree, seed):
    ckpt = torch.load(f'best_checkpoint_{basis}_d{degree}_seed{seed}.pt',
                       map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    d = sd['time_params.net__0__weight'].shape[0]
    func = TimeParameterizedNet(ODEFunc(), tspan=[T0, T1], basis=basis, d=d)
    func.load_state_dict(sd)
    func.eval()
    return func, sd, d, ckpt


def panel1_weight_norm(ax, basis, sd, d, color):
    ts = torch.linspace(T0, T1, 250)
    norms = []
    for t in ts:
        t_norm = ((t - T0) / (T1 - T0)).clamp(0.0, 1.0)
        a = build_basis(t_norm, basis, d)
        total_sq = sum(torch.einsum('d,d...->...', a, sd[k]).pow(2).sum().item()
                        for k in sd if k.startswith('time_params.'))
        norms.append(total_sq ** 0.5)
    ax.plot(ts.numpy(), norms, label=basis, color=color, lw=2)


def panel2_jacobian(ax, basis, func, true_y, val_end_idx, color):
    y0_test = true_y[val_end_idx:val_end_idx + 1].squeeze(0)
    t_test = t_grid[val_end_idx:]
    with torch.no_grad():
        pred_test = odeint(func, y0_test, t_test, rtol=1e-7, atol=1e-9).squeeze(1)

    max_res = []
    for i in range(0, len(t_test), 5):
        y = pred_test[i].clone().requires_grad_(True)

        def f(y_in):
            return func(torch.tensor(t_test[i].item()), y_in.unsqueeze(0)).squeeze(0)

        J = torch.autograd.functional.jacobian(f, y)
        eigs = torch.linalg.eigvals(J)
        max_res.append(eigs.real.max().item())
    ts_sampled = t_test[::5].numpy()
    ax.plot(ts_sampled, max_res, label=basis, color=color, lw=2, marker='o', markersize=3)
    return pred_test, t_test


def panel3_error_growth(ax, basis, pred_test, t_test, true_y_test, color):
    num = (pred_test - true_y_test).pow(2).sum(dim=1).sqrt()
    den = true_y_test.pow(2).sum(dim=1).sqrt()
    errs = (num / den).detach().numpy()
    ax.plot((t_test - VAL_END_T).numpy(), errs, label=basis, color=color, lw=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--degree', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    true_y = get_true_trajectory()
    val_end_idx = int(0.80 * DATA_SIZE)
    true_y_test = true_y[val_end_idx:].squeeze(1)

    colors = {'monomial': 'tab:orange', 'legendre': 'tab:blue'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for basis in ['monomial', 'legendre']:
        func, sd, d, ckpt = load_func(basis, args.degree, args.seed)
        panel1_weight_norm(axes[0], basis, sd, d, colors[basis])
        pred_test, t_test = panel2_jacobian(axes[1], basis, func, true_y, val_end_idx, colors[basis])
        panel3_error_growth(axes[2], basis, pred_test, t_test, true_y_test, colors[basis])

    ax = axes[0]
    ax.axvline(TRAIN_END_T, color='gray', ls='--', lw=1)
    ax.axvline(VAL_END_T, color='gray', ls='--', lw=1)
    ax.text(TRAIN_END_T, ax.get_ylim()[1], ' train/val', va='top', fontsize=8, color='gray')
    ax.text(VAL_END_T, ax.get_ylim()[1], ' val/test', va='top', fontsize=8, color='gray')
    ax.set_xlabel('absolute time $t$')
    ax.set_ylabel(r'$\|\theta_{NODE}(t)\|$')
    ax.set_title('(a) Weight trajectory magnitude\n(no blow-up in either basis)')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('absolute time $t$ (test region)')
    ax.set_ylabel(r'max Re(eig($\partial f/\partial u$))')
    ax.set_title('(b) Local Jacobian stability\n(legendre more stable, not less)')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.set_xlabel('$t - 20$ (time since test region start)')
    ax.set_ylabel('pointwise relative error')
    ax.set_title('(c) Error growth over test window\n(smooth compounding, not a jump)')
    ax.legend(fontsize=8)

    fig.suptitle(f'Diagnosing the Legendre vs. monomial test-region error gap '
                 f'(degree={args.degree}, seed={args.seed})', fontsize=12, y=1.03)
    fig.tight_layout()
    out_path = f'diagnostic_figure_d{args.degree}_seed{args.seed}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
