"""
Diagnostic: for a trained checkpoint, integrate forward through the TEST
region (t in [20, 25]) starting from the true state at t=20 -- exactly like
the final evaluation in ode_demo_NEW.py -- but record the relative error at
EVERY time point along the way, not just the aggregate. This tells us
whether the elevated test error is:
  (a) already large immediately after t=20 (the instantaneous vector field
      f(u, t; theta(t)) is just wrong there, independent of integration), or
  (b) small right after t=20 and grows over the 5-time-unit window
      (compounding integration drift / instability), or
  (c) something in between / non-monotonic.
"""
import torch
import sys
from torchdiffeq import odeint

T0, T1 = 0.0, 25.0

# Regenerate the exact same ground truth trajectory ode_demo_NEW.py uses.
# NOTE: ode_demo_NEW.py seeds torch with args.seed BEFORE creating true_y0/
# true_A, so this must match seed=0 (the checkpoint we're inspecting) for
# the ground truth to line up.
true_y0 = torch.tensor([[2., 0.]])
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
DATA_SIZE = 500
t = torch.linspace(T0, T1, DATA_SIZE)


class ODEFunc(torch.nn.Module):
    """Copied standalone from ode_demo_NEW.py -- NOT imported, since that
    module runs its full training script as a side effect of import (no
    __main__ guard)."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 2),
        )
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y ** 3)


class LambdaLocal(torch.nn.Module):
    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


sys.path.insert(0, '..')
from timeparam_NEW import TimeParameterizedNet  # noqa: E402 (safe: no side effects)


class LambdaLocal(torch.nn.Module):
    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


def get_true_trajectory():
    with torch.no_grad():
        return odeint(LambdaLocal(), true_y0, t, method='rk4')


def relative_error_pointwise(pred, true):
    # matches the aggregate relative_error convention (||pred-true||/||true||)
    # but computed independently at each time point instead of over the whole window
    num = (pred - true).pow(2).sum(dim=(1, 2)).sqrt()
    den = true.pow(2).sum(dim=(1, 2)).sqrt()
    return (num / den).tolist()


def profile(basis, degree=3, seed=0):
    ckpt_path = f'best_checkpoint_{basis}_d{degree}_seed{seed}.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    d = sd['time_params.net__0__weight'].shape[0]

    base = ODEFunc()
    func = TimeParameterizedNet(base, tspan=[T0, T1], basis=basis, d=d)
    func.load_state_dict(sd)
    func.eval()

    true_y = get_true_trajectory()
    val_end_idx = int(0.80 * DATA_SIZE)  # matches --val_end_frac 0.80 default

    y0_test = true_y[val_end_idx:val_end_idx + 1].squeeze(0)  # (1,1,2) -> (1,2)
    t_test = t[val_end_idx:]
    true_y_test = true_y[val_end_idx:]

    with torch.no_grad():
        pred_test = odeint(func, y0_test, t_test, rtol=1e-7, atol=1e-9)

    errs = relative_error_pointwise(pred_test, true_y_test)
    print(f"\n=== {basis} test-region error profile (checkpoint test_relerr={ckpt['test_relerr']:.3f}) ===")
    for i in range(0, len(t_test), 10):
        print(f"  t={t_test[i].item():6.2f}  (t - 20 = {t_test[i].item()-20:5.2f})   pointwise relerr={errs[i]:8.4f}")
    print(f"  ...final point t={t_test[-1].item():.2f}: relerr={errs[-1]:.4f}")
    print(f"  first-point relerr: {errs[0]:.4f}   last-point relerr: {errs[-1]:.4f}   ratio last/first: {errs[-1]/max(errs[0],1e-8):.2f}")


if __name__ == '__main__':
    for basis in ['monomial', 'legendre']:
        profile(basis)
