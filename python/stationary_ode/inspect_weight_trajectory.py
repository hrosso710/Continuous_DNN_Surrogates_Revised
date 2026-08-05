"""
Diagnostic: evaluate ||theta_NODE(t)|| across the full absolute time domain
[0, 25] for a trained checkpoint, using the SAME basis-construction logic as
TimeParameterizedNet (duplicated here in numpy-free pure Python/torch so we
don't need to reconstruct the full model -- we only need the raw (d, *shape)
coefficient tensors from the checkpoint's state_dict).

Purpose: directly test whether the fitted weight trajectory does something
qualitatively different past the train/val boundary (t=16) or val/test
boundary (t=20) for legendre vs monomial -- i.e. whether the test-region
error gap traces back to visibly bad extrapolation in theta(t) itself,
as opposed to e.g. compounding ODE integration error from an otherwise
well-behaved theta(t).
"""
import torch
import sys

T0, T1 = 0.0, 25.0
TRAIN_END_T, VAL_END_T = 16.0, 20.0


def build_basis(t_norm, basis, d):
    if basis == 'monomial':
        return torch.stack([t_norm ** i for i in range(d)])
    elif basis == 'legendre':
        x = 2 * t_norm - 1
        a = [torch.tensor(1.0)]
        if d > 1:
            a.append(x)
        for n in range(2, d):
            Pn = ((2 * n - 1) * x * a[-1] - (n - 1) * a[-2]) / n
            a.append(Pn)
        return torch.stack(a)
    else:
        raise ValueError(basis)


def theta_norm_curve(ckpt_path, basis, n_grid=250):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    time_param_keys = [k for k in sd if k.startswith('time_params.')]
    d = sd[time_param_keys[0]].shape[0]

    ts = torch.linspace(T0, T1, n_grid)
    norms = []
    for t in ts:
        t_norm = ((t - T0) / (T1 - T0)).clamp(0.0, 1.0)
        a = build_basis(t_norm, basis, d)  # (d,)
        total_sq = 0.0
        for k in time_param_keys:
            P = sd[k]  # (d, *orig_shape)
            theta_t = torch.einsum('d,d...->...', a, P)  # weight matrix AT this t
            total_sq += theta_t.pow(2).sum().item()
        norms.append(total_sq ** 0.5)
    return ts.tolist(), norms, d


def summarize(basis, seed=0, degree=3):
    ckpt_path = f'best_checkpoint_{basis}_d{degree}_seed{seed}.pt'
    ts, norms, d = theta_norm_curve(ckpt_path, basis)
    print(f"\n=== {basis} (d={d} basis functions, i.e. degree {d-1}) ===")

    def region_stats(lo, hi, label):
        vals = [n for t, n in zip(ts, norms) if lo <= t <= hi]
        print(f"  {label:22s} t in [{lo:5.1f},{hi:5.1f}]  "
              f"min={min(vals):8.3f}  max={max(vals):8.3f}  "
              f"mean={sum(vals)/len(vals):8.3f}  range/mean={((max(vals)-min(vals))/(sum(vals)/len(vals))):6.2f}")

    region_stats(T0, TRAIN_END_T, "TRAIN (supervised)")
    region_stats(TRAIN_END_T, VAL_END_T, "VAL (early-stop only)")
    region_stats(VAL_END_T, T1, "TEST (never touched)")

    # Print a coarse trace so we can see the shape, not just min/max/mean
    print("  trace (every 10th grid point):")
    for i in range(0, len(ts), 10):
        marker = ""
        if abs(ts[i] - TRAIN_END_T) < 0.15:
            marker = "  <-- train/val boundary"
        elif abs(ts[i] - VAL_END_T) < 0.15:
            marker = "  <-- val/test boundary"
        print(f"    t={ts[i]:6.2f}  ||theta(t)||={norms[i]:8.3f}{marker}")


if __name__ == '__main__':
    for basis in ['monomial', 'legendre']:
        summarize(basis)
