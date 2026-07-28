"""
train_surrogate_node.py

Optimize-then-discretize neural ODE counterpart to the MATLAB
discretize-then-optimize pipeline (runExperiment_v2.m), for Table 5:
ELM/CDR/DCR surrogate modeling via a continuous-depth ODE.

Architecture mirrors the MATLAB side's block1/block2/readout structure,
swapping the discrete nt-step ResNet/Hamiltonian dynamics for a
continuous-depth ODE integrated via torchdiffeq:

    Y (params) --[Embed: Linear]--> z0
    z0 --[integrate dz/dt = f_theta(t,z), t in [0,T]]--> z(T)
    z(T) --[Readout: Linear]--> Chat (predicted targets)

f_theta is a plain MLP (no Hamiltonian structure -- confirmed 2026-07,
matching ode_demo_NEW.py's ODEFunc convention for the neural-ODE side).
When --basis != none, f_theta's weights are wrapped in TimeParameterizedNet
(timeparam_NEW.py) exactly as in ode_demo_NEW.py.

FIX (2026-07): CDR/DCR's raw Y is ~100x smaller in magnitude than a default
PyTorch nn.Linear embed layer can use with normal learning rates -- gradients
on the embed weight matrix are proportional to Y's magnitude, so with Y this
tiny they're swamped by the bias term's normal-scale gradients. Diagnosed via
a real-data test: the trained model's error matched the trivial "predict the
training mean" baseline almost exactly (CDR val relerr 0.181 baseline vs.
0.182 trained) -- confirming Y was contributing ~nothing. Fix: z-score
normalize Y using TRAIN statistics only (mean/std), applied to all splits.
This deviates from matlab/data/setupCDR.m and setupDCR.m's "raw, no
normalization" convention -- necessary because first-order gradient descent
needs reasonable input scale, unlike whatever conditioning MATLAB's own
optimizers (ADAM/GNvpro) provide there. ELM is unaffected (its paramrange
normalization already scales Y to [-1,1]; confirmed via the same
mean-baseline check that it's learning real signal, not stuck).

DATA-SPLIT NOTE: ELM (NNERDS.mat) uses the split baked into the .mat file
(idtrain/idval/idtest) -- reproducible exactly. CDR/DCR use MATLAB's
randperm under MATLAB's own RNG state, which is NOT reproducible bit-for-
bit from Python. This script instead draws its OWN independent 400/200/rest
random split via --seed and NumPy -- same convention (nTrain=400, nVal=200)
but NOT the identical partition MATLAB used. Confirmed acceptable 2026-07;
flag this explicitly if Table 5 is ever compared sample-for-sample against
a MATLAB run on the same dataset.

Usage:
    python3 train_surrogate_node.py --dataset CDR --basis legendre --degree 3 --seed 0 \
        --results_json results/results_table5_CDR_legendre_d3_seed0.json
"""

import argparse
import builtins
import os
import random
import sys

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from timeparam_NEW import TimeParameterizedNet


def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    return builtins.print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data loading -- mirrors matlab/data/setup{CDR,DCR,NNERDS}.m exactly for
# normalization; see module docstring for the split-reproducibility caveat.
# ---------------------------------------------------------------------------

def load_ELM(data_dir):
    """Mirrors matlab/data/setupNNERDS.m. Split is the exact one baked into
    NNERDS.mat (idtrain/idval/idtest), so this IS reproducible vs. MATLAB."""
    mat = sio.loadmat(os.path.join(data_dir, 'NNERDS.mat'))
    xall = mat['xall']              # (2486, 15)
    yall = mat['yall']              # (2486, 10)
    paramrange = mat['paramrange']  # (15, 2)
    # MATLAB indices are 1-indexed.
    idtrain = mat['idtrain'].flatten().astype(int) - 1
    idval = mat['idval'].flatten().astype(int) - 1
    idtest = mat['idtest'].flatten().astype(int) - 1

    pr_mean = paramrange.mean(axis=1)          # (15,)
    pr_range = paramrange[:, 1] - paramrange[:, 0]  # (15,)

    def norm_Y(Y):
        return (Y - pr_mean) * 2.0 / pr_range

    Ytrain, Yval, Ytest = norm_Y(xall[idtrain]), norm_Y(xall[idval]), norm_Y(xall[idtest])

    Ctrain_raw = yall[idtrain]
    minC = Ctrain_raw.min(axis=0)
    maxC = (Ctrain_raw - minC).max(axis=0)

    def norm_C(C):
        return (C - minC) / maxC

    Ctrain, Cval, Ctest = norm_C(Ctrain_raw), norm_C(yall[idval]), norm_C(yall[idtest])
    return Ytrain, Ctrain, Yval, Cval, Ytest, Ctest


def _random_split(Y, C, seed, nTrain=400, nVal=200):
    """400/200/rest random split via NumPy, matching MATLAB's convention in
    magnitude only -- see module docstring for why this isn't bit-identical
    to MATLAB's randperm-based split."""
    n = Y.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    if nTrain + nVal > n:
        print(f"[WARNING] nTrain+nVal ({nTrain + nVal}) > n ({n}) -- "
              f"falling back to 400/200 anyway per matlab convention; "
              f"this will error if n < 600.")
    idxTrain, idxVal, idxTest = idx[:nTrain], idx[nTrain:nTrain + nVal], idx[nTrain + nVal:]
    return (Y[idxTrain], C[idxTrain], Y[idxVal], C[idxVal], Y[idxTest], C[idxTest])


def load_CDR(data_dir, seed):
    """Mirrors matlab/data/setupCDR.m for C (raw, no normalization). Y is
    z-score normalized using TRAIN stats -- see module docstring: matlab's
    convention is raw Y too, but that starves gradient-based training here."""
    mat = sio.loadmat(os.path.join(data_dir, 'CDR_Data.mat'))
    Y, C = mat['Y'].T, mat['C'].T   # (800,55) / (800,72) -- samples x features
    Yt, Ct, Yv, Cv, Yte, Cte = _random_split(Y, C, seed)
    Ymean, Ystd = Yt.mean(axis=0), Yt.std(axis=0)
    return (Yt - Ymean) / Ystd, Ct, (Yv - Ymean) / Ystd, Cv, (Yte - Ymean) / Ystd, Cte


def load_DCR(data_dir, seed):
    """Mirrors matlab/data/setupDCR.m: C is mean-centered using the TRAIN
    split's mean (subtracted from train/val/test alike). Y is z-score
    normalized using TRAIN stats -- see module docstring / load_CDR."""
    mat = sio.loadmat(os.path.join(data_dir, 'DCR_Data.mat'))
    Y, C = mat['Y'].T, mat['C'].T   # (10000,3) / (10000,882)
    Yt, Ct, Yv, Cv, Ytest, Ctest = _random_split(Y, C, seed)
    meanTrain = Ct.mean(axis=0)
    Ymean, Ystd = Yt.mean(axis=0), Yt.std(axis=0)
    Yt_n, Yv_n, Ytest_n = (Yt - Ymean) / Ystd, (Yv - Ymean) / Ystd, (Ytest - Ymean) / Ystd
    return Yt_n, Ct - meanTrain, Yv_n, Cv - meanTrain, Ytest_n, Ctest - meanTrain


LOADERS = {'ELM': load_ELM, 'CDR': load_CDR, 'DCR': load_DCR}


# ---------------------------------------------------------------------------
# Model -- plain-MLP dynamics (no Hamiltonian structure), continuous-depth
# ODE via torchdiffeq, optionally time-parameterized via TimeParameterizedNet.
# ---------------------------------------------------------------------------

class ODEFunc(nn.Module):
    """Plain MLP vector field dz/dt = f(z). Ignores t directly (as in
    ode_demo_NEW.py's ODEFunc) -- time-dependence, when present, comes
    entirely from TimeParameterizedNet varying this net's weights over t."""

    def __init__(self, nc, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nc, hidden),
            nn.Tanh(),
            nn.Linear(hidden, nc),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        return self.net(z)


class SurrogateODENet(nn.Module):
    """Embed -> integrate dz/dt=f_theta(t,z), t in [0,T] -> Readout."""

    def __init__(self, input_dim, output_dim, nc, hidden, T, basis, degree,
                 odeint_fn, method, rtol, atol):
        super().__init__()
        self.embed = nn.Linear(input_dim, nc)
        func = ODEFunc(nc, hidden)
        if basis == 'none':
            self.func = func
        else:
            self.func = TimeParameterizedNet(func, tspan=[0.0, T], basis=basis, d=degree)
        self.readout = nn.Linear(nc, output_dim)
        self.T = T
        self.odeint_fn = odeint_fn
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, Y, func_override=None):
        func = func_override if func_override is not None else self.func
        z0 = self.embed(Y)
        t_span = torch.tensor([0.0, self.T], device=Y.device, dtype=z0.dtype)
        zT = self.odeint_fn(func, z0, t_span, method=self.method,
                             rtol=self.rtol, atol=self.atol)[-1]
        return self.readout(zT)


class ProgressWrapperNet(nn.Module):
    """Counts ODE function calls, for NFE reporting -- same pattern as
    ode_demo_NEW.py."""

    def __init__(self, net, desc="Evaluating ODE"):
        super().__init__()
        self.net = net
        self.counter = 0
        self.desc = desc
        self.pbar = tqdm(desc=self.desc, unit=" calls", file=sys.stdout, dynamic_ncols=True)

    def forward(self, t, y):
        self.counter += 1
        return self.net(t, y)

    def close(self):
        self.pbar.set_description(f"{self.desc} (done, {self.counter} calls)")
        self.pbar.close()


def relative_error(pred, true, eps=1e-12):
    """||pred-true||_2 / ||true||_2 -- same convention as MATLAB's
    relErrTrain/Val/Test and ode_demo_NEW.py's relative_error."""
    num = torch.sqrt(torch.sum((pred - true) ** 2))
    den = torch.sqrt(torch.sum(true ** 2)) + eps
    return (num / den).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ELM', 'CDR', 'DCR'], required=True)
    parser.add_argument('--data_dir', type=str, default='../matlab/data',
                         help='Directory containing NNERDS.mat / CDR_Data.mat / DCR_Data.mat')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--atol', type=float, default=1e-9)

    parser.add_argument('--basis', choices=['legendre', 'monomial', 'none'], default='legendre')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--nc', type=int, default=15, help='Embedding / dynamics width, matches MATLAB nc=15')
    parser.add_argument('--hidden', type=int, default=50, help='ODEFunc hidden width')
    parser.add_argument('--T', type=float, default=1.0, help='Integration horizon, confirmed T=1')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-2,
                         help='Default changed 2026-07 from 1e-3 after diagnostic showed '
                              '1e-3 was too small even with normalized Y.')
    parser.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='adam',
                         help='Default changed 2026-07 from rmsprop -- validated on real '
                              'CDR data with normalized Y (see module docstring).')
    parser.add_argument('--test_freq', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--lr_decay_every', type=int, default=0)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--early_stop_patience', type=int, default=0)

    parser.add_argument('--results_json', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint_fn
    else:
        from torchdiffeq import odeint as odeint_fn

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(f"The device being used is {device}.")
    print(f"Dataset: {args.dataset} | Seed: {args.seed} | Basis: {args.basis} | "
          f"Degree: {args.degree} | T: {args.T}")

    Ytr, Ctr, Yv, Cv, Yte, Cte = LOADERS[args.dataset](args.data_dir, args.seed) \
        if args.dataset != 'ELM' else LOADERS[args.dataset](args.data_dir)
    print(f"Split sizes: train={Ytr.shape[0]} val={Yv.shape[0]} test={Yte.shape[0]} "
          f"| input_dim={Ytr.shape[1]} output_dim={Ctr.shape[1]}")

    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Ytr, Ctr, Yv, Cv, Yte, Cte = to_t(Ytr), to_t(Ctr), to_t(Yv), to_t(Cv), to_t(Yte), to_t(Cte)

    if args.basis == 'none':
        args.degree = 0  # placeholder, matches MATLAB/ode_demo_NEW.py convention for basis='none'
        print("Using STATIC (identity) baseline -- no time-parameterization.")

    model = SurrogateODENet(
        input_dim=Ytr.shape[1], output_dim=Ctr.shape[1], nc=args.nc, hidden=args.hidden,
        T=args.T, basis=args.basis, degree=args.degree, odeint_fn=odeint_fn,
        method=args.method, rtol=args.rtol, atol=args.atol,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params}")

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_decay_every > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every,
                                               gamma=args.lr_decay_gamma)

    n_train = Ytr.shape[0]
    best_val_relerr = float('inf')
    best_state_dict = None
    best_iter = None
    no_improve_checks = 0
    stopped_early = False
    stop_iter = None

    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()
        idx = np.random.randint(0, n_train, size=min(args.batch_size, n_train))
        Yb, Cb = Ytr[idx], Ctr[idx]
        Cb_hat = model(Yb)
        loss = torch.mean(torch.abs(Cb_hat - Cb))  # L1, diagnostic only
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if itr % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                Cv_hat = model(Yv)
                val_relerr = relative_error(Cv_hat, Cv)
            print(f"Iter {itr:04d} | Val RelErr: {val_relerr:.6f}")

            if val_relerr < best_val_relerr:
                best_val_relerr = val_relerr
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                best_iter = itr
                no_improve_checks = 0
            else:
                no_improve_checks += 1
                if args.early_stop_patience > 0:
                    print(f"  No improvement ({no_improve_checks}/{args.early_stop_patience} checks since last best)")
                    if no_improve_checks >= args.early_stop_patience:
                        stopped_early = True
                        stop_iter = itr
                        print(f"\nEarly stopping triggered at iter {itr} "
                              f"(no improvement for {args.early_stop_patience} consecutive val checks). "
                              f"Reporting best checkpoint from iter {best_iter}.")
                        break

    if best_state_dict is None:
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        best_iter = args.niters
    model.load_state_dict(best_state_dict)
    model.eval()

    def eval_region(Y, C, desc):
        wrapped = ProgressWrapperNet(model.func, desc=desc)
        with torch.no_grad():
            relerr = relative_error(model(Y, func_override=wrapped), C)
        nfe = wrapped.counter
        wrapped.close()
        return relerr, nfe

    train_relerr, train_nfe = eval_region(Ytr, Ctr, "Evaluating TRAIN region")
    val_relerr, val_nfe = eval_region(Yv, Cv, "Evaluating VAL region")
    test_relerr, test_nfe = eval_region(Yte, Cte, "Evaluating TEST region (held out)")

    print(f"\nTraining complete (stopped_early={stopped_early}, best_iter={best_iter}).")
    print(f"  TRAIN relerr: {train_relerr:.6f} (NFE: {train_nfe})")
    print(f"  VAL   relerr: {val_relerr:.6f} (NFE: {val_nfe})  [selection criterion]")
    print(f"  TEST  relerr: {test_relerr:.6f} (NFE: {test_nfe})  [held out, reported once]")

    summary = {
        'dataset': args.dataset,
        'basis': args.basis,
        'degree': args.degree,
        'seed': args.seed,
        'nc': args.nc,
        'hidden': args.hidden,
        'T': args.T,
        'niters_requested': args.niters,
        'batch_size': args.batch_size,
        'grad_clip': args.grad_clip,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay_gamma': args.lr_decay_gamma,
        'early_stop_patience': args.early_stop_patience,
        'rtol': args.rtol,
        'atol': args.atol,
        'stopped_early': stopped_early,
        'stop_iter': stop_iter,
        'best_iter': best_iter,
        'trainable_params': num_params,
        'reported_train_relerr': train_relerr,
        'reported_val_relerr': val_relerr,
        'reported_test_relerr': test_relerr,
        'reported_train_nfe': train_nfe,
        'reported_test_nfe': test_nfe,
        'reported_iter': best_iter,
    }
    results_path = args.results_json or f"results_table5_{args.dataset}_{args.basis}_d{args.degree}_seed{args.seed}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {results_path}")


if __name__ == '__main__':
    main()
