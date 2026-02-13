"""
Seed/thread sweep benchmark producing averaged curves.

We average loss curves across seeds and visualize the effect of Torch threading
for the competitor (NNEinFact). Our NumPy methods are run once per seed under a
fixed thread env configuration.

Outputs:
  - mean loss vs outer iteration
  - mean loss vs wall time (via interpolation to a common time grid)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _set_thread_env(n: Optional[int]):
    """
    Best-effort: set common thread env vars BEFORE importing numpy/torch.

    Parameters
    ----------
    n:
        Number of threads to request for BLAS/Accelerate/OpenMP backends.
        If None, does nothing.
    """
    if n is None:
        return
    n = int(n)
    for k in (
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[k] = str(n)


@dataclass
class Curve:
    """
    Convenience container for a single method run.

    Attributes
    ----------
    loss:
        Mean loss history (typically length iters+1).
    tt:
        Time history in seconds (same length as `loss`).
    """
    loss: List[float]
    tt: List[float]


def _prepend_initial(loss: Sequence[float], tt: Sequence[float], loss0: float) -> Curve:
    """Make curves length iters+1 by prepending (t=0, loss0)."""
    loss2 = [float(loss0)] + [float(x) for x in loss]
    tt2 = [0.0] + [float(x) for x in tt]
    return Curve(loss2, tt2)


def _stack_mean_std(arrs: List[Sequence[float]]) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Compute elementwise mean and std across multiple aligned 1D arrays.

    Parameters
    ----------
    arrs:
        List of 1D sequences of equal length.

    Returns
    -------
    mean, std:
        Arrays of the same length as the input sequences.
    """
    import numpy as np

    A = np.asarray(arrs, dtype=float)
    return np.nanmean(A, axis=0), np.nanstd(A, axis=0)


def _interp_to_grid(tt: "np.ndarray", loss: "np.ndarray", grid: "np.ndarray") -> "np.ndarray":
    """
    Interpolate a (time, loss) curve onto a common time grid.

    Non-finite values are removed before interpolation. Values outside the
    original time range are set to NaN.

    Parameters
    ----------
    tt:
        1D time array (seconds).
    loss:
        1D loss array (same length as `tt`).
    grid:
        1D target time grid (seconds).

    Returns
    -------
    ndarray
        Interpolated loss values on `grid` (NaN outside the original range).
    """
    import numpy as np

    mask = np.isfinite(tt) & np.isfinite(loss)
    tt = tt[mask]
    loss = loss[mask]
    if tt.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    # Ensure increasing time (should already be).
    order = np.argsort(tt)
    tt = tt[order]
    loss = loss[order]
    # np.interp extrapolates with endpoints by default; we prefer NaN outside range.
    out = np.interp(grid, tt, loss)
    out[(grid < tt[0]) | (grid > tt[-1])] = np.nan
    return out


def _make_time_grid(curves: List[Curve], n_points: int = 200) -> "np.ndarray":
    """
    Build a common time grid shared across curves for mean±std aggregation.

    The grid spans `[0, t_max]` where `t_max` is the minimum of each curve's
    final time, so that every curve can contribute over the full grid.

    Parameters
    ----------
    curves:
        List of curves to align.
    n_points:
        Number of grid points.

    Returns
    -------
    ndarray
        1D time grid (seconds).
    """
    import numpy as np

    # Common grid up to the minimum max-time across curves so everyone contributes.
    t_max = min(float(max(c.tt)) for c in curves if c.tt)
    return np.linspace(0.0, t_max, n_points, dtype=float)


def _plot_mean_curves(
    curves_by_method: Dict[str, List[Curve]],
    title: str,
    ylabel: str,
    yscale: str,
    start_iter: int,
    save_dir: Optional[Path],
    filename: str,
    fmt: Tuple[str, ...],
    dpi: int,
):
    """
    Plot mean±std curves vs iteration and vs time and save to disk.

    Parameters
    ----------
    curves_by_method:
        Dict mapping method name -> list of `Curve` objects (one per seed).
    title:
        Figure title prefix.
    ylabel:
        Y-axis label (e.g. "mean loss").
    yscale:
        Matplotlib y-axis scale ("linear" or "log").
    start_iter:
        Number of initial points to skip in the iteration plot.
    save_dir:
        Directory to save figures into. If None, no files are saved.
    filename:
        Output filename stem (no extension).
    fmt:
        Tuple of extensions, e.g. ("png","pdf").
    dpi:
        DPI for raster outputs.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 2, 1)
    for name, curves in curves_by_method.items():
        L = [np.asarray(c.loss, dtype=float) for c in curves]
        # align by truncating to min length (should all match)
        m = min(len(x) for x in L)
        L = [x[:m] for x in L]
        mean, std = _stack_mean_std(L)
        x = np.arange(m, dtype=int)
        s = int(start_iter)
        x = x[s:]
        mean = mean[s:]
        std = std[s:]
        ax1.plot(x, mean, label=name, linewidth=2)
        ax1.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax1.set_title(title + " — mean loss vs iter")
    ax1.set_xlabel("outer iteration")
    ax1.set_ylabel(ylabel)
    ax1.set_yscale(yscale)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    # Build a shared time grid per method group; to compare fairly, use a global grid
    # across all methods.
    all_curves = [c for curves in curves_by_method.values() for c in curves]
    grid = _make_time_grid(all_curves, n_points=250)

    for name, curves in curves_by_method.items():
        Ys = []
        for c in curves:
            tt = np.asarray(c.tt, dtype=float)
            loss = np.asarray(c.loss, dtype=float)
            Ys.append(_interp_to_grid(tt, loss, grid))
        mean, std = _stack_mean_std(Ys)
        ax2.plot(grid, mean, label=name, linewidth=2)
        ax2.fill_between(grid, mean - std, mean + std, alpha=0.15)

    ax2.set_title(title + " — mean loss vs time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(ylabel)
    ax2.set_yscale(yscale)
    ax2.legend()

    fig.tight_layout()

    saved = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for ext in fmt:
            ext = ext.lstrip(".").lower()
            out = save_dir / f"{filename}.{ext}"
            fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
            saved.append(out)
        print("Saved figures:")
        for p in saved:
            print("  ", p)

    plt.close(fig)


def main():
    """
    Entry point for the multi-seed sweep benchmark.

    Runs multiple random seeds and produces a single figure with mean±std bands.
    If the competitor is enabled, overlays multiple NNEinFact curves with
    different `torch.set_num_threads(...)` values to visualize the threading effect.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["cp", "tucker"], default="tucker")
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-12)

    ap.add_argument("--shape", type=int, nargs="+", default=[60, 52, 44, 36])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 7, 6, 5])

    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)

    ap.add_argument("--threads", type=int, default=1, help="Set BLAS/Accelerate thread env vars.")
    ap.add_argument("--torch_threads_list", type=int, nargs="+", default=[1, 4, 8])

    ap.add_argument("--run_einfact", action="store_true")
    ap.add_argument("--einfact_path", type=str, default="einfact.py")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--einfact_iters", type=int, default=None)

    ap.add_argument("--save_dir", type=str, default="figures")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fmt", type=str, nargs="+", default=["png", "pdf"])
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--start_iter", type=int, default=1)

    args = ap.parse_args()

    # Thread env BEFORE imports
    _set_thread_env(args.threads)

    # headless-friendly backend
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np
    from comm.core import beta_divergence, to_float32

    beta = float(args.beta)
    eps = float(args.eps)
    shape = tuple(int(x) for x in args.shape)
    n_seeds = int(args.n_seeds)
    seed0 = int(args.seed0)
    iters = int(args.iters)
    inner = int(args.inner)

    curves: Dict[str, List[Curve]] = {}

    def add_curve(name: str, c: Curve):
        curves.setdefault(name, []).append(c)

    for s in range(seed0, seed0 + n_seeds):
        rng = np.random.default_rng(int(s))

        if args.problem == "cp":
            from comm.models.cp import cp_bcomm, cp_jcomm, cp_mu_unfolding, cp_reconstruct
            R = int(args.rank)
            N = len(shape)

            A_true = [rng.random((shape[n], R), dtype=np.float32) + 1e-3 for n in range(N)]
            X = cp_reconstruct(A_true) + np.float32(1e-3)
            A0 = [rng.random((shape[n], R), dtype=np.float32) + 1e-3 for n in range(N)]
            X = to_float32(X)
            A0 = to_float32(A0)

            Xhat0 = np.maximum(cp_reconstruct(A0), eps)
            loss0 = beta_divergence(X, Xhat0, beta) / X.size

            _, lb, tb = cp_bcomm(X, A0, beta=beta, n_outer=iters, eps=eps)
            _, lj, tj = cp_jcomm(X, A0, beta=beta, n_outer=iters, n_inner=inner, eps=eps)
            _, lmu, tmu = cp_mu_unfolding(X, A0, beta=beta, n_outer=iters, eps=eps)

            add_curve("B-CoMM", _prepend_initial(lb / X.size, tb, loss0))
            add_curve("J-CoMM", _prepend_initial(lj / X.size, tj, loss0))
            add_curve("MU-unfold", _prepend_initial(lmu / X.size, tmu, loss0))

            if args.run_einfact:
                from comm.competitors.einfact_wrapper import run_nneinfact_cp
                import torch

                alpha_e = 1.0
                beta_e = beta - 1.0
                einfact_iters = iters if args.einfact_iters is None else int(args.einfact_iters)

                for tt_n in args.torch_threads_list:
                    torch.set_num_threads(int(tt_n))
                    le, te = run_nneinfact_cp(
                        args.einfact_path,
                        X,
                        A0,
                        alpha=alpha_e,
                        beta_ab=beta_e,
                        max_iter=einfact_iters,
                        device=args.device,
                        disable_valsplit=True,
                        seed=int(s),
                    )
                    add_curve(f"NNEinFact (torch={int(tt_n)})", Curve(le.tolist(), te.tolist()))

        else:
            from comm.models.tucker import tucker_bcomm, tucker_jcomm, tucker_mu_unfolding, tucker_reconstruct
            ranks = tuple(int(x) for x in args.ranks)
            if len(ranks) != len(shape):
                raise ValueError("Length of --ranks must match length of --shape.")
            N = len(shape)

            G_true = rng.random(ranks, dtype=np.float32) + 1e-3
            A_true = [rng.random((shape[n], ranks[n]), dtype=np.float32) + 1e-3 for n in range(N)]
            X = tucker_reconstruct(G_true, A_true) + np.float32(1e-3)

            G0 = rng.random(ranks, dtype=np.float32) + 1e-3
            A0 = [rng.random((shape[n], ranks[n]), dtype=np.float32) + 1e-3 for n in range(N)]

            X = to_float32(X)
            A0 = to_float32(A0)
            G0 = to_float32(G0)

            Xhat0 = np.maximum(tucker_reconstruct(G0, A0), eps)
            loss0 = beta_divergence(X, Xhat0, beta) / X.size

            _, _, lb, tb = tucker_bcomm(X, G0, A0, beta=beta, n_outer=iters, eps=eps)
            _, _, lj, tj = tucker_jcomm(X, G0, A0, beta=beta, n_outer=iters, n_inner=inner, eps=eps)
            _, _, lmu, tmu = tucker_mu_unfolding(X, G0, A0, beta=beta, n_outer=iters, eps=eps)

            add_curve("B-CoMM", _prepend_initial(lb / X.size, tb, loss0))
            add_curve("J-CoMM", _prepend_initial(lj / X.size, tj, loss0))
            add_curve("MU-unfold", _prepend_initial(lmu / X.size, tmu, loss0))

            if args.run_einfact:
                from comm.competitors.einfact_wrapper import run_nneinfact_tucker
                import torch

                alpha_e = 1.0
                beta_e = beta - 1.0
                einfact_iters = iters if args.einfact_iters is None else int(args.einfact_iters)

                for tt_n in args.torch_threads_list:
                    torch.set_num_threads(int(tt_n))
                    le, te = run_nneinfact_tucker(
                        args.einfact_path,
                        X,
                        A0,
                        G0,
                        alpha=alpha_e,
                        beta_ab=beta_e,
                        max_iter=einfact_iters,
                        device=args.device,
                        disable_valsplit=True,
                        seed=int(s),
                    )
                    add_curve(f"NNEinFact (torch={int(tt_n)})", Curve(le.tolist(), te.tolist()))

    yscale = "log" if args.logy else "linear"
    title = f"{args.problem.upper()} sweep — beta={beta}, seeds={seed0}..{seed0+n_seeds-1}, threads={int(args.threads)}"
    fname = (
        f"sweep_{args.problem}_beta{beta}_shape{'x'.join(map(str, shape))}"
        f"_seeds{seed0}-{seed0+n_seeds-1}_threads{int(args.threads)}"
    )
    if args.problem == "cp":
        fname += f"_R{int(args.rank)}"
    else:
        fname += f"_ranks{'x'.join(map(str, map(int, args.ranks)))}"

    save_dir = None if args.no_save else Path(args.save_dir)

    _plot_mean_curves(
        curves_by_method=curves,
        title=title,
        ylabel="mean loss",
        yscale=yscale,
        start_iter=int(args.start_iter),
        save_dir=save_dir,
        filename=fname,
        fmt=tuple(args.fmt),
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()

