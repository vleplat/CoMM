"""
Apples-to-apples benchmark runner.

Goal: compare our NumPy+opt_einsum methods vs Torch competitor (NNEinFact)
under controlled CPU threading and using comparable metrics (mean loss, time-to-target).

This script is intentionally non-plotting to avoid GUI backend issues on macOS.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _set_thread_env(n: Optional[int]):
    """
    Best-effort: set common thread env vars BEFORE importing numpy/torch.
    On macOS/Accelerate, VECLIB_MAXIMUM_THREADS is often the one that matters.
    """
    if n is None:
        return
    n = int(n)
    keys = [
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    for k in keys:
        os.environ[k] = str(n)


def _time_to_reach(loss: Sequence[float], tt: Sequence[float], target: float) -> float:
    """
    Return the first timestamp at which `loss` reaches `target` (<= target).

    Non-finite entries are ignored.

    Parameters
    ----------
    loss:
        Loss history (mean loss).
    tt:
        Time history in seconds (same length as `loss`).
    target:
        Target loss threshold.

    Returns
    -------
    float
        The first time `tt[k]` such that `loss[k] <= target`, or NaN if never reached.
    """
    import numpy as np

    loss = np.asarray(loss, dtype=float)
    tt = np.asarray(tt, dtype=float)
    mask = np.isfinite(loss) & np.isfinite(tt)
    idx = np.where(mask & (loss <= float(target)))[0]
    return float(tt[int(idx[0])]) if idx.size else float("nan")


def _last_finite(loss: Sequence[float], tt: Sequence[float]) -> Tuple[float, float]:
    """
    Get the last finite (loss, time) pair from histories.

    Parameters
    ----------
    loss:
        Loss history.
    tt:
        Time history.

    Returns
    -------
    (float, float)
        `(loss_last, time_last)` for the last index with finite values, else `(nan, nan)`.
    """
    import numpy as np

    loss = np.asarray(loss, dtype=float)
    tt = np.asarray(tt, dtype=float)
    mask = np.isfinite(loss) & np.isfinite(tt)
    if not mask.any():
        return float("nan"), float("nan")
    j = int(np.where(mask)[0][-1])
    return float(loss[j]), float(tt[j])


@dataclass
class RunResult:
    name: str
    loss_mean: List[float]
    time_s: List[float]

    def summary(self) -> Dict[str, float]:
        """
        Summarize the run with last finite values.

        Returns
        -------
        dict
            Keys: `last_loss_mean`, `last_time_s`, `n_points`.
        """
        last_loss, last_t = _last_finite(self.loss_mean, self.time_s)
        return {"last_loss_mean": last_loss, "last_time_s": last_t, "n_points": float(len(self.loss_mean))}


def _print_table(rows: List[Dict[str, str]]):
    """
    Print a simple aligned text table to stdout.

    Parameters
    ----------
    rows:
        List of dicts with identical keys (column names).
    """
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    def _line(values: Dict[str, str]) -> str:
        return "  ".join([str(values.get(c, "")).ljust(widths[c]) for c in cols])

    print(_line({c: c for c in cols}))
    print(_line({c: "-" * widths[c] for c in cols}))
    for r in rows:
        print(_line({c: str(r.get(c, "")) for c in cols}))


def main():
    """
    Entry point for the apples-to-apples benchmark.

    This script focuses on *fairness*:
    - controls BLAS/Accelerate threads via env vars (best-effort)
    - controls Torch CPU threads
    - reports both final metrics and time-to-target metrics
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["cp", "tucker"], default="tucker")
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--shape", type=int, nargs="+", default=[60, 52, 44, 36])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 7, 6, 5])

    ap.add_argument("--threads", type=int, default=1, help="Set BLAS/Accelerate thread env vars.")
    ap.add_argument("--torch_threads", type=int, default=1, help="torch.set_num_threads(n).")

    ap.add_argument("--run_einfact", action="store_true")
    ap.add_argument("--einfact_path", type=str, default="einfact.py")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--einfact_iters", type=int, default=None)

    ap.add_argument(
        "--target_rel",
        type=float,
        nargs="+",
        default=[0.9, 0.7, 0.5, 0.3],
        help="Targets as multiples of initial mean loss (e.g., 0.5 means 50% of init).",
    )

    ap.add_argument("--save_json", type=str, default=None, help="Optional path to save results JSON.")

    args = ap.parse_args()

    # IMPORTANT: set threading env vars BEFORE importing numpy/torch/comm.
    _set_thread_env(args.threads)

    import numpy as np
    from comm.core import beta_divergence, to_float32

    beta = float(args.beta)
    eps = float(args.eps)
    shape = tuple(args.shape)
    rng = np.random.default_rng(int(args.seed))

    results: List[RunResult] = []

    if args.problem == "cp":
        from comm.models.cp import cp_bcomm, cp_jcomm, cp_mu_unfolding, cp_reconstruct

        R = int(args.rank)
        N = len(shape)

        A_true = [rng.random((shape[n], R), dtype=np.float32) + 1e-3 for n in range(N)]
        X = cp_reconstruct(A_true) + np.float32(1e-3)
        A0 = [rng.random((shape[n], R), dtype=np.float32) + 1e-3 for n in range(N)]

        X = to_float32(X)
        A0 = to_float32(A0)

        # initial mean loss
        Xhat0 = np.maximum(cp_reconstruct(A0), eps)
        loss0 = beta_divergence(X, Xhat0, beta) / X.size

        _, loss_b, t_b = cp_bcomm(X, A0, beta=beta, n_outer=int(args.iters), eps=eps)
        _, loss_j, t_j = cp_jcomm(X, A0, beta=beta, n_outer=int(args.iters), n_inner=int(args.inner), eps=eps)
        _, loss_mu, t_mu = cp_mu_unfolding(X, A0, beta=beta, n_outer=int(args.iters), eps=eps)

        results.extend(
            [
                RunResult("B-CoMM", (loss_b / X.size).tolist(), t_b.tolist()),
                RunResult("J-CoMM", (loss_j / X.size).tolist(), t_j.tolist()),
                RunResult("MU-unfold", (loss_mu / X.size).tolist(), t_mu.tolist()),
            ]
        )

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

        _, _, loss_b, t_b = tucker_bcomm(X, G0, A0, beta=beta, n_outer=int(args.iters), eps=eps)
        _, _, loss_j, t_j = tucker_jcomm(X, G0, A0, beta=beta, n_outer=int(args.iters), n_inner=int(args.inner), eps=eps)
        _, _, loss_mu, t_mu = tucker_mu_unfolding(X, G0, A0, beta=beta, n_outer=int(args.iters), eps=eps)

        results.extend(
            [
                RunResult("B-CoMM", (loss_b / X.size).tolist(), t_b.tolist()),
                RunResult("J-CoMM", (loss_j / X.size).tolist(), t_j.tolist()),
                RunResult("MU-unfold", (loss_mu / X.size).tolist(), t_mu.tolist()),
            ]
        )

    # competitor
    if args.run_einfact:
        import torch

        torch.set_num_threads(int(args.torch_threads))

        alpha_einfact = 1.0
        beta_einfact = float(beta) - 1.0
        einfact_iters = int(args.iters) if args.einfact_iters is None else int(args.einfact_iters)

        if args.problem == "cp":
            from comm.competitors.einfact_wrapper import run_nneinfact_cp

            loss_e, t_e = run_nneinfact_cp(
                args.einfact_path,
                X,  # type: ignore[name-defined]
                A0,  # type: ignore[name-defined]
                alpha=alpha_einfact,
                beta_ab=beta_einfact,
                max_iter=einfact_iters,
                device=args.device,
                disable_valsplit=True,
                seed=int(args.seed),
            )
        else:
            from comm.competitors.einfact_wrapper import run_nneinfact_tucker

            loss_e, t_e = run_nneinfact_tucker(
                args.einfact_path,
                X,  # type: ignore[name-defined]
                A0,  # type: ignore[name-defined]
                G0,  # type: ignore[name-defined]
                alpha=alpha_einfact,
                beta_ab=beta_einfact,
                max_iter=einfact_iters,
                device=args.device,
                disable_valsplit=True,
                seed=int(args.seed),
            )

        results.append(RunResult("NNEinFact", loss_e.tolist(), t_e.tolist()))

    # report
    print(f"problem={args.problem}  beta={beta}  shape={shape}")
    if args.problem == "cp":
        print(f"rank={int(args.rank)}")
    else:
        print(f"ranks={tuple(int(x) for x in args.ranks)}")
    print(f"threads={int(args.threads)}  torch_threads={int(args.torch_threads)}  seed={int(args.seed)}")
    print(f"initial_mean_loss={loss0:.6e}")

    # summary (final)
    rows = []
    for rr in results:
        s = rr.summary()
        rows.append(
            {
                "method": rr.name,
                "last_loss": f"{s['last_loss_mean']:.3e}",
                "last_time(s)": f"{s['last_time_s']:.3f}",
                "points": str(int(s["n_points"])),
            }
        )
    _print_table(rows)

    # time-to-target
    targets = [float(loss0) * float(r) for r in args.target_rel]
    rows = []
    for rr in results:
        row = {"method": rr.name}
        for rel, tgt in zip(args.target_rel, targets):
            row[f"t@{rel:g}x"] = f"{_time_to_reach(rr.loss_mean, rr.time_s, tgt):.3f}"
        rows.append(row)
    print("\nTime-to-target (seconds):")
    _print_table(rows)

    # save
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "problem": args.problem,
                "beta": beta,
                "shape": shape,
                "rank": int(args.rank),
                "ranks": tuple(int(x) for x in args.ranks),
                "iters": int(args.iters),
                "inner": int(args.inner),
                "threads": int(args.threads) if args.threads is not None else None,
                "torch_threads": int(args.torch_threads) if args.torch_threads is not None else None,
                "seed": int(args.seed),
                "initial_mean_loss": float(loss0),
            },
            "results": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()

