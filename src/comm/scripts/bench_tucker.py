import argparse
import os
from pathlib import Path
import numpy as np

from comm.core import to_float32
from comm.models.tucker import tucker_bcomm, tucker_jcomm, tucker_mu_unfolding, tucker_reconstruct


def main():
    """
    Synthetic Tucker benchmark runner.

    Runs B-CoMM, J-CoMM, and an unfolding MU baseline on a synthetic Tucker tensor,
    optionally running the NNEinFact competitor as well.

    Execute as:
        `python -m comm.scripts.bench_tucker ...`
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--shape", type=int, nargs="+", default=[60, 52, 44, 36])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 7, 6, 5])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=1)

    # competitor (optional)
    ap.add_argument("--run_einfact", action="store_true")
    ap.add_argument("--einfact_path", type=str, default="einfact.py")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--einfact_iters", type=int, default=None, help="Iterations for NNEinFact (default: --iters).")
    ap.add_argument("--torch_threads", type=int, default=None, help="torch.set_num_threads(n) for competitor.")

    # plotting/saving
    ap.add_argument("--save_dir", type=str, default="figures")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fmt", type=str, nargs="+", default=["png", "pdf"])
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--logy", action="store_true", help="Plot losses with log-scaled y-axis.")
    ap.add_argument("--start_iter", type=int, default=1, help="Skip first points in plots (default: 1).")

    args = ap.parse_args()

    if args.no_show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    from comm.utils.plotting import plot_histories

    rng = np.random.default_rng(args.seed)

    shape = tuple(args.shape)
    ranks = tuple(args.ranks)
    N = len(shape)
    assert len(ranks) == N, "Length of --ranks must match length of --shape."

    beta = float(args.beta)
    eps = float(args.eps)

    # --- synthetic Tucker data
    G_true = rng.random(ranks, dtype=np.float32) + 1e-3
    A_true = [rng.random((shape[n], ranks[n]), dtype=np.float32) + 1e-3 for n in range(N)]
    X = tucker_reconstruct(G_true, A_true) + np.float32(1e-3)

    G0 = rng.random(ranks, dtype=np.float32) + 1e-3
    A0 = [rng.random((shape[n], ranks[n]), dtype=np.float32) + 1e-3 for n in range(N)]

    X = to_float32(X)
    A0 = to_float32(A0)
    G0 = to_float32(G0)

    # --- run methods
    _, _, loss_b, t_b = tucker_bcomm(X, G0, A0, beta=beta, n_outer=args.iters, eps=eps)
    _, _, loss_j, t_j = tucker_jcomm(X, G0, A0, beta=beta, n_outer=args.iters, n_inner=args.inner, eps=eps)
    _, _, loss_mu, t_mu = tucker_mu_unfolding(X, G0, A0, beta=beta, n_outer=args.iters, eps=eps)

    # --- compare on MEAN loss
    scale = X.size
    histories = {
        "B-CoMM": (loss_b / scale, t_b),
        "J-CoMM": (loss_j / scale, t_j),
        "MU-unfold": (loss_mu / scale, t_mu),
    }

    if args.run_einfact:
        from comm.competitors.einfact_wrapper import run_nneinfact_tucker
        if args.torch_threads is not None:
            import torch
            torch.set_num_threads(int(args.torch_threads))
        alpha_einfact = 1.0
        beta_einfact = beta - 1.0
        einfact_iters = args.iters if args.einfact_iters is None else int(args.einfact_iters)
        loss_e, t_e = run_nneinfact_tucker(
            args.einfact_path, X, A0, G0,
            alpha=alpha_einfact, beta_ab=beta_einfact,
            max_iter=einfact_iters, device=args.device,
            disable_valsplit=True, seed=args.seed
        )
        histories["NNEinFact"] = (loss_e, t_e)

    def _summ(name, loss, tt):
        """
        Build a one-line summary string for a (loss, time) history.

        Parameters
        ----------
        name:
            Method label.
        loss:
            Loss history (array-like).
        tt:
            Time history (array-like).

        Returns
        -------
        str
            Human-readable summary including last finite loss/time.
        """
        loss = np.asarray(loss, dtype=float)
        tt = np.asarray(tt, dtype=float)
        idx = np.where(np.isfinite(loss) & np.isfinite(tt))[0]
        if idx.size == 0:
            return f"{name:10s}  last_loss=nan  last_time=nan"
        j = int(idx[-1])
        return f"{name:10s}  last_loss={loss[j]:.3e}  last_time={tt[j]:.3f}s  (points={loss.size})"

    print(_summ("B-CoMM", loss_b / scale, t_b))
    print(_summ("J-CoMM", loss_j / scale, t_j))
    print(_summ("MU-unfold", loss_mu / scale, t_mu))
    if "NNEinFact" in histories:
        le, te = histories["NNEinFact"]
        print(_summ("NNEinFact", le, te))

    title = f"Tucker order-{N}, beta={beta}"
    fname = (
        f"tucker_order{N}_beta{beta}_shape{'x'.join(map(str, shape))}"
        f"_ranks{'x'.join(map(str, ranks))}_seed{args.seed}"
    )

    save_dir = None if args.no_save else Path(args.save_dir)

    fig, paths = plot_histories(
        histories,
        title=title,
        ylabel="mean loss",
        save_dir=save_dir,
        filename=fname,
        dpi=args.dpi,
        fmt=tuple(args.fmt),
        yscale=("log" if args.logy else "linear"),
        start_iter=args.start_iter,
        show=(not args.no_show),
        close=False,
    )

    if save_dir is not None:
        print("Saved figures:")
        for p in paths:
            print("  ", p)

    if args.no_show:
        plt.close(fig)


if __name__ == "__main__":
    main()
