import argparse
import os
from pathlib import Path
import numpy as np

from comm.core import to_float32
from comm.models.tucker import tucker_bcomm, tucker_jcomm

from comm.competitors.einfact_wrapper import patch_einfact_py, import_einfact


def load_npz_tensor(path: str, key: str = "Y") -> np.ndarray:
    """
    Load a tensor stored in an `.npz` file.

    Parameters
    ----------
    path:
        Path to the `.npz` file.
    key:
        Array key inside the archive (default: `"Y"`).

    Returns
    -------
    ndarray
        Loaded tensor.

    Raises
    ------
    KeyError
        If `key` is not present in the archive.
    """
    z = np.load(path)
    if key not in z:
        raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(z.keys())}")
    return z[key]


def run_nneinfact_uber(
    einfact_path: str,
    Y: np.ndarray,
    beta: float,
    k: int,
    r: int,
    device: str,
    max_iter: int,
    seed: int,
    disable_valsplit: bool = True,
    return_params: bool = False,
):
    """
    Run NNEinFact on Uber tensor with the model from their demo:
      model_str = 'wr,hr,dr,ikr,jkr->whdij'
    and map our beta-divergence parameter beta (in [0,2)) to their (alpha,beta_ab):
      alpha = 1.0, beta_ab = beta - 1.0
    so that their AB-loss matches beta-divergence (up to constants) and is reported as MEAN loss.

    Parameters
    ----------
    einfact_path:
        Path to the local `einfact.py` file.
    Y:
        Data tensor (NumPy array).
    beta:
        β-divergence parameter in our notation.
    k, r:
        NNEinFact latent dimensions used in their Uber demo model string.
    device:
        Torch device string ("cpu" or "cuda").
    max_iter:
        Maximum number of iterations passed to NNEinFact.
    seed:
        RNG seed used for reproducibility.
    disable_valsplit:
        If True, disables NNEinFact's internal validation split so training uses all entries.

    Returns
    -------
    loss_e:
        1D array of mean losses as reported by NNEinFact.
    t_e:
        1D array of elapsed times (seconds) as reported by NNEinFact.
    params:
        (Optional) list of learned parameters as NumPy arrays, in the same order
        as `model.param_strs`. Only returned when `return_params=True`.
    """
    # Their init uses global np.random, so set it for reproducibility
    np.random.seed(seed)

    # patch + import
    patch_einfact_py(einfact_path, disable_valsplit=disable_valsplit)
    mod = import_einfact(einfact_path)

    # torch seed too
    import torch
    torch.manual_seed(seed)

    NNEinFact = mod.NNEinFact

    model_str = "wr,hr,dr,ikr,jkr->whdij"
    rhs = model_str.split("->")[1]  # 'whdij'
    assert len(rhs) == Y.ndim, f"RHS '{rhs}' does not match Y.ndim={Y.ndim}"

    shape_dict = {**dict(zip(rhs, Y.shape)), "k": int(k), "r": int(r)}

    alpha_einfact = 1.0
    beta_einfact = float(beta) - 1.0

    model = NNEinFact(
        model_str,
        shape_dict=shape_dict,
        device=device,
        alpha=alpha_einfact,
        beta=beta_einfact,
    )

    hist = model.fit(Y, max_iter=max_iter, verbose=False, mask=np.ones_like(Y, dtype=bool))
    loss_e = np.array(hist["loss"], dtype=float)   # mean loss
    t_e = np.array(hist["time"], dtype=float)
    if return_params:
        params = model.get_params()
        return loss_e, t_e, params
    return loss_e, t_e


def run_nneinfact_uber_tucker(
    einfact_path: str,
    Y: np.ndarray,
    beta: float,
    ranks: tuple[int, int, int, int, int],
    device: str,
    max_iter: int,
    seed: int,
    disable_valsplit: bool = True,
    return_params: bool = False,
):
    """
    Run NNEinFact on the Uber tensor using a **Tucker** model string.

    The Tucker model used here is:
        'wa,hb,dc,if,je,abcfe->whdij'

    where (w,h,d,i,j) are the data modes and (a,b,c,f,e) are Tucker latent modes.
    The core has shape (a,b,c,f,e).

    Parameters
    ----------
    einfact_path:
        Path to local `einfact.py`.
    Y:
        Uber tensor, shape (w,h,d,i,j).
    beta:
        β-divergence parameter in our notation.
    ranks:
        Tucker ranks (a,b,c,f,e) for modes (w,h,d,i,j) respectively.
    device:
        Torch device string ("cpu" or "cuda").
    max_iter:
        Maximum number of iterations.
    seed:
        RNG seed.
    disable_valsplit:
        If True, disables NNEinFact internal validation split.
    return_params:
        If True, also return the learned parameter list.

    Returns
    -------
    loss_e, t_e:
        Mean loss and time histories from NNEinFact.
    params:
        Optional parameter list (wa, hb, dc, if, je, abcfe) if `return_params=True`.
    """
    np.random.seed(seed)

    patch_einfact_py(einfact_path, disable_valsplit=disable_valsplit)
    mod = import_einfact(einfact_path)

    import torch
    torch.manual_seed(seed)

    model_str = "wa,hb,dc,if,je,abcfe->whdij"
    rhs = model_str.split("->")[1]  # 'whdij'
    assert len(rhs) == Y.ndim, f"RHS '{rhs}' does not match Y.ndim={Y.ndim}"

    # map data dims
    shape_dict = {**dict(zip(rhs, Y.shape))}

    # map latent dims
    a, b, c, f, e = (int(x) for x in ranks)
    shape_dict.update({"a": a, "b": b, "c": c, "f": f, "e": e})

    alpha_einfact = 1.0
    beta_einfact = float(beta) - 1.0

    NNEinFact = mod.NNEinFact
    model = NNEinFact(
        model_str,
        shape_dict=shape_dict,
        device=device,
        alpha=alpha_einfact,
        beta=beta_einfact,
    )

    hist = model.fit(Y, max_iter=max_iter, verbose=False, mask=np.ones_like(Y, dtype=bool))
    loss_e = np.array(hist["loss"], dtype=float)
    t_e = np.array(hist["time"], dtype=float)
    if return_params:
        return loss_e, t_e, model.get_params()
    return loss_e, t_e


def plot_hour_factor(
    Theta_HR: np.ndarray,
    save_path: Path,
    xlabel: str = "Latent components (r)",
    ylabel: str = "Hours of Day",
    hour_tick_step: int = 3,
    dpi: int = 300,
):
    """
    Visualize the 'hour' factor matrix from the Uber NNEinFact model.

    In the NNEinFact Uber demo model string:
        'wr,hr,dr,ikr,jkr->whdij'
    the second parameter 'hr' is a matrix of shape (H, r), where H=24.
    This is what the original demo plots as "Theta_HK" (naming aside).

    Parameters
    ----------
    Theta_HR:
        Hour factor of shape (24, r).
    save_path:
        Output path for the figure (extension determines format).
    xlabel, ylabel:
        Axis labels.
    hour_tick_step:
        Step between displayed hour tick labels.
    dpi:
        DPI for raster outputs.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    H, r = Theta_HR.shape
    fig = plt.figure(figsize=(7.5, 4.0))
    ax = plt.gca()
    im = ax.imshow(Theta_HR, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(r))
    ax.set_xticklabels([f"{i}" for i in range(1, r + 1)])

    ticks = list(range(0, H, max(1, int(hour_tick_step))))
    labels = []
    for h in ticks:
        hh = h + 1  # display 1..24
        if hh == 12:
            labels.append("12 PM")
        elif hh < 12:
            labels.append(f"{hh} AM")
        else:
            labels.append(f"{hh-12} PM")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_tucker_components(
    wa: np.ndarray,
    hb: np.ndarray,
    dc: np.ndarray,
    i_f: np.ndarray,
    j_e: np.ndarray,
    core_abcfe: np.ndarray,
    save_path: Path,
    n_components: int = 3,
    dpi: int = 300,
):
    """
    Create an interpretable Tucker-style visualization (approximate) for Uber.

    This is meant as a *rough analog* of the qualitative figure in the NNEinFact paper,
    but for a Tucker model:
        wa,hb,dc,if,je,abcfe -> whdij

    We build components by selecting the strongest (b,c) pairs under a
    week-weighted core magnitude, then plot per-component:
    - Temporal pattern: outer(hb[:,b], dc[:,c])  -> shape (24, 7)
    - Spatial loading:  i_f @ core_slice @ j_e^T -> shape (100, 100)
      where core_slice[f,e] = sum_a wa_mean[a] * core[a,b,c,f,e]

    Parameters
    ----------
    wa, hb, dc, i_f, j_e:
        Tucker factor matrices for modes (w,h,d,i,j).
    core_abcfe:
        Tucker core tensor of shape (a,b,c,f,e).
    save_path:
        Output path for the figure (extension determines format).
    n_components:
        Number of (b,c) components to display.
    dpi:
        DPI for raster outputs.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    wa = np.asarray(wa, dtype=float)
    hb = np.asarray(hb, dtype=float)
    dc = np.asarray(dc, dtype=float)
    i_f = np.asarray(i_f, dtype=float)
    j_e = np.asarray(j_e, dtype=float)
    core = np.asarray(core_abcfe, dtype=float)

    # weights over the 'a' dimension (week mode); use mean to keep simple.
    wa_mean = wa.mean(axis=0)  # (a,)
    wa_mean = wa_mean / (wa_mean.sum() + 1e-12)

    # score (b,c) by aggregating core magnitudes weighted by wa_mean, summing over spatial dims.
    # core shape: (a,b,c,f,e)
    core_bc = np.tensordot(wa_mean, core, axes=(0, 0))  # (b,c,f,e)
    score_bc = core_bc.sum(axis=(2, 3))  # (b,c)

    b_dim, c_dim = score_bc.shape
    flat = score_bc.reshape(-1)
    top = np.argsort(flat)[::-1][: max(1, int(n_components))]
    bc_pairs = [(int(t // c_dim), int(t % c_dim)) for t in top]

    fig, axes = plt.subplots(len(bc_pairs), 2, figsize=(10.5, 3.2 * len(bc_pairs)))
    if len(bc_pairs) == 1:
        axes = np.array([axes])

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for row, (b, c) in enumerate(bc_pairs):
        # temporal heatmap
        temporal = np.outer(hb[:, b], dc[:, c])  # (24,7)
        ax_t = axes[row, 0]
        im_t = ax_t.imshow(temporal, aspect="auto")
        ax_t.set_title(f"Temporal pattern (b={b+1}, c={c+1})")
        ax_t.set_xlabel("Day of week")
        ax_t.set_ylabel("Hour of day")
        ax_t.set_xticks(np.arange(7))
        ax_t.set_xticklabels(day_labels)
        ax_t.set_yticks(list(range(0, 24, 3)))
        ax_t.set_yticklabels([str(h) for h in range(0, 24, 3)])
        fig.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

        # spatial map (100x100)
        core_slice = core_bc[b, c, :, :]  # (f,e)
        spatial = np.einsum("if,fe,je->ij", i_f, core_slice, j_e, optimize=True)
        ax_s = axes[row, 1]
        im_s = ax_s.imshow(spatial, aspect="equal")
        ax_s.set_title("Spatial loading (i×j)")
        ax_s.set_xlabel("j")
        ax_s.set_ylabel("i")
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    """
    Real-data benchmark runner for the Uber tensor example.

    Runs Tucker B-CoMM/J-CoMM on a provided `.npz` tensor and (optionally) the
    NNEinFact competitor with the model string used in their demo.

    Execute as:
        `python -m comm.scripts.bench_real_uber ...`
    """
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data", type=str, default="data/Y.npz")
    ap.add_argument("--key", type=str, default="Y")

    # divergence + iterations
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=0)

    # Tucker ranks (for our methods)
    # Default chosen moderately; adjust as you like.
    ap.add_argument("--tucker_ranks", type=int, nargs="+", default=[10, 10, 5, 10, 10])

    # competitor (optional)
    ap.add_argument("--run_einfact", action="store_true")
    ap.add_argument("--einfact_path", type=str, default="einfact.py")
    ap.add_argument("--device", type=str, default="cpu")  # "cuda" if available
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--r", type=int, default=10)
    ap.add_argument(
        "--einfact_model",
        type=str,
        choices=["custom", "tucker"],
        default="custom",
        help="Which NNEinFact model to run on Uber (default: custom).",
    )
    ap.add_argument(
        "--einfact_tucker_ranks",
        type=int,
        nargs=5,
        default=None,
        help="Tucker ranks (a b c f e) for NNEinFact Tucker model; default uses --tucker_ranks.",
    )

    # plotting / saving
    ap.add_argument("--save_dir", type=str, default="figures")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fmt", type=str, nargs="+", default=["png", "pdf"])
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--logy", action="store_true", help="Plot losses with log-scaled y-axis.")
    ap.add_argument("--start_iter", type=int, default=1, help="Skip first points in plots (default: 1).")

    # competitor threads (optional)
    ap.add_argument("--torch_threads", type=int, default=None, help="torch.set_num_threads(n) for competitor.")

    # competitor visualization
    ap.add_argument("--plot_hour_factor", action="store_true", help="Save the NNEinFact hour-factor heatmap (hr).")
    ap.add_argument("--hour_tick_step", type=int, default=3, help="Hour tick label step for the heatmap (default: 3).")
    ap.add_argument(
        "--plot_tucker_components",
        action="store_true",
        help="For NNEinFact Tucker model: save a 2-column figure (temporal + spatial) for top components.",
    )
    ap.add_argument("--n_components", type=int, default=3, help="Number of Tucker components to display (default: 3).")

    args = ap.parse_args()

    if args.no_show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    from comm.utils.plotting import plot_histories

    # ---- load real tensor
    Y = load_npz_tensor(args.data, key=args.key)
    print(f"Loaded Y with shape={Y.shape}, dtype={Y.dtype}")

    # work in float32 (important for memory here)
    Y = to_float32(Y)

    N = Y.ndim
    ranks = tuple(args.tucker_ranks)
    if len(ranks) != N:
        raise ValueError(f"--tucker_ranks must have length {N}, got {len(ranks)}")

    beta = float(args.beta)
    eps = float(args.eps)

    print(f"Our Tucker ranks (for B/J-CoMM): {ranks}")
    if args.run_einfact:
        if args.einfact_model == "custom":
            print(f"NNEinFact custom latent sizes: k={int(args.k)}, r={int(args.r)} (these are not Tucker ranks)")
        else:
            tr = tuple(args.einfact_tucker_ranks) if args.einfact_tucker_ranks is not None else ranks
            print(f"NNEinFact Tucker ranks (a,b,c,f,e): {tuple(int(x) for x in tr)}")

    # ---- init Tucker (positive)
    rng = np.random.default_rng(args.seed)
    G0 = (rng.random(ranks, dtype=np.float32) + 1e-3).astype(np.float32)
    A0 = [(rng.random((Y.shape[n], ranks[n]), dtype=np.float32) + 1e-3).astype(np.float32) for n in range(N)]

    # ---- run our Tucker CoMM methods
    # Note: no MU-unfolding baseline here: for Y.shape=(27,24,7,100,100), unfolding Tucker baseline
    # would require forming huge Kronecker matrices (impossible).
    _, _, loss_b, t_b = tucker_bcomm(Y, G0, A0, beta=beta, n_outer=args.iters, eps=eps)
    _, _, loss_j, t_j = tucker_jcomm(Y, G0, A0, beta=beta, n_outer=args.iters, n_inner=args.inner, eps=eps)

    # ---- convert our SUM loss to MEAN loss (NNEinFact logs mean)
    scale = Y.size
    histories = {
        "B-CoMM (Tucker)": (loss_b / scale, t_b),
        "J-CoMM (Tucker)": (loss_j / scale, t_j),
    }

    # ---- competitor (optional)
    if args.run_einfact:
        if args.torch_threads is not None:
            import torch
            torch.set_num_threads(int(args.torch_threads))

        if args.einfact_model == "custom":
            if args.plot_hour_factor:
                loss_e, t_e, params = run_nneinfact_uber(
                    einfact_path=args.einfact_path,
                    Y=Y,
                    beta=beta,
                    k=args.k,
                    r=args.r,
                    device=args.device,
                    max_iter=args.iters,
                    seed=args.seed,
                    disable_valsplit=True,
                    return_params=True,
                )
            else:
                loss_e, t_e = run_nneinfact_uber(
                    einfact_path=args.einfact_path,
                    Y=Y,
                    beta=beta,
                    k=args.k,
                    r=args.r,
                    device=args.device,
                    max_iter=args.iters,
                    seed=args.seed,
                    disable_valsplit=True,
                    return_params=False,
                )

            # Optional visualization: plot the 'hr' factor (2nd param in this model string).
            if args.plot_hour_factor and (not args.no_save):
                Theta_HR = np.asarray(params[1], dtype=float)
                out = Path(args.save_dir) / (
                    f"uber_nneinfact_hour_factor_custom_beta{beta}_k{int(args.k)}_r{int(args.r)}_seed{int(args.seed)}.png"
                )
                plot_hour_factor(
                    Theta_HR,
                    save_path=out,
                    hour_tick_step=int(args.hour_tick_step),
                    dpi=int(args.dpi),
                )
                print("Saved hour-factor heatmap:", out)
            histories["NNEinFact custom (wr,hr,dr,ikr,jkr)"] = (loss_e, t_e)

        else:
            tr = tuple(args.einfact_tucker_ranks) if args.einfact_tucker_ranks is not None else ranks
            tr = tuple(int(x) for x in tr)
            if args.plot_hour_factor:
                loss_e, t_e, params = run_nneinfact_uber_tucker(
                    einfact_path=args.einfact_path,
                    Y=Y,
                    beta=beta,
                    ranks=tr,  # a,b,c,f,e
                    device=args.device,
                    max_iter=args.iters,
                    seed=args.seed,
                    disable_valsplit=True,
                    return_params=True,
                )
            else:
                loss_e, t_e = run_nneinfact_uber_tucker(
                    einfact_path=args.einfact_path,
                    Y=Y,
                    beta=beta,
                    ranks=tr,
                    device=args.device,
                    max_iter=args.iters,
                    seed=args.seed,
                    disable_valsplit=True,
                    return_params=False,
                )

            # For Tucker model 'wa,hb,dc,if,je,abcfe->whdij', the hour factor is 'hb' (2nd param).
            if args.plot_hour_factor and (not args.no_save):
                Theta_HB = np.asarray(params[1], dtype=float)  # hb
                out = Path(args.save_dir) / (
                    f"uber_nneinfact_hour_factor_tucker_beta{beta}_ranks{'x'.join(map(str, tr))}_seed{int(args.seed)}.png"
                )
                plot_hour_factor(
                    Theta_HB,
                    save_path=out,
                    xlabel="Latent components (b)",
                    hour_tick_step=int(args.hour_tick_step),
                    dpi=int(args.dpi),
                )
                print("Saved hour-factor heatmap:", out)

            if args.plot_tucker_components and (not args.no_save):
                wa, hb, dc, i_f, j_e, core = params  # model_str param order
                out = Path(args.save_dir) / (
                    f"uber_nneinfact_tucker_components_beta{beta}_ranks{'x'.join(map(str, tr))}_seed{int(args.seed)}.png"
                )
                plot_tucker_components(
                    wa=wa,
                    hb=hb,
                    dc=dc,
                    i_f=i_f,
                    j_e=j_e,
                    core_abcfe=core,
                    save_path=out,
                    n_components=int(args.n_components),
                    dpi=int(args.dpi),
                )
                print("Saved Tucker component figure:", out)
            histories["NNEinFact Tucker (wa,hb,dc,if,je,abcfe)"] = (loss_e, t_e)

    def _summ(name, loss, tt):
        loss = np.asarray(loss, dtype=float)
        tt = np.asarray(tt, dtype=float)
        idx = np.where(np.isfinite(loss) & np.isfinite(tt))[0]
        if idx.size == 0:
            return f"{name:18s}  last_loss=nan  last_time=nan"
        j = int(idx[-1])
        return f"{name:18s}  last_loss={loss[j]:.3e}  last_time={tt[j]:.3f}s  (points={loss.size})"

    # print short summary (helps diagnosing threading effects)
    for k, (loss, tt) in histories.items():
        print(_summ(k, loss, tt))

    title = f"Uber Y — order-{N}, beta={beta}"
    fname = (
        f"uber_tucker_order{N}_beta{beta}"
        f"_shape{'x'.join(map(str, Y.shape))}"
        f"_ranks{'x'.join(map(str, ranks))}"
        f"_seed{args.seed}"
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
        start_iter=int(args.start_iter),
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
