# import matplotlib.pyplot as plt


# def plot_histories(histories, title, ylabel="mean loss"):
#     """
#     histories: dict name -> (loss_array, time_array)
#     """
#     plt.style.use("seaborn-v0_8-whitegrid")

#     fig = plt.figure(figsize=(12, 4))

#     ax1 = plt.subplot(1, 2, 1)
#     for name, (loss, tt) in histories.items():
#         ax1.plot(loss, label=name, linewidth=2)
#     ax1.set_title(title + " — loss vs iter")
#     ax1.set_xlabel("outer iteration")
#     ax1.set_ylabel(ylabel)
#     ax1.legend()

#     ax2 = plt.subplot(1, 2, 2)
#     for name, (loss, tt) in histories.items():
#         ax2.plot(tt, loss, label=name, linewidth=2)
#     ax2.set_title(title + " — loss vs time")
#     ax2.set_xlabel("time (s)")
#     ax2.set_ylabel(ylabel)
#     ax2.legend()

#     plt.tight_layout()
#     return fig
"""
Plotting utilities for benchmark scripts.

These helpers are intentionally simple and designed to work in headless
environments (e.g., CI, macOS non-GUI terminals) by respecting the `MPLBACKEND`
environment variable set by the calling script.
"""

import os
from pathlib import Path
import matplotlib

# Let env var override; otherwise, `bench_* --no_show` will set MPLBACKEND=Agg.
if "MPLBACKEND" in os.environ:
    matplotlib.use(os.environ["MPLBACKEND"], force=True)

import matplotlib.pyplot as plt
import numpy as np


def plot_histories(
    histories,
    title,
    ylabel="mean loss",
    save_dir=None,
    filename=None,
    dpi=300,
    fmt=("png", "pdf"),
    yscale="linear",
    start_iter=0,
    show=False,
    close=False,
):
    """
    Plot loss vs iteration and loss vs wall time for multiple methods.

    The function:
    - optionally skips the first `start_iter` points (useful to hide iteration-0 init)
    - ignores non-finite values (NaN/Inf) when plotting
    - can save figures in multiple formats

    Parameters
    ----------
    histories : dict
        name -> (loss_array, time_array)
    title : str
    ylabel : str
    save_dir : str or Path or None
        If provided, save figures into this directory (created if missing).
    filename : str or None
        Base filename without extension. If None, derived from title.
    dpi : int
        DPI for raster formats (png).
    fmt : str or tuple[str]
        Output format(s), e.g. "png" or ("png","pdf","svg").
    yscale : {"linear","log"}
        Y-axis scale for both subplots.
    start_iter : int
        Number of initial points to skip in both loss and time curves.
    show : bool
        If True, calls plt.show().
    close : bool
        If True, closes the figure after saving/showing.

    Returns
    -------
    fig : matplotlib.figure.Figure
    saved_paths : list[Path]
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 2, 1)
    for name, (loss, tt) in histories.items():
        loss = np.asarray(loss, dtype=float)
        s = int(start_iter) if start_iter is not None else 0
        if s > 0:
            s = min(s, loss.size)  # allow skipping beyond length safely
            loss = loss[s:]
        mask = np.isfinite(loss)
        x = np.arange((s if s > 0 else 0), (s if s > 0 else 0) + loss.size, dtype=int)
        ax1.plot(x[mask], loss[mask], label=name, linewidth=2)
    ax1.set_title(title + " — loss vs iter")
    ax1.set_xlabel("outer iteration")
    ax1.set_ylabel(ylabel)
    ax1.set_yscale(yscale)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for name, (loss, tt) in histories.items():
        loss = np.asarray(loss, dtype=float)
        tt = np.asarray(tt, dtype=float)
        s = int(start_iter) if start_iter is not None else 0
        if s > 0:
            s = min(s, loss.size, tt.size)
            loss = loss[s:]
            tt = tt[s:]
        mask = np.isfinite(loss) & np.isfinite(tt)
        ax2.plot(tt[mask], loss[mask], label=name, linewidth=2)
    ax2.set_title(title + " — loss vs time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel(ylabel)
    ax2.set_yscale(yscale)
    ax2.legend()

    fig.tight_layout()

    saved_paths = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # make a safe default filename
        if filename is None:
            safe = title.strip().lower()
            safe = safe.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-", "."))
            filename = safe[:180] if safe else "figure"

        if isinstance(fmt, str):
            fmt = (fmt,)

        for ext in fmt:
            ext = ext.lstrip(".").lower()
            out = save_dir / f"{filename}.{ext}"

            # DPI applies mainly to raster outputs; harmless for vector formats.
            fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
            saved_paths.append(out)

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, saved_paths
