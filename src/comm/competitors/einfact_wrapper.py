# comm/competitors/einfact_wrapper.py
from __future__ import annotations

import pathlib
import importlib.util
import numpy as np


def patch_einfact_py(path: str, disable_valsplit: bool = True):
    """
    Patch einfact.py:
      (i) fix NameError: uses 'device' instead of 'self.device'
      (ii) optionally disable internal 5% validation split (train on all entries)
    """
    p = pathlib.Path(path)
    txt = p.read_text()

    # (i) NameError fix
    txt = txt.replace("str(device)", "str(self.device)")

    # (ii) disable internal 5% val split
    if disable_valsplit:
        txt = txt.replace(
            "val_selector = torch.rand(initial_mask_bool.shape, device=self.device) < 0.05",
            "val_selector = torch.zeros(initial_mask_bool.shape, device=self.device, dtype=torch.bool)"
        )

    p.write_text(txt)


def import_einfact(path: str):
    p = pathlib.Path(path).resolve()
    spec = importlib.util.spec_from_file_location("einfact", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def nneinfact_set_params(model, params_np, clamp=1e-10, swap_fn=None):
    """
    Inject exact initial params into NNEinFact and rebuild cached paths / Y_hat.
    params_np order must match model.param_strs.
    swap_fn: pass mod.swap if available; otherwise a callable (model_str, i)->einsum_str.
    """
    import torch
    from opt_einsum import contract, contract_path

    model.P_params = [
        torch.as_tensor(np.maximum(P, clamp), device=model.device, dtype=torch.float32)
        for P in params_np
    ]

    # recompute cached Y_hat
    model.y_path = contract_path(model.model_str, *model.P_params)[0]
    model.Y_hat = contract(model.model_str, *model.P_params, optimize=model.y_path).clamp(min=clamp)

    # rebuild the "swap" einsum strings + contraction paths
    # Prefer module-level swap (as in upstream einfact.py); fall back to model.swap if present.
    if swap_fn is None:
        swap_fn = getattr(model, "swap", None)
    if swap_fn is None:
        raise AttributeError("Could not find swap() (expected module-level mod.swap).")

    model.einsum_str = [swap_fn(model.model_str, i) for i in range(len(model.P_params))]
    model.contract_paths = [None] * len(model.P_params)

    for i in range(len(model.P_params)):
        others = [model.P_params[j] for j in range(len(model.P_params)) if j != i] + [model.Y_hat]
        model.contract_paths[i] = contract_path(model.einsum_str[i], *others, memory_limit=1e10)[0]


def run_nneinfact_cp(
    einfact_path: str,
    X: np.ndarray,
    A0,
    alpha: float,
    beta_ab: float,
    max_iter: int = 40,
    device: str = "cpu",
    disable_valsplit: bool = True,
    seed: int = 0,
):
    """
    CP competitor run with same init A0.
    Returns (loss_mean_history, time_history).
    """
    patch_einfact_py(einfact_path, disable_valsplit=disable_valsplit)
    mod = import_einfact(einfact_path)

    import torch
    torch.manual_seed(seed)

    NNEinFact = mod.NNEinFact
    swap_fn = getattr(mod, "swap", None)  # prefer module-level swap

    N = len(A0)
    letters = [chr(ord("a") + i) for i in range(N)]
    model_str = ",".join([f"{letters[i]}r" for i in range(N)]) + "->" + "".join(letters)

    shape_dict = {letters[i]: A0[i].shape[0] for i in range(N)}
    shape_dict["r"] = A0[0].shape[1]

    model = NNEinFact(model_str, shape_dict=shape_dict, alpha=alpha, beta=beta_ab, device=device)
    nneinfact_set_params(model, A0, clamp=1e-10, swap_fn=swap_fn)

    hist = model.fit(X, max_iter=max_iter, verbose=False, mask=np.ones_like(X, dtype=bool))
    return np.array(hist["loss"], dtype=float), np.array(hist["time"], dtype=float)


def run_nneinfact_tucker(
    einfact_path: str,
    X: np.ndarray,
    A0,
    G0,
    alpha: float,
    beta_ab: float,
    max_iter: int = 40,
    device: str = "cpu",
    disable_valsplit: bool = True,
    seed: int = 0,
):
    """
    Tucker competitor run with same init (A0, G0).
    Returns (loss_mean_history, time_history).
    """
    patch_einfact_py(einfact_path, disable_valsplit=disable_valsplit)
    mod = import_einfact(einfact_path)

    import torch
    torch.manual_seed(seed)

    NNEinFact = mod.NNEinFact
    swap_fn = getattr(mod, "swap", None)

    N = len(A0)
    i_letters = [chr(ord("i") + i) for i in range(N)]  # data indices
    a_letters = [chr(ord("a") + i) for i in range(N)]  # latent indices

    # ia,jb,kc,ld,abcd->ijkl
    model_str = ",".join([f"{i_letters[i]}{a_letters[i]}" for i in range(N)]) \
                + "," + "".join(a_letters) + "->" + "".join(i_letters)

    shape_dict = {i_letters[i]: A0[i].shape[0] for i in range(N)}
    for i in range(N):
        shape_dict[a_letters[i]] = A0[i].shape[1]

    model = NNEinFact(model_str, shape_dict=shape_dict, alpha=alpha, beta=beta_ab, device=device)

    params0 = [*A0, G0]  # IMPORTANT: factors first, then core
    nneinfact_set_params(model, params0, clamp=1e-10, swap_fn=swap_fn)

    hist = model.fit(X, max_iter=max_iter, verbose=False, mask=np.ones_like(X, dtype=bool))
    return np.array(hist["loss"], dtype=float), np.array(hist["time"], dtype=float)
