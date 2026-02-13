"""
Unfolding-based baselines used for validation/comparison.

These routines intentionally form matricizations and Khatri–Rao / Kronecker
products. They are useful to:
- sanity-check contraction-only implementations
- provide a simple baseline (may be slower / numerically less stable)
"""

import numpy as np


def unfold_mode(T, n: int):
    """
    Compute the mode-`n` unfolding (matricization) of a tensor.

    The unfolding uses axis order `[n] + [0..N-1 except n]` (in increasing order).

    Parameters
    ----------
    T:
        Input tensor/ndarray of shape `(I1, ..., IN)`.
    n:
        Mode index to unfold along (0-based).

    Returns
    -------
    ndarray
        Matrix of shape `(T.shape[n], prod_{m!=n} T.shape[m])`.
    """
    N = T.ndim
    axes = [n] + [m for m in range(N) if m != n]
    return np.transpose(T, axes).reshape(T.shape[n], -1)


def khatri_rao_list(mats):
    """
    Compute the column-wise Khatri–Rao product for a list of matrices.

    If `mats = [A1, A2, ..., Ak]` with shapes `(I1,R), (I2,R), ..., (Ik,R)`,
    returns a matrix of shape `((I1*...*Ik), R)` with:
    `K[:, r] = kron(Ak[:,r], ..., A1[:,r])` (consistent with the iterative build).

    Parameters
    ----------
    mats:
        List of 2D arrays with the same number of columns `R`.

    Returns
    -------
    ndarray
        Khatri–Rao product matrix of shape `(prod I_k, R)`.
    """
    if len(mats) == 0:
        raise ValueError("khatri_rao_list expects at least one matrix.")
    K = mats[0]
    for A in mats[1:]:
        # (I,R) and (J,R) -> (IJ,R)
        K = (A[:, None, :] * K[None, :, :]).reshape(A.shape[0] * K.shape[0], K.shape[1])
    return K


def cp_mttkrp_unfold(T, factors, n: int):
    """
    Compute CP MTTKRP via unfolding (baseline).

    Computes:
      `MTTKRP = T_(n) @ KRP(factors_except_n)`

    With the unfolding convention in `unfold_mode`, the Khatri–Rao product must
    be built using `reversed(other_modes)`.

    Parameters
    ----------
    T:
        Input tensor/ndarray of shape `(I1,...,IN)`.
    factors:
        List of factor matrices `A^(m)` with shape `(I_m, R)`.
    n:
        Mode index (0-based).

    Returns
    -------
    ndarray
        Matrix of shape `(I_n, R)`.
    """
    N = len(factors)
    others = [m for m in range(N) if m != n]
    Tn = unfold_mode(T, n)
    K = khatri_rao_list([factors[m] for m in reversed(others)])
    return Tn @ K


def kron_list(mats):
    """
    Compute the Kronecker product of a list of matrices (in the given order).

    Parameters
    ----------
    mats:
        List of 2D arrays.

    Returns
    -------
    ndarray
        Kronecker product matrix.
    """
    if len(mats) == 0:
        return np.array([[1.0]])
    K = mats[0]
    for A in mats[1:]:
        K = np.kron(K, A)
    return K


def tucker_Bn_unfold(core, factors, n: int):
    """
    Build Tucker baseline matrix B^(n) for factor updates using unfoldings.

    Produces `B^(n)` such that:
      `Xhat_(n) = A^(n) @ B^(n)`

    with:
      `B^(n) = G_(n) @ kron(A^(m) for m!=n)^T`,
    where "others" are in increasing mode order.

    Parameters
    ----------
    core:
        Tucker core tensor `G` of shape `(J1,...,JN)`.
    factors:
        List of factor matrices `A^(m)` with shape `(I_m, J_m)`.
    n:
        Mode index (0-based).

    Returns
    -------
    ndarray
        Baseline matrix `B^(n)` of shape `(J_n, prod_{m!=n} I_m)`.
    """
    N = len(factors)
    others = [m for m in range(N) if m != n]
    Gn = unfold_mode(core, n)                          # (J_n, prod J_rest)
    K = kron_list([factors[m] for m in others])        # (prod I_rest, prod J_rest)
    return Gn @ K.T                                    # (J_n, prod I_rest)
