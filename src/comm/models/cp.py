"""
Nonnegative CP decomposition algorithms under β-divergences.

Implements:
- B-CoMM: block contraction-only MM updates (multiplicative)
- J-CoMM: joint-MM outer/inner scheme using reference-powered tensors
- MU-unfolding: unfolding-based baseline for comparison
"""

import time
import numpy as np
from comm.core import _symbols, einsum, gamma_beta, beta_divergence, safe_PQ
from comm.core import mu_update_inplace, mu_update_from_ref, chi1, chi2
from comm.baselines import cp_mttkrp_unfold

RIDX = "R"


def cp_reconstruct(factors):
    """
    Reconstruct a CP tensor from factor matrices using einsum.

    For factors `A^(n)` of shape `(I_n, R)`, returns:
    \(\widehat X_{i_1...i_N} = \sum_{r=1}^R \prod_n A^{(n)}_{i_n r}\).

    Parameters
    ----------
    factors:
        List of factor matrices, each of shape `(I_n, R)`.

    Returns
    -------
    ndarray
        Reconstruction tensor of shape `(I_1, ..., I_N)`.
    """
    N = len(factors)
    idx, _ = _symbols(N, reserved=(RIDX,))
    expr_in = ",".join([f"{idx[n]}{RIDX}" for n in range(N)])
    expr_out = "".join(idx)
    return einsum(expr_in + "->" + expr_out, *factors)


def cp_contr(T, factors, n):
    """
    Compute the CP contraction (MTTKRP-like) without unfoldings.

    Returns the matrix of shape `(I_n, R)` with entries:
    \[
      M_{i_n r} = \sum_{i_{-n}} T_{i_1...i_N} \prod_{m\neq n} A^{(m)}_{i_m r}.
    \]

    Parameters
    ----------
    T:
        Tensor to contract, shape `(I_1, ..., I_N)`.
    factors:
        List of factor matrices, each of shape `(I_m, R)`.
    n:
        Mode index (0-based) of the output factor dimension.

    Returns
    -------
    ndarray
        Contracted matrix of shape `(I_n, R)`.
    """
    N = len(factors)
    idx, _ = _symbols(N, reserved=(RIDX,))
    subs = ["".join(idx)]
    ops = [T]
    for m in range(N):
        if m == n:
            continue
        subs.append(f"{idx[m]}{RIDX}")
        ops.append(factors[m])
    expr = ",".join(subs) + "->" + f"{idx[n]}{RIDX}"
    return einsum(expr, *ops)


# -----------------------
# B-CoMM and J-CoMM
# -----------------------
def cp_bcomm(X, factors0, beta: float, n_outer=50, eps=1e-12):
    """
    Run B-CoMM (block contraction-only MM) for nonnegative CP under β-divergence.

    Each outer iteration performs N block updates. For each mode n:
    - build current reconstruction Xhat
    - compute safe P/Q tensors
    - compute numerator/denominator via contraction-only `cp_contr`
    - apply multiplicative update

    Parameters
    ----------
    X:
        Observed tensor, shape `(I_1,...,I_N)`.
    factors0:
        Initial factor list, each `(I_n, R)` (nonnegative).
    beta:
        β-divergence parameter.
    n_outer:
        Number of outer iterations.
    eps:
        Positivity safeguard (and denominator floor).

    Returns
    -------
    factors:
        Updated factor list.
    losses:
        1D array of length `n_outer` with **sum** β-divergence values.
    times:
        1D array of length `n_outer` with elapsed wall time (seconds).
    """
    factors = [A.copy() for A in factors0]
    N = len(factors)
    g = gamma_beta(beta)

    losses, times = [], []
    t0 = time.perf_counter()

    for _ in range(n_outer):
        for n in range(N):
            Xhat = cp_reconstruct(factors)
            P, Q, _ = safe_PQ(X, Xhat, beta, eps)
            Num = cp_contr(P, factors, n)
            Den = cp_contr(Q, factors, n)
            mu_update_inplace(factors[n], Num, Den, g, eps)

        Xhat = np.maximum(cp_reconstruct(factors), eps)
        losses.append(beta_divergence(X, Xhat, beta))
        times.append(time.perf_counter() - t0)

    return factors, np.array(losses), np.array(times)


def cp_jcomm(X, factors0, beta: float, n_outer=50, n_inner=1, eps=1e-12):
    """
    Run J-CoMM (joint majorization-minimization) for nonnegative CP under β-divergence.

    Outer loop:
    - set reference factors \\(\\widetilde A\\) and reference reconstruction \\(\\widetilde X\\)
    - compute reference-powered tensors \\(\\widetilde P, \\widetilde Q\\)

    Inner loop:
    - update each factor from its reference \\(\\widetilde A^{(n)}\\) using transformed factors
      χ₁/χ₂ and contraction-only numerators/denominators.

    Parameters
    ----------
    X:
        Observed tensor.
    factors0:
        Initial factor list.
    beta:
        β-divergence parameter.
    n_outer:
        Number of outer iterations.
    n_inner:
        Number of inner iterations (surrogate-decrease steps) per outer iteration.
    eps:
        Positivity safeguard (and denominator floor).

    Returns
    -------
    factors:
        Updated factor list.
    losses:
        1D array of length `n_outer` with **sum** β-divergence values (evaluated after outer iterations).
    times:
        1D array of length `n_outer` with elapsed wall time (seconds).
    """
    factors = [A.copy() for A in factors0]
    N = len(factors)
    g = gamma_beta(beta)

    losses, times = [], []
    t0 = time.perf_counter()

    for _ in range(n_outer):
        Atilde = [A.copy() for A in factors]
        Xhat_tilde = cp_reconstruct(Atilde)
        Ptilde, Qtilde, _ = safe_PQ(X, Xhat_tilde, beta, eps)

        A = [A.copy() for A in Atilde]
        A1 = [chi1(A[n], Atilde[n], beta) for n in range(N)]
        A2 = [chi2(A[n], Atilde[n], beta) for n in range(N)]

        for _inner in range(n_inner):
            for n in range(N):
                NumJ = cp_contr(Ptilde, A1, n)
                DenJ = cp_contr(Qtilde, A2, n)
                A[n] = mu_update_from_ref(Atilde[n], NumJ, DenJ, g, eps)
                A1[n] = chi1(A[n], Atilde[n], beta)
                A2[n] = chi2(A[n], Atilde[n], beta)

        factors = A
        Xhat = np.maximum(cp_reconstruct(factors), eps)
        losses.append(beta_divergence(X, Xhat, beta))
        times.append(time.perf_counter() - t0)

    return factors, np.array(losses), np.array(times)


# -----------------------
# MU unfolding baseline
# -----------------------
def cp_mu_unfolding(X, factors0, beta: float, n_outer=50, eps=1e-12):
    """
    Run an unfolding-based MU baseline for CP under β-divergence.

    Uses `cp_mttkrp_unfold` to compute numerator/denominator via explicit unfolding
    and Khatri–Rao products. This is primarily used as a baseline and for sanity checks.

    Parameters
    ----------
    X:
        Observed tensor.
    factors0:
        Initial factor list.
    beta:
        β-divergence parameter.
    n_outer:
        Number of outer iterations.
    eps:
        Positivity safeguard (and denominator floor).

    Returns
    -------
    factors:
        Updated factor list.
    losses:
        1D array of length `n_outer` with **sum** β-divergence values.
    times:
        1D array of length `n_outer` with elapsed wall time (seconds).
    """
    factors = [A.copy() for A in factors0]
    N = len(factors)
    g = gamma_beta(beta)

    losses, times = [], []
    t0 = time.perf_counter()

    for _ in range(n_outer):
        for n in range(N):
            Xhat = cp_reconstruct(factors)
            P, Q, _ = safe_PQ(X, Xhat, beta, eps)
            Num = cp_mttkrp_unfold(P, factors, n)
            Den = cp_mttkrp_unfold(Q, factors, n)
            # Keep the unfolding baseline numerically comparable to contraction-only updates:
            # apply the same ratio clipping + eps floor.
            mu_update_inplace(factors[n], Num, Den, g, eps)

        Xhat = np.maximum(cp_reconstruct(factors), eps)
        losses.append(beta_divergence(X, Xhat, beta))
        times.append(time.perf_counter() - t0)

    return factors, np.array(losses), np.array(times)
