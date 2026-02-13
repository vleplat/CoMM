"""
Nonnegative Tucker decomposition algorithms under β-divergences.

Implements:
- B-CoMM: block contraction-only MM updates (multiplicative) for factors and core
- J-CoMM: joint-MM outer/inner scheme using reference-powered tensors
- MU-unfolding: unfolding-based baseline for comparison
"""

import time
import numpy as np
from comm.core import _symbols, einsum, gamma_beta, beta_divergence, safe_PQ
from comm.core import mu_update_inplace, mu_update_from_ref, chi1, chi2
from comm.baselines import unfold_mode, kron_list, tucker_Bn_unfold


def tucker_reconstruct(core, factors):
    """
    Reconstruct a Tucker tensor from core and factor matrices using einsum.

    For core `G` of shape `(J1,...,JN)` and factors `A^(n)` of shape `(I_n, J_n)`,
    returns:
    \(\widehat X = G \\times_1 A^{(1)} \\times_2 \\cdots \\times_N A^{(N)}\).

    Parameters
    ----------
    core:
        Core tensor `G`, shape `(J1,...,JN)`.
    factors:
        List of factor matrices, each of shape `(I_n, J_n)`.

    Returns
    -------
    ndarray
        Reconstruction tensor of shape `(I1,...,IN)`.
    """
    N = len(factors)
    i_syms, j_syms = _symbols(N, reserved=("R",))
    core_sub = "".join(j_syms)
    out_sub  = "".join(i_syms)
    fac_subs = [f"{i_syms[n]}{j_syms[n]}" for n in range(N)]
    expr = core_sub + "," + ",".join(fac_subs) + "->" + out_sub
    return einsum(expr, core, *factors)


def tucker_core_contr(T, factors):
    """
    Contract a tensor into Tucker core coordinates via n-mode products.

    Computes:
      `C = T ×_1 A1^T ×_2 ... ×_N AN^T`

    For convenience, each element in `factors` can be provided as either
    `A_n` of shape `(I_n, J_n)` or `A_n.T` of shape `(J_n, I_n)`.

    Parameters
    ----------
    T:
        Tensor to contract, shape `(I1,...,IN)`.
    factors:
        List of matrices, each either `(I_n, J_n)` or `(J_n, I_n)`.

    Returns
    -------
    ndarray
        Contracted tensor of shape `(J1,...,JN)`.
    """
    N = len(factors)
    i_syms, j_syms = _symbols(N, reserved=("R",))
    T_sub = "".join(i_syms)
    core_sub = "".join(j_syms)

    fac_subs = []
    for n, A in enumerate(factors):
        In = T.shape[n]
        i = i_syms[n]
        j = j_syms[n]
        if A.shape[0] == In:       # (I_n, J_n)
            fac_subs.append(f"{i}{j}")
        elif A.shape[1] == In:     # (J_n, I_n)
            fac_subs.append(f"{j}{i}")
        else:
            raise ValueError(f"Factor {n} has shape {A.shape}, incompatible with mode size {In}.")

    expr = T_sub + "," + ",".join(fac_subs) + "->" + core_sub
    return einsum(expr, T, *factors)


def tucker_factor_contr(T, core, factors, n):
    """
    Contraction used to build Tucker factor numerators/denominators without unfoldings.

    Returns a matrix `M` of shape `(I_n, J_n)` corresponding to:
    \[
      M_{i_n j_n} = \sum_{i_{-n}} \sum_{j_{-n}}
        T_{i_1...i_N}\, G_{j_1...j_N}\, \prod_{m\neq n} A^{(m)}_{i_m j_m}.
    \]

    Parameters
    ----------
    T:
        Tensor to contract in data space, shape `(I1,...,IN)`.
    core:
        Core tensor `G`, shape `(J1,...,JN)`.
    factors:
        List of factor matrices `A^(m)`, each shape `(I_m, J_m)`.
    n:
        Mode index (0-based) of the output factor.

    Returns
    -------
    ndarray
        Matrix of shape `(I_n, J_n)`.
    """
    N = len(factors)
    i_syms, j_syms = _symbols(N, reserved=("R",))
    T_sub    = "".join(i_syms)
    core_sub = "".join(j_syms)

    ops = [T, core]
    subs = [T_sub, core_sub]
    for m in range(N):
        if m == n:
            continue
        ops.append(factors[m])
        subs.append(f"{i_syms[m]}{j_syms[m]}")

    expr = ",".join(subs) + "->" + f"{i_syms[n]}{j_syms[n]}"
    return einsum(expr, *ops)


# -----------------------
# B-CoMM and J-CoMM
# -----------------------
def tucker_bcomm(X, core0, factors0, beta: float, n_outer=50, eps=1e-12):
    """
    Run B-CoMM (block contraction-only MM) for nonnegative Tucker under β-divergence.

    Each outer iteration:
    1) updates every factor matrix `A^(n)` once (holding others fixed)
    2) updates the core tensor `G`

    Parameters
    ----------
    X:
        Observed tensor, shape `(I1,...,IN)`.
    core0:
        Initial core tensor, shape `(J1,...,JN)`.
    factors0:
        Initial factors list, each shape `(I_n, J_n)`.
    beta:
        β-divergence parameter.
    n_outer:
        Number of outer iterations.
    eps:
        Positivity safeguard (and denominator floor).

    Returns
    -------
    core:
        Updated core tensor.
    factors:
        Updated factor list.
    losses:
        1D array of length `n_outer` with **sum** β-divergence values.
    times:
        1D array of length `n_outer` with elapsed wall time (seconds).
    """
    core = core0.copy()
    factors = [A.copy() for A in factors0]
    N = len(factors)
    g = gamma_beta(beta)

    losses, times = [], []
    t0 = time.perf_counter()

    for _ in range(n_outer):
        # update factors
        for n in range(N):
            Xhat = tucker_reconstruct(core, factors)
            P, Q, _ = safe_PQ(X, Xhat, beta, eps)
            Num = tucker_factor_contr(P, core, factors, n)
            Den = tucker_factor_contr(Q, core, factors, n)
            mu_update_inplace(factors[n], Num, Den, g, eps)

        # update core
        Xhat = tucker_reconstruct(core, factors)
        P, Q, _ = safe_PQ(X, Xhat, beta, eps)
        NumG = tucker_core_contr(P, [A.T for A in factors])
        DenG = tucker_core_contr(Q, [A.T for A in factors])
        mu_update_inplace(core, NumG, DenG, g, eps)

        Xhat = np.maximum(tucker_reconstruct(core, factors), eps)
        losses.append(beta_divergence(X, Xhat, beta))
        times.append(time.perf_counter() - t0)

    return core, factors, np.array(losses), np.array(times)


def tucker_jcomm(X, core0, factors0, beta: float, n_outer=50, n_inner=1, eps=1e-12):
    """
    Run J-CoMM (joint majorization-minimization) for nonnegative Tucker under β-divergence.

    Outer loop:
    - set reference blocks (\\(\\widetilde G, \\widetilde A\\))
    - compute reference reconstruction and reference-powered tensors

    Inner loop:
    - update factors and core from their references using χ₁/χ₂ transforms and
      contraction-only numerators/denominators.

    Parameters
    ----------
    X:
        Observed tensor.
    core0:
        Initial core tensor.
    factors0:
        Initial factor list.
    beta:
        β-divergence parameter.
    n_outer:
        Number of outer iterations.
    n_inner:
        Number of inner steps per outer iteration.
    eps:
        Positivity safeguard (and denominator floor).

    Returns
    -------
    core:
        Updated core tensor.
    factors:
        Updated factor list.
    losses:
        1D array of length `n_outer` with **sum** β-divergence values (evaluated after outer iterations).
    times:
        1D array of length `n_outer` with elapsed wall time (seconds).
    """
    core = core0.copy()
    factors = [A.copy() for A in factors0]
    N = len(factors)
    g = gamma_beta(beta)

    losses, times = [], []
    t0 = time.perf_counter()

    for _ in range(n_outer):
        Gt = core.copy()
        At = [A.copy() for A in factors]
        Xhat_tilde = tucker_reconstruct(Gt, At)
        Ptilde, Qtilde, _ = safe_PQ(X, Xhat_tilde, beta, eps)

        G = Gt.copy()
        A = [A.copy() for A in At]

        A1 = [chi1(A[n], At[n], beta) for n in range(N)]
        A2 = [chi2(A[n], At[n], beta) for n in range(N)]
        G1 = chi1(G, Gt, beta)
        G2 = chi2(G, Gt, beta)

        for _inner in range(n_inner):
            # factors
            for n in range(N):
                NumJ = tucker_factor_contr(Ptilde, G1, A1, n)
                DenJ = tucker_factor_contr(Qtilde, G2, A2, n)
                A[n] = mu_update_from_ref(At[n], NumJ, DenJ, g, eps)
                A1[n] = chi1(A[n], At[n], beta)
                A2[n] = chi2(A[n], At[n], beta)

            # core
            NumG = tucker_core_contr(Ptilde, [B.T for B in A1])
            DenG = tucker_core_contr(Qtilde, [B.T for B in A2])
            G = mu_update_from_ref(Gt, NumG, DenG, g, eps)
            G1 = chi1(G, Gt, beta)
            G2 = chi2(G, Gt, beta)

        core, factors = G, A
        Xhat = np.maximum(tucker_reconstruct(core, factors), eps)
        losses.append(beta_divergence(X, Xhat, beta))
        times.append(time.perf_counter() - t0)

    return core, factors, np.array(losses), np.array(times)


# -----------------------
# MU unfolding baseline
# -----------------------
def tucker_mu_unfolding(X, core0, factors0, beta: float, n_outer=50, eps=1e-12):
    """
    Run an unfolding-based MU baseline for Tucker under β-divergence.

    Uses explicit unfoldings and large Kronecker products (`tucker_Bn_unfold`),
    which can be memory-heavy and numerically unstable at larger sizes.

    Parameters
    ----------
    X:
        Observed tensor.
    core0:
        Initial core tensor.
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
    core:
        Updated core tensor.
    factors:
        Updated factor list.
    losses:
        1D array of length `n_outer` with **sum** β-divergence values.
    times:
        1D array of length `n_outer` with elapsed wall time (seconds).
    """
    core = core0.copy()
    factors = [A.copy() for A in factors0]
    N = len(factors)
    g = gamma_beta(beta)

    losses, times = [], []
    t0 = time.perf_counter()

    for _ in range(n_outer):
        # factor updates
        for n in range(N):
            Xhat = tucker_reconstruct(core, factors)
            P, Q, _ = safe_PQ(X, Xhat, beta, eps)
            Pn = unfold_mode(P, n)
            Qn = unfold_mode(Q, n)
            Bn = tucker_Bn_unfold(core, factors, n)
            Num = Pn @ Bn.T
            Den = Qn @ Bn.T
            ratio = Num / (Den + eps)
            factors[n] = np.maximum(factors[n] * (ratio ** g), eps)

        # core update (using mode-0)
        Xhat = tucker_reconstruct(core, factors)
        P, Q, _ = safe_PQ(X, Xhat, beta, eps)
        P0 = unfold_mode(P, 0)
        Q0 = unfold_mode(Q, 0)
        K = kron_list([factors[m] for m in range(1, N)])
        Num0 = factors[0].T @ P0 @ K
        Den0 = factors[0].T @ Q0 @ K
        NumG = Num0.reshape(core.shape)
        DenG = Den0.reshape(core.shape)
        ratioG = NumG / (DenG + eps)
        core = np.maximum(core * (ratioG ** g), eps)

        Xhat = np.maximum(tucker_reconstruct(core, factors), eps)
        losses.append(beta_divergence(X, Xhat, beta))
        times.append(time.perf_counter() - t0)

    return core, factors, np.array(losses), np.array(times)
