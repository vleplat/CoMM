"""
Core numerical utilities used across CoMM.

This module centralizes:
- β-divergence objective helpers
- the multiplicative-update exponent γ(β)
- numerically safe construction of P/Q tensors
- a cached `einsum` wrapper (optionally powered by opt_einsum)
- Joint-MM transforms χ₁/χ₂
- low-allocation multiplicative update helpers
"""

import numpy as np

try:
    import opt_einsum as oe
    USE_OE = True
except Exception:
    oe = None
    USE_OE = False


# -----------------------
# beta divergence + gamma
# -----------------------
def gamma_beta(beta: float) -> float:
    """
    Compute the multiplicative-update exponent γ(β) used for β-divergence MU/MM.

    The convention matches the paper and common β-NMF practice:
    - γ(β) = 1 / (2 - β) for β < 1
    - γ(β) = 1 for 1 ≤ β < 2

    Parameters
    ----------
    beta:
        β-divergence parameter (typically in [0, 2)).

    Returns
    -------
    float
        The MU exponent γ(β).
    """
    return 1.0 / (2.0 - beta) if beta < 1.0 else 1.0


def beta_divergence(X, Y, beta: float) -> float:
    """
    Compute the **sum** of entrywise β-divergences \(D_β(X, Y)\).

    Notes
    -----
    - This function returns a **sum** over all entries, not a mean.
    - For β in {0, 1} we use the standard continuous extensions (IS and KL).
    - Assumes `Y` has strictly positive entries; use `safe_PQ` or clipping if needed.

    Parameters
    ----------
    X:
        Observed nonnegative tensor/array.
    Y:
        Model reconstruction (must be strictly positive entrywise for β≤1).
    beta:
        β-divergence parameter.

    Returns
    -------
    float
        Sum of β-divergences over all entries.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if beta == 1.0:
        return float(np.sum(np.where(X > 0, X * np.log(X / Y), 0.0) - X + Y))
    if beta == 0.0:
        return float(np.sum(X / Y - np.log(X / Y) - 1.0))

    return float(np.sum((X**beta + (beta - 1.0) * Y**beta - beta * X * (Y ** (beta - 1.0)))
                        / (beta * (beta - 1.0))))


def safe_PQ(X, Xhat, beta: float, eps: float):
    """
    Construct the MU/MM tensors P and Q safely for β-divergence updates.

    We define (elementwise):
    - \(P = X \\odot \\widehat X^{β-2}\)
    - \(Q = \\widehat X^{β-1}\)

    For β < 2, negative powers can appear; therefore we clamp `Xhat` to `eps`
    before exponentiation.

    Parameters
    ----------
    X:
        Observed nonnegative array.
    Xhat:
        Current reconstruction array.
    beta:
        β-divergence parameter.
    eps:
        Positive floor used to clip `Xhat` and avoid `0**(negative)`.

    Returns
    -------
    P:
        Array with same shape as `Xhat`.
    Q:
        Array with same shape as `Xhat`.
    Xhat_clipped:
        `Xhat` clipped to be at least `eps` entrywise.
    """
    Xhat = np.maximum(Xhat, eps)
    P = X * (Xhat ** (beta - 2.0))
    Q = (Xhat ** (beta - 1.0))
    return P, Q, Xhat


# -----------------------
# symbols for einsum
# -----------------------
def _symbols(N: int, reserved=("R",)):
    """
    Generate distinct einsum index symbols.

    Parameters
    ----------
    N:
        Number of symbols required.
    reserved:
        Characters that must not be used (e.g. 'R' reserved for rank index).

    Returns
    -------
    low, up:
        Two lists of length `N`, with lowercase and uppercase symbols respectively.

    Raises
    ------
    ValueError
        If `N` exceeds the available alphabet size after excluding reserved chars.
    """
    low = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in reserved]
    up  = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in reserved]
    if N > len(low) or N > len(up):
        raise ValueError(f"N={N} too large for available symbols.")
    return low[:N], up[:N]


# -----------------------
# cached einsum planner
# -----------------------
_EINSUM_CACHE = {}

def einsum(expr, *operands):
    """
    Evaluate an einsum contraction with optional caching of contraction expressions.

    If `opt_einsum` is available, we build and cache a compiled contraction
    expression keyed by `(expr, operand_shapes)` to avoid repeatedly re-planning
    the same contraction in tight loops.

    Parameters
    ----------
    expr:
        Einsum expression string.
    *operands:
        Arrays participating in the contraction.

    Returns
    -------
    ndarray
        Result of the contraction.
    """
    if USE_OE:
        key = (expr, tuple(op.shape for op in operands))
        fn = _EINSUM_CACHE.get(key)
        if fn is None:
            fn = oe.contract_expression(expr, *[op.shape for op in operands], optimize="optimal")
            _EINSUM_CACHE[key] = fn
        return fn(*operands)
    return np.einsum(expr, *operands, optimize=True)


# -----------------------
# Joint-MM transforms
# -----------------------
def chi1(Z, Zt, beta: float):
    """
    Joint-MM transform χ₁,β applied entrywise.

    Defined as:
    \(\\chi_{1,β}(Z, \\widetilde Z) = \\widetilde Z^{2-β} \\odot Z^{β-1}\).

    Parameters
    ----------
    Z:
        Current inner-loop variable (array).
    Zt:
        Reference variable \\(\\widetilde Z\\) (array, same shape as `Z`).
    beta:
        β-divergence parameter.

    Returns
    -------
    ndarray
        Transformed array with same shape as `Z`.
    """
    return (Zt ** (2.0 - beta)) * (Z ** (beta - 1.0))


def chi2(Z, Zt, beta: float):
    """
    Joint-MM transform χ₂,β applied entrywise.

    Defined as:
    - if β < 1: \(\\chi_{2,β}(Z, \\widetilde Z) = Z\)
    - if 1 ≤ β < 2: \(\\chi_{2,β}(Z, \\widetilde Z) = Z^{β} \\odot \\widetilde Z^{1-β}\)

    Parameters
    ----------
    Z:
        Current inner-loop variable (array).
    Zt:
        Reference variable \\(\\widetilde Z\\) (array, same shape as `Z`).
    beta:
        β-divergence parameter.

    Returns
    -------
    ndarray
        Transformed array with same shape as `Z`.
    """
    if beta < 1.0:
        return Z
    return (Z ** beta) * (Zt ** (1.0 - beta))


# -----------------------
# fast MU update helpers
# -----------------------
def mu_update_inplace(A, Num, Den, g, eps, clip_ratio=1e10):
    """
    In-place multiplicative update with clipping and positivity safeguard.

    Performs:
    \(A \\leftarrow \\max( A \\odot (\\mathrm{Num}/(\\mathrm{Den}+\\varepsilon))^{g}, \\varepsilon)\)

    The update is done in-place to minimize temporaries. The ratio is optionally
    clipped to limit numerical blow-ups.

    Parameters
    ----------
    A:
        Array to update in-place.
    Num, Den:
        Numerator and denominator arrays (broadcastable to `A`).
    g:
        Exponent γ(β).
    eps:
        Positive floor; also used in the denominator as `Den + eps`.
    clip_ratio:
        If not None, clip ratio to `[0, clip_ratio]` before exponentiation.

    Returns
    -------
    ndarray
        Reference to the updated `A` (same object).
    """
    ratio = Num / (Den + eps)
    if clip_ratio is not None:
        np.clip(ratio, 0.0, clip_ratio, out=ratio)
    np.power(ratio, g, out=ratio)
    A *= ratio
    np.maximum(A, eps, out=A)
    return A


def mu_update_from_ref(Aref, Num, Den, g, eps, clip_ratio=1e10):
    """
    Multiplicative update computed from a fixed reference array.

    Returns:
    \(A_{new} = \\max( A_{ref} \\odot (\\mathrm{Num}/(\\mathrm{Den}+\\varepsilon))^{g}, \\varepsilon)\)

    This is used by joint-MM inner updates where the reference block
    \\(\\widetilde A\\) remains fixed.

    Parameters
    ----------
    Aref:
        Reference array \\(\\widetilde A\\).
    Num, Den:
        Numerator and denominator arrays (broadcastable to `Aref`).
    g:
        Exponent γ(β).
    eps:
        Positive floor; also used in the denominator as `Den + eps`.
    clip_ratio:
        If not None, clip ratio to `[0, clip_ratio]` before exponentiation.

    Returns
    -------
    ndarray
        Updated array (new allocation).
    """
    ratio = Num / (Den + eps)
    if clip_ratio is not None:
        np.clip(ratio, 0.0, clip_ratio, out=ratio)
    np.power(ratio, g, out=ratio)
    Anew = Aref.copy()
    Anew *= ratio
    np.maximum(Anew, eps, out=Anew)
    return Anew


def to_float32(x):
    """
    Convert input arrays (or a list of arrays) to `np.float32`.

    Parameters
    ----------
    x:
        Array-like or list of array-like objects.

    Returns
    -------
    ndarray or list[ndarray]
        Converted object(s) with dtype `float32`.
    """
    if isinstance(x, list):
        return [to_float32(v) for v in x]
    return np.asarray(x, dtype=np.float32)
