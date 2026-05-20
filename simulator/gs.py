"""
simulator.gs — Time-Domain Gerchberg-Saxton (TD-GS) phase retrieval.

Algorithm (Solli 2009 / Jalali Lab variant)
-------------------------------------------
Given two intensity measurements

    I1[n] = |A(t_n, z=L1)|^2    after dispersion beta2_L1
    I2[n] = |A(t_n, z=L2)|^2    after dispersion beta2_L2

find the original complex field u[n] such that both constraints are
satisfied simultaneously.

Iteration (one full step):
    1. Apply forward dispersion:  v1 = D(u, beta2_L1)
    2. Replace modulus:           v1 <- sqrt(I1) * v1 / |v1|
    3. Invert dispersion:         u  = D(v1, -beta2_L1)
    4. Apply forward dispersion:  v2 = D(u, beta2_L2)
    5. Replace modulus:           v2 <- sqrt(I2) * v2 / |v2|
    6. Invert dispersion:         u  = D(v2, -beta2_L2)

Optional: apply a support constraint after step 6 (zero energy outside
the known pulse window).

Multiple random-phase restarts are run and the best solution (lowest
residual) is returned.

Global-phase ambiguity
----------------------
Phase retrieval is invariant to e^(i*phi_0).  All RMSE computations
align the recovered field to the reference via the optimal global phase.
"""

from __future__ import annotations
import numpy as np
from .dispersion import batch_propagate, propagate


def _align_global_phase(u_rec: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Rotate u_rec by the optimal global phase to minimise |u_rec - u_ref|."""
    inner = np.vdot(u_ref, u_rec)            # conjugate-linear in first arg
    if abs(inner) < 1e-30:
        return u_rec
    return u_rec * (inner / abs(inner)).conj()


def _rmse(u_rec: np.ndarray, u_ref: np.ndarray) -> float:
    """Phase-aligned RMSE normalised by the RMS of the reference."""
    u_aligned = _align_global_phase(u_rec, u_ref)
    norm = np.sqrt(np.mean(np.abs(u_ref) ** 2))
    if norm == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.abs(u_aligned - u_ref) ** 2)) / norm)


def _residual(
    u: np.ndarray,
    amp1: np.ndarray,
    amp2: np.ndarray,
    t_axis: np.ndarray,
    beta2_L1: float,
    beta2_L2: float,
) -> float:
    """Modulus residual (no ground truth needed) — used for restart selection."""
    v1 = propagate(u, t_axis, beta2_L1)
    v2 = propagate(u, t_axis, beta2_L2)
    r1 = np.sqrt(np.mean((np.abs(v1) - amp1) ** 2))
    r2 = np.sqrt(np.mean((np.abs(v2) - amp2) ** 2))
    return float(r1 + r2)


def td_gs(
    I1: np.ndarray,
    I2: np.ndarray,
    t_axis: np.ndarray,
    beta2_L1: float,
    beta2_L2: float,
    *,
    n_restarts: int = 8,
    n_iter: int = 250,
    support_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    u_true: np.ndarray | None = None,
) -> dict:
    """Run TD-GS phase retrieval with multiple random-phase restarts.

    Parameters
    ----------
    I1, I2 : ndarray, real, shape (N,)
        Measured intensity after dispersions beta2_L1 and beta2_L2.
    t_axis : ndarray, real, shape (N,)
        Uniformly-spaced time axis (seconds).
    beta2_L1, beta2_L2 : float
        Accumulated GVD for the two channels (s²/rad).
        Ratio |beta2_L2 / beta2_L1| should exceed 1.33 (Solli 2009).
    n_restarts : int
        Number of independent random-phase restarts.
    n_iter : int
        GS iterations per restart.
    support_mask : ndarray, bool, shape (N,), optional
        If given, zeros out u outside the support after each full step.
        Helps when the pulse occupies a fraction of the time window.
    rng : numpy Generator, optional
        Random number generator for reproducibility.
    u_true : ndarray, complex, shape (N,), optional
        Ground-truth field for RMSE tracking (research / debug only).

    Returns
    -------
    result : dict with keys:
        ``u_best``      — best recovered field, shape (N,)
        ``residual``    — modulus residual of best solution (no ground truth needed)
        ``rmse``        — phase-aligned RMSE vs u_true, or None if not provided
        ``rmse_history``— list of per-restart final RMSE (only if u_true given)
        ``res_history`` — list of per-restart final residuals

    Raises
    ------
    ValueError
        If I1/I2 contain negative values or have incompatible shapes.
    """
    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)
    N = len(I1)
    if I1.shape != I2.shape:
        raise ValueError(f"I1 shape {I1.shape} != I2 shape {I2.shape}")
    if len(t_axis) != N:
        raise ValueError(f"t_axis length {len(t_axis)} != I1 length {N}")
    if np.any(I1 < 0) or np.any(I2 < 0):
        raise ValueError("Intensity arrays must be non-negative")

    amp1 = np.sqrt(np.maximum(I1, 0.0))
    amp2 = np.sqrt(np.maximum(I2, 0.0))

    if rng is None:
        rng = np.random.default_rng(0)

    # Seed the initial modulus from the first measurement after back-propagation
    u_seed_mag = np.abs(propagate(amp1.astype(complex), t_axis, -beta2_L1))

    best_u = None
    best_res = np.inf
    res_history = []
    rmse_history = [] if u_true is not None else None

    for _ in range(n_restarts):
        phase = rng.uniform(-np.pi, np.pi, N)
        u = u_seed_mag * np.exp(1j * phase)

        for _ in range(n_iter):
            # --- constraint 1 ---
            v1 = propagate(u, t_axis, beta2_L1)
            mag1 = np.abs(v1)
            safe = mag1 > 1e-30 * mag1.max()
            v1[safe] = amp1[safe] * v1[safe] / mag1[safe]
            u = propagate(v1, t_axis, -beta2_L1)

            # --- constraint 2 ---
            v2 = propagate(u, t_axis, beta2_L2)
            mag2 = np.abs(v2)
            safe = mag2 > 1e-30 * mag2.max()
            v2[safe] = amp2[safe] * v2[safe] / mag2[safe]
            u = propagate(v2, t_axis, -beta2_L2)

            if support_mask is not None:
                u[~support_mask] = 0.0

        res = _residual(u, amp1, amp2, t_axis, beta2_L1, beta2_L2)
        res_history.append(res)
        if u_true is not None:
            rmse_history.append(_rmse(u, u_true))
        if res < best_res:
            best_res = res
            best_u = u.copy()

    rmse_best = _rmse(best_u, u_true) if u_true is not None else None

    return {
        "u_best": best_u,
        "residual": best_res,
        "rmse": rmse_best,
        "rmse_history": rmse_history,
        "res_history": res_history,
    }
