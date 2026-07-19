"""Photosynthetic light-harvesting energy transfer: linear algebra, a
differential equation, and machine learning, all on the same physical system.

Light absorbed by one chlorophyll hops between pigments by Forster resonance
energy transfer (FRET) before reaching the reaction center. Population
dynamics among N chromophores obey a LINEAR ODE

    dp/dt = K p

where K is an N x N rate matrix (off-diagonal = FRET hop rates, diagonal =
decay rates). That makes the whole problem:

  * LINEAR ALGEBRA -- K's eigenvalues are the system's normal decay modes
    (the same "differential operator -> eigenvalue problem" move as
    dgs.bessel_linalg, just for a transfer-rate matrix instead of a
    Laplacian).
  * A DIFFERENTIAL EQUATION -- solved in closed form via matrix
    exponential / eigendecomposition, p(t) = sum_i c_i v_i exp(lambda_i t).
  * MACHINE LEARNING IN BIOLOGY -- given only a noisy fluorescence decay
    curve (the experimentally measurable quantity), a gradient-descent fit
    recovers the underlying rate matrix -- the same autograd machinery as
    dgs.differentiable_optics_tutorial, applied to a biological inverse
    problem instead of an optical one.
"""

import numpy as np


# -- Forster resonance energy transfer rate -------------------------------------

def forster_rate(r, r0, tau_donor):
    """FRET hop rate k = (1/tau_donor) * (R0/r)^6 -- the famous inverse-sixth-
    power distance law (R0 = Forster radius, the distance at which transfer
    and intrinsic decay are equally likely)."""
    return (1.0 / tau_donor) * (r0 / r) ** 6


# -- Linear algebra: the rate matrix and its eigenmodes -------------------------

def build_rate_matrix(transfer_rates, decay_rates):
    """Build the N x N population-transfer rate matrix K for dp/dt = K p.

    transfer_rates: dict {(i, j): k_ij} -- FRET rate from chromophore i to j.
    decay_rates: length-N array of each chromophore's intrinsic decay rate
    (fluorescence + non-radiative loss, with NO transfer).

    K[j, i] += k_ij (population flows into j from i), K[i, i] -= k_ij (i loses
    population) and -= decay_rates[i] (intrinsic decay) -- a standard
    chemical-kinetics rate-matrix construction, but it is exactly the
    "build a matrix from a graph of couplings" move linear algebra always
    reduces ODEs on a network to.
    """
    N = len(decay_rates)
    K = np.zeros((N, N))
    for (i, j), k_ij in transfer_rates.items():
        K[j, i] += k_ij
        K[i, i] -= k_ij
    for i in range(N):
        K[i, i] -= decay_rates[i]
    return K


def eigen_decomposition_modes(K):
    """Eigenvalues/eigenvectors of the rate matrix K -- the normal decay
    modes of the coupled system. Real parts of eigenvalues are <= 0 for a
    physical (population-conserving-or-decaying) rate matrix; returns them
    sorted slowest-decaying (closest to zero) first, since that mode
    dominates the long-time fluorescence tail."""
    eigvals, eigvecs = np.linalg.eig(K)
    order = np.argsort(-eigvals.real)   # least negative (slowest decay) first
    return eigvals[order], eigvecs[:, order]


# -- Differential equation: closed-form solution via eigendecomposition --------

def solve_population_dynamics(K, p0, t):
    """Solve dp/dt = K p, p(0) = p0, at times `t` via eigendecomposition:
    p(t) = V diag(exp(lambda t)) V^{-1} p0 -- the exact linear-ODE solution,
    no numerical integrator needed because K is constant in time."""
    eigvals, eigvecs = np.linalg.eig(K)
    coeffs = np.linalg.solve(eigvecs, p0)
    t = np.asarray(t, dtype=float)
    # p(t)_n = sum_i coeffs[i] * eigvecs[n,i] * exp(eigvals[i] * t)
    modes = np.exp(np.outer(t, eigvals))                      # (T, N) exp(lambda_i t)
    p_t = np.einsum("ni,ti,i->tn", eigvecs, modes, coeffs)
    return p_t.real


def fluorescence_signal(p_t, radiative_rates):
    """Observable fluorescence intensity F(t) = sum_n k_rad[n] * p_n(t) -- what
    a real time-resolved fluorescence experiment actually measures (you don't
    see individual chromophore populations, only their weighted photon output)."""
    return p_t @ np.asarray(radiative_rates)


# -- Machine learning in biology: recover K from a fluorescence decay curve ----

def fit_transfer_rate_from_decay(t, F_observed, p0, radiative_rates, n_steps=400, lr=0.1, seed=0):
    """Given only the observable fluorescence curve F(t) (not the individual
    chromophore populations), fit the donor->acceptor FRET rate k by gradient
    descent -- the inverse problem real time-resolved fluorescence
    spectroscopy solves to measure inter-chromophore distances.

    Builds K(k) and propagates p(t) = matrix_exp(K*t) p0 entirely in torch
    (torch.linalg.matrix_exp is differentiable), so autograd gives
    d(loss)/d(k) directly -- the same gradient-descent inverse-problem
    machinery as dgs.differentiable_optics_tutorial, applied here to a
    biological rate constant instead of an optical phase.
    """
    import torch

    t_torch = torch.tensor(np.asarray(t, dtype=np.float32))
    F_obs = torch.tensor(np.asarray(F_observed, dtype=np.float32))
    p0_t = torch.tensor(np.asarray(p0, dtype=np.float32))
    rad = torch.tensor(np.asarray(radiative_rates, dtype=np.float32))

    log_k = torch.tensor(float(np.log(1.0)), requires_grad=True)  # optimize in log-space, k>0
    opt = torch.optim.Adam([log_k], lr=lr)

    def model_curve(k):
        K = torch.zeros((2, 2), dtype=torch.float32)
        K[1, 0] = k
        K[0, 0] = -k - rad[0]
        K[1, 1] = -rad[1]
        # propagate at each time point via matrix_exp(K*t) p0
        p_t = torch.stack([torch.linalg.matrix_exp(K * ti) @ p0_t for ti in t_torch])
        return p_t @ rad

    history = []
    for _ in range(n_steps):
        opt.zero_grad()
        k = torch.exp(log_k)
        F_model = model_curve(k)
        loss = torch.mean((F_model - F_obs) ** 2)
        loss.backward()
        opt.step()
        history.append(loss.item())

    k_fit = float(torch.exp(log_k).item())
    return {"k_fit": k_fit, "loss": history[-1], "loss_history": history}
