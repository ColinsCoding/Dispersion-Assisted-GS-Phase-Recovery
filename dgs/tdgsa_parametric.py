"""Parametric-phase-model fix for the TDGSA flat-phase degeneracy found in
notebooks/ece279_tdgsa_recreation.ipynb.

That investigation showed free-form per-sample GS phase retrieval (one free
phase value per time sample) has a huge degenerate solution space for
smooth, large-phase-excursion pulses: 2 dispersion planes, 3 planes, and
2000 GPU-batched random restarts all converged to spurious solutions
(mostly jagged/noisy, one smooth-but-flat) that satisfy the intensity
measurements almost exactly while being nowhere near the true phase.

The fix here is a DIFFERENT search space, not a better optimizer over the
same one: restrict the phase to a low-order polynomial in normalized time,
phi(tau) = sum_k theta_k * tau^k (a handful of real numbers instead of one
free value per sample). Jagged noise isn't representable in this family, so
that entire degenerate region is eliminated by construction.

Two distinct failure modes were found and handled:

  1. Gas-cell pulse (pure quadratic phase): a direct gradient descent
     (Adam) from theta=0 converges to the true coefficients essentially
     exactly (loss ~1e-14) -- this case's reduced loss landscape is
     well-behaved.
  2. Cubic-phase pulse: gradient descent from theta=0 gets stuck in a
     shallow local minimum (loss ~1e-4, wrong coefficients) even though the
     TRUE coefficients are a much sharper, much lower-loss global minimum
     (verified directly by a loss scan: true k=0.06 gives loss=1.5e-13,
     13 orders of magnitude below any nearby local minimum). The reduced
     parameter space is now cheap enough (1-4 numbers, not 4000) that a
     GPU-batched grid/multi-start search over it -- something that
     completely failed in the original per-sample search space -- finds
     the sharp global minimum reliably.
"""

import numpy as np
import torch

C_LIGHT = 299_792_458.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gdd_ps2(D_ps_per_nm, lambda_nm=1550.0):
    lam_m = lambda_nm * 1e-9
    D_s_per_m = D_ps_per_nm * 1e-3
    beta2L_s2 = -D_s_per_m * lam_m ** 2 / (2 * np.pi * C_LIGHT)
    return beta2L_s2 * 1e24


def _H(D_ps_per_nm, dt_ps, N, device, lambda_nm=1550.0):
    f = torch.fft.fftfreq(N, dt_ps, device=device)
    return torch.exp(1j * 0.5 * _gdd_ps2(D_ps_per_nm, lambda_nm) * (2 * np.pi * f) ** 2).to(torch.complex64)


def _disperse(field, H):
    return torch.fft.ifft(torch.fft.fft(field, dim=-1) * H, dim=-1)


def _loss(theta_batch, powers, envelope, H1, H2, I1, I2):
    """theta_batch: (B, degree+1). powers: (degree+1, N). Returns (B,) loss."""
    phase = theta_batch @ powers                                    # (B, N)
    E = envelope.unsqueeze(0).to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
    I1p = torch.abs(_disperse(E, H1)) ** 2
    I2p = torch.abs(_disperse(E, H2)) ** 2
    return torch.mean((I1p - I1.unsqueeze(0)) ** 2, dim=-1) + torch.mean((I2p - I2.unsqueeze(0)) ** 2, dim=-1)


def fit_parametric_phase(I1, I2, D1, D2, dt_ps, tau, envelope, degree=3,
                          n_grid=2000, coef_range=0.3, n_polish=500, lr=0.01,
                          lambda_nm=1550.0, device=DEVICE, seed=0):
    """Recover phase as a degree-k polynomial in normalized time `tau`, by
    grid-searching the leading (highest-degree) coefficient on a GPU batch
    (cheap: n_grid candidates evaluated in one shot), then polishing the
    single best candidate with gradient descent.

    Parameters
    ----------
    I1, I2   : (N,) measured intensities at dispersions D1, D2 (ps/nm)
    dt_ps    : sample spacing in ps
    tau      : (N,) normalized time axis (e.g. t_ns / T0_ns) -- the caller's
               choice of normalization; coefficients are returned in these units
    envelope : (N,) known amplitude envelope (assumed known/measured separately
               -- this module recovers PHASE only, matching the "phase
               retrieval" problem statement, not blind amplitude+phase retrieval)
    degree   : polynomial degree (only the top coefficient is grid-searched;
               lower-order coefficients start at 0 and are refined during polish)
    n_grid   : number of leading-coefficient candidates in the grid search
    coef_range : grid search covers [-coef_range, +coef_range] for the leading coefficient
    n_polish : gradient-descent steps after grid search

    Returns
    -------
    theta  : (degree+1,) numpy array, phase = sum_k theta[k] * tau^k
    loss   : final loss (should be many orders of magnitude below the grid's
             runner-up losses if the pulse's true phase is genuinely low-order)
    """
    I1_t = torch.as_tensor(I1, dtype=torch.float32, device=device)
    I2_t = torch.as_tensor(I2, dtype=torch.float32, device=device)
    tau_t = torch.as_tensor(tau, dtype=torch.float32, device=device)
    env_t = torch.as_tensor(envelope, dtype=torch.float32, device=device)
    N = tau_t.shape[0]

    H1 = _H(D1, dt_ps, N, device, lambda_nm)
    H2 = _H(D2, dt_ps, N, device, lambda_nm)
    powers = torch.stack([tau_t ** k for k in range(degree + 1)])   # (degree+1, N)

    # grid search: sweep only the leading coefficient, lower ones fixed at 0
    k_grid = torch.linspace(-coef_range, coef_range, n_grid, device=device)
    theta_grid = torch.zeros(n_grid, degree + 1, device=device)
    theta_grid[:, degree] = k_grid
    with torch.no_grad():
        losses = _loss(theta_grid, powers, env_t, H1, H2, I1_t, I2_t)
    best_idx = int(torch.argmin(losses))
    theta = theta_grid[best_idx].clone().requires_grad_(True)

    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(n_polish):
        opt.zero_grad()
        loss = _loss(theta.unsqueeze(0), powers, env_t, H1, H2, I1_t, I2_t)[0]
        loss.backward()
        opt.step()

    return theta.detach().cpu().numpy(), float(loss.detach())


def evaluate_polynomial_phase(theta, tau):
    """phase(tau) = sum_k theta[k] * tau^k, as a plain numpy function."""
    tau = np.asarray(tau)
    return sum(theta[k] * tau ** k for k in range(len(theta)))


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    fs = 200e9
    N = int(round(20e-9 * fs))
    t_ns = np.linspace(-10, 10, N)
    dt_ps = float((t_ns[1] - t_ns[0]) * 1000.0)
    T0_ns = 2.0
    tau = t_ns / T0_ns

    print("=== Gas cell (quadratic phase), D1=-353, D2=-872 ps/nm ===")
    envelope_gas = np.exp(-0.5 * tau ** 2)
    true_phase_gas = -0.5 * tau ** 2
    E_true_gas = envelope_gas * np.exp(1j * true_phase_gas)
    Ht1 = _H(-353.0, dt_ps, N, DEVICE).cpu().numpy()
    Ht2 = _H(-872.0, dt_ps, N, DEVICE).cpu().numpy()
    I1_gas = np.abs(np.fft.ifft(np.fft.fft(E_true_gas) * Ht1)) ** 2
    I2_gas = np.abs(np.fft.ifft(np.fft.fft(E_true_gas) * Ht2)) ** 2

    theta_gas, loss_gas = fit_parametric_phase(I1_gas, I2_gas, -353.0, -872.0, dt_ps,
                                                tau, envelope_gas, degree=2)
    print(f"  recovered theta={theta_gas}  (true: [0, 0, -0.5])  final loss={loss_gas:.3e}")

    print("\n=== Cubic-phase pulse, D1=-600, D2=-900 ps/nm ===")
    tau5 = t_ns / (5 * T0_ns)
    envelope_cubic = np.exp(-0.5 * tau5 ** 2)
    true_phase_cubic = 0.06 * tau ** 3
    E_true_cubic = envelope_cubic * np.exp(1j * true_phase_cubic)
    Ht1c = _H(-600.0, dt_ps, N, DEVICE).cpu().numpy()
    Ht2c = _H(-900.0, dt_ps, N, DEVICE).cpu().numpy()
    I1_cubic = np.abs(np.fft.ifft(np.fft.fft(E_true_cubic) * Ht1c)) ** 2
    I2_cubic = np.abs(np.fft.ifft(np.fft.fft(E_true_cubic) * Ht2c)) ** 2

    theta_cubic, loss_cubic = fit_parametric_phase(I1_cubic, I2_cubic, -600.0, -900.0, dt_ps,
                                                    tau, envelope_cubic, degree=3)
    print(f"  recovered theta={theta_cubic}  (true: [0, 0, 0, 0.06])  final loss={loss_cubic:.3e}")
