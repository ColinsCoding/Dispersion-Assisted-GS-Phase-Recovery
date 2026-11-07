"""Pulse moments <t>, <t^2> and numerical group delay -- what I1 and I2 say
about the phase BEFORE any Gerchberg-Saxton iteration.

Units follow dgs.gs_core.disperse exactly: t is the sample index n, nu is
np.fft.fftfreq (cycles/sample), H(nu) = exp(i pi D nu^2). Verified sign:
a component at frequency nu arrives at  tau(nu) = -D nu  samples.

1. NUMERICAL GROUP DELAY  tau(nu) = -(1/2pi) d psi/d nu,  E~ = a e^{i psi}
   Logarithmic differentiation:  d ln E~ / d nu = a'/a + i psi'
   so  tau = -(1/2pi) Im(E~'/E~)  -- no phase unwrapping needed. And E~' is
   known EXACTLY from the FFT:  E~'(nu) = FFT[-i 2 pi n E[n]], giving
       tau(nu) = Re( FFT[n E] / FFT[E] )
   Three estimators are compared: unwrap + gradient, log-derivative with a
   finite difference, log-derivative via FFT (exact up to round-off).

2. MOMENT IDENTITY (any pulse shape, pure 2nd-order dispersion)
   With spectral weight w(nu) = |E~|^2 / sum |E~|^2:
       <t>(D)   = <tau0> - D <nu>
       sigma^2(D) = sigma0^2 - 2 D cov(nu, tau0) + D^2 var(nu)
   EXACT: sigma^2 is a quadratic in D. var(nu) comes from a spectrum
   (an OSA trace); cov(nu, tau0) is the chirp. With I1, I2 at D1, D2 and the
   spectrum, sigma0^2 and the chirp follow in closed form -- 2 equations,
   2 unknowns. Without the spectrum you need a THIRD intensity trace.
   Noise sensitivity of the chirp estimate scales as 1/|D1 - D2|: the
   measurement-diversity requirement, seen from a different angle.

3. PHYSICAL GROUP VELOCITY in fused silica (Malitson Sellmeier):
   n_g = n - lambda dn/dlambda,  v_g = c / n_g,
   D_lambda = -(lambda / c) d^2 n / d lambda^2.
   Exact derivatives via SymPy vs. finite differences: the step-size U-curve.
"""
import numpy as np
import sympy as sp

from dgs.gs_core import disperse

C_LIGHT = 299_792_458.0


# ── test pulses ──────────────────────────────────────────────────────

def chirped_pulse(N=8192, T0=40.0, chirp=0.0, cubic=0.0, nu0=0.0, shape="gauss",
                  center=None):
    """Pulse centred in an N-sample window.

    Time-domain phase  chirp*(s^2)/2 + cubic*(s^3)/6  with s = (n-center)/T0
    (radians). shape: 'gauss' (exp(-s^2/2)) or 'sech'. nu0 is a carrier
    frequency in cycles/sample.
    """
    if N < 16 or T0 <= 0:
        raise ValueError("N >= 16 and T0 > 0 required")
    if shape not in ("gauss", "sech"):
        raise ValueError("shape must be 'gauss' or 'sech'")
    n = np.arange(N, dtype=float)
    c = N / 2 if center is None else center
    s = (n - c) / T0
    amp = np.exp(-s ** 2 / 2) if shape == "gauss" else 1 / np.cosh(s)
    phase = chirp * s ** 2 / 2 + cubic * s ** 3 / 6 + 2 * np.pi * nu0 * (n - c)
    return amp * np.exp(1j * phase)


# ── 1. group delay estimators ────────────────────────────────────────

def group_delay_fft(E):
    """tau(nu) = Re(FFT[n E] / FFT[E]): exact log-derivative, no unwrapping.

    Returned in fftfreq order, units of samples.
    """
    E = np.asarray(E, dtype=complex)
    n = np.arange(len(E))
    Ek = np.fft.fft(E)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.real(np.fft.fft(n * E) / Ek)


def _spectrum_relative(E, t_ref):
    """fftshifted spectrum with the linear phase of a delay t_ref removed."""
    E = np.asarray(E, dtype=complex)
    nu = np.fft.fftfreq(len(E))
    return np.fft.fftshift(np.fft.fft(E) * np.exp(2j * np.pi * nu * t_ref))


def group_delay_logdiff_fd(E, t_ref=0.0):
    """Log-derivative with a central finite difference in nu (neighbour bins).

    t_ref removes a known bulk delay first. Without it, a pulse at the window
    centre has a linear spectral phase of pi per bin -- the Nyquist limit of
    any neighbour-bin phase estimate.
    """
    Ek = _spectrum_relative(E, t_ref)
    dnu = 1.0 / len(Ek)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = -np.imag(np.gradient(Ek, dnu) / Ek) / (2 * np.pi)
    return np.fft.ifftshift(tau) + t_ref


def group_delay_unwrap(E, t_ref=0.0):
    """Classic route: unwrap the spectral phase, differentiate numerically."""
    Ek = _spectrum_relative(E, t_ref)
    psi = np.unwrap(np.angle(Ek))
    tau = -np.gradient(psi, 1.0 / len(Ek)) / (2 * np.pi)
    return np.fft.ifftshift(tau) + t_ref


# ── 2. moments ───────────────────────────────────────────────────────

def temporal_moments(I):
    """(<t>, sigma_t^2) of an intensity trace, t = sample index."""
    I = np.asarray(I, dtype=float)
    if np.any(I < 0) or I.sum() <= 0:
        raise ValueError("intensity must be non-negative with positive total")
    n = np.arange(len(I))
    w = I / I.sum()
    m = float(w @ n)
    return m, float(w @ (n - m) ** 2)


def spectral_moments(E):
    """Spectral quantities that enter the moment identity.

    Returns dict: mean_nu, var_nu, mean_tau, cov_nu_tau, amp_term, where
    sigma0^2 = var over |E~|^2 of tau0  +  amp_term  (amplitude-shape part).
    """
    E = np.asarray(E, dtype=complex)
    N = len(E)
    Ek = np.fft.fft(E)
    nu = np.fft.fftfreq(N)
    w = np.abs(Ek) ** 2
    w = w / w.sum()
    tau = group_delay_fft(E)
    tau = np.where(np.isfinite(tau), tau, 0.0)
    m_nu = float(w @ nu)
    m_tau = float(w @ tau)
    return {
        "mean_nu": m_nu,
        "var_nu": float(w @ (nu - m_nu) ** 2),
        "mean_tau": m_tau,
        "cov_nu_tau": float(w @ ((nu - m_nu) * (tau - m_tau))),
        "var_tau": float(w @ (tau - m_tau) ** 2),
    }


def _sigma2(I):
    """sigma_t^2 without positivity checks (noisy traces may dip below 0)."""
    I = np.asarray(I, dtype=float)
    n = np.arange(len(I))
    w = I / I.sum()
    m = w @ n
    return float(w @ (n - m) ** 2)


def predicted_sigma2(D, sigma0_sq, cov_nu_tau, var_nu):
    """sigma^2(D) = sigma0^2 - 2 D cov + D^2 var(nu)."""
    D = np.asarray(D, dtype=float)
    return sigma0_sq - 2 * D * cov_nu_tau + D ** 2 * var_nu


def chirp_from_two_traces(I1, I2, D1, D2, var_nu):
    """Closed-form (sigma0^2, cov(nu, tau0)) from two intensity traces + spectrum.

    Subtract the known D^2 var(nu) term from each measured width, leaving two
    linear equations  y_i = sigma0^2 - 2 D_i cov.
    """
    if D1 == D2:
        raise ValueError("D1 == D2 gives no diversity")
    if var_nu <= 0:
        raise ValueError("var_nu must be positive")
    y1 = temporal_moments(I1)[1] - D1 ** 2 * var_nu
    y2 = temporal_moments(I2)[1] - D2 ** 2 * var_nu
    cov = -(y1 - y2) / (2 * (D1 - D2))
    sigma0_sq = y1 + 2 * D1 * cov
    return sigma0_sq, cov


def quadratic_from_three_traces(Is, Ds):
    """No spectrum? Fit sigma^2(D) through three traces: (sigma0^2, cov, var_nu)."""
    if len(Is) != 3 or len(Ds) != 3 or len(set(Ds)) != 3:
        raise ValueError("need three traces at three distinct D")
    s2 = [temporal_moments(I)[1] for I in Is]
    A = np.array([[1.0, -2 * d, d ** 2] for d in Ds])
    return tuple(np.linalg.solve(A, s2))


def chirp_estimate_spread(E, D1, D2, snr_db=30.0, n_trials=200, threshold=None,
                          seed=0):
    """Monte Carlo std of the chirp (cov) estimate with additive intensity noise.

    Noise std = peak(I) * 10^(-snr_db/20), white, on every sample. var_nu is
    taken from the noiseless spectrum (an OSA averages well).

    threshold: None -> use the noisy traces as-is (negative samples kept, so
    the estimate is unbiased but noisy). A number k -> zero every
    sample below k * noise_std before taking moments. Far-away noise is
    weighted by (t - <t>)^2 in sigma^2, so without a floor it dominates.
    Returns (std of estimate, mean of estimate, true cov).
    """
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2")
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative")
    rng = np.random.default_rng(seed)
    sm = spectral_moments(E)
    clean = [np.abs(disperse(E, D)) ** 2 for D in (D1, D2)]
    est = []
    for _ in range(n_trials):
        noisy = []
        for Ic in clean:
            sd = Ic.max() * 10 ** (-snr_db / 20)
            I = Ic + sd * rng.standard_normal(len(E))
            if threshold is not None:
                I = np.where(I > threshold * sd, I, 0.0)
            noisy.append(I)
        y = [_sigma2(I) - D ** 2 * sm["var_nu"] for I, D in zip(noisy, (D1, D2))]
        est.append(-(y[0] - y[1]) / (2 * (D1 - D2)))
    return float(np.std(est)), float(np.mean(est)), sm["cov_nu_tau"]


def wrapped_error(tau, tau_true, N):
    """Group-delay error modulo the window length N (delay is periodic in N)."""
    return (np.asarray(tau) - tau_true + N / 2) % N - N / 2


# ── 3. physical group velocity in silica ─────────────────────────────

_lam = sp.Symbol("lambda", positive=True)          # micrometres
_B = (0.6961663, 0.4079426, 0.8974794)
_Cum = (0.0684043, 0.1162414, 9.896161)
N_SILICA = sp.sqrt(1 + sum(b * _lam ** 2 / (_lam ** 2 - c ** 2) for b, c in zip(_B, _Cum)))


def silica_index(lam_um):
    """Phase index of fused silica, Malitson (1965) Sellmeier, lambda in um."""
    lam = np.asarray(lam_um, dtype=float)
    return np.sqrt(1 + sum(b * lam ** 2 / (lam ** 2 - c ** 2) for b, c in zip(_B, _Cum)))


_dn = sp.lambdify(_lam, sp.diff(N_SILICA, _lam), "numpy")
_d2n = sp.lambdify(_lam, sp.diff(N_SILICA, _lam, 2), "numpy")


def group_index_exact(lam_um):
    """n_g = n - lambda dn/dlambda with the SymPy-exact derivative."""
    lam = np.asarray(lam_um, dtype=float)
    return silica_index(lam) - lam * _dn(lam)


def group_index_fd(lam_um, h_um):
    """n_g with a central finite difference of step h (um)."""
    if h_um <= 0:
        raise ValueError("h_um must be positive")
    lam = np.asarray(lam_um, dtype=float)
    dn = (silica_index(lam + h_um) - silica_index(lam - h_um)) / (2 * h_um)
    return silica_index(lam) - lam * dn


def material_dispersion_ps_nm_km(lam_um):
    """D = -(lambda/c) d^2n/dlambda^2 in ps/(nm km) (bulk silica, no waveguide term)."""
    lam = np.asarray(lam_um, dtype=float)
    d2n_per_m2 = _d2n(lam) * 1e12                  # 1/um^2 -> 1/m^2
    D_s_per_m2 = -(lam * 1e-6) / C_LIGHT * d2n_per_m2
    return D_s_per_m2 * 1e12 / 1e9 * 1e3          # s/m^2 -> ps/(nm km)


def group_velocity(lam_um):
    """v_g = c / n_g (m/s)."""
    return C_LIGHT / group_index_exact(lam_um)
