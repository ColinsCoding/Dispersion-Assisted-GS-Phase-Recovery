"""3D time-stretch imaging, the angstrom limit of calculus, and a Schrodinger particle classifier.

In plain words:

1. 3D STEAM. Jalali's STEAM camera (Goda, Tsia, Jalali, Nature 2009) maps one image LINE
   onto the spectrum of a laser pulse (a diffraction grating sends each wavelength to a
   different x), then stretches the pulse in a dispersive fiber so the spectrum comes out
   as a time trace (the dispersive Fourier transform). One pulse = one line; the next pulse
   sees the object moved by flow = the next y. Adding a reference arm makes the spectrum an
   interferogram whose PHASE is the surface height: phi = 4 pi h / lambda in reflection.
   That is z. So x <- wavelength, y <- pulse number, z <- interferometric phase: a 3D
   surface per few hundred pulses, at the pulse rate (tens of MHz lines per second).
   This module simulates it and recovers a 20 nm bump and a 3.1 angstrom atomic step.

2. The angstrom limit. "Take h -> 0" is calculus; a computer cannot. The central second
   difference has truncation error ~h^2 f''''/12 but round-off ~4 eps |f| / h^2, so the
   error is smallest at h* ~ (48 eps)^{1/4} and grows again below it. Physical "limits"
   are similar: interferometry resolves heights far below a wavelength (lambda dphi / 4 pi
   reaches angstroms), but the continuum Schrodinger model itself stops at about one
   angstrom, the size of an atom.

3. Particle classifier. Solve the 1D Schrodinger equation (finite differences, angstrom
   and eV units) for an electron, muon, proton or deuteron in a box, a harmonic trap or a
   finite well whose sizes jitter like fabricated devices. Features from the first four
   energy levels -- log E1 and the ratios E2/E1, E3/E1, E4/E1 -- feed a logistic-regression
   classifier. The ratios identify the POTENTIAL (box 4, 9, 16; harmonic 3, 5, 7; V-shaped
   linear well 2.30, 3.19, 4.01 from Airy zeros -- the triangular well of a MOSFET inversion
   layer); the scale identifies the MASS, with exponent -1, -1/2, -1/3 respectively. But the
   scale also depends on the device size, so a heavier particle in a smaller well can imitate
   a lighter one: proton vs deuteron (mass ratio 2) gets confused as geometry jitter grows,
   first in the linear well, where the mass dependence is weakest (2^{1/3} = 1.26).
   (A deep finite well is not used: for heavy particles its spectrum is indistinguishable
   from a box.)
   Same pipeline as time-stretch cell classification: physics forward model -> features
   -> classifier, limited by the degeneracies of the physics.

NumPy, SciPy (eigh_tridiagonal), scikit-learn for the classifier.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

HBAR2_2ME = 3.80998212   # hbar^2 / (2 m_e) in eV angstrom^2
MASS = {"electron": 1.0, "muon": 206.7682830, "proton": 1836.15267343, "deuteron": 3670.48296788}
POTENTIALS = ("box", "harmonic", "linear")


def _pos(name, v):
    if not np.all(np.isfinite(v)) or np.any(np.asarray(v) <= 0):
        raise ValueError(f"{name} must be positive, got {v}")


# ---------------------------------------------------------------- 1. 3D STEAM (spectral-encoded interferometry)

def wavenumber_grid(n=1024, lam_min=1530e-9, lam_max=1570e-9):
    """Uniform grid in k = 2 pi / lambda across the pulse bandwidth (what a time-stretch DFT samples)."""
    _pos("lam_min", lam_min)
    if lam_max <= lam_min:
        raise ValueError("lam_max must exceed lam_min")
    return np.linspace(2 * np.pi / lam_max, 2 * np.pi / lam_min, n)


def spectral_encoding_x(k, field_of_view):
    """Linear map of the band onto x in [0, field_of_view] (diffraction-grating spectral encoding)."""
    _pos("field_of_view", field_of_view)
    return (k - k.min()) / (k.max() - k.min()) * field_of_view


def demo_surface(x, y, bump_height=20e-9, bump_sigma=3e-6, step_height=3.1e-10):
    """Height map: a Gaussian 'cell' bump (upper right) plus an atomic step (Si(111)-like,
    ~3.1 angstrom) along y = 0. The bump is kept away from the left half, where the step is measured."""
    X, Y = np.meshgrid(x, y)
    x0 = x.min() + 0.75 * (x.max() - x.min())
    y0 = y.min() + 0.75 * (y.max() - y.min())
    return (bump_height * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * bump_sigma ** 2))
            + step_height * (Y > 0))


def step_estimate(scan, x_frac=(0.1, 0.4), y_gap=3e-6):
    """Step height from the retrieved map: mean over the left strip above minus below y = 0
    (excluding |y| < y_gap), with its standard error. Averaging many pixels is what reaches angstroms."""
    x, y, R = scan["x"], scan["y"], scan["R"]
    cols = (x > x.max() * x_frac[0]) & (x < x.max() * x_frac[1])
    # lines are the independent samples: each carries its own referencing offset, shared by all
    # its pixels, so the standard error must come from per-line means, not from pixels
    hi, lo = R[y > y_gap][:, cols].mean(axis=1), R[y < -y_gap][:, cols].mean(axis=1)
    est = hi.mean() - lo.mean()
    se = np.sqrt(hi.var(ddof=1) / hi.size + lo.var(ddof=1) / lo.size)
    return float(est), float(se)


def interferogram(k, h_line, path_difference, visibility=0.8, snr_db=30.0, rng=None):
    """Spectral interferogram of one line: 1 + V cos(k (d + 2 h(x(k)))) plus white noise at the given SNR."""
    _pos("path_difference", path_difference)
    rng = np.random.default_rng(0) if rng is None else rng
    I = 1 + visibility * np.cos(k * (path_difference + 2 * h_line))
    return I + 10 ** (-snr_db / 20) * rng.standard_normal(k.size)


def retrieve_height_line(k, I, path_difference, sideband_halfwidth=None):
    """Fourier-transform fringe analysis (Takeda): isolate the +1 sideband, take its phase,
    remove the carrier k*d, divide by 2k. Returns height (m), up to a constant."""
    n = k.size
    F = np.fft.fft(I - I.mean())
    dk = k[1] - k[0]
    f = np.fft.fftfreq(n, dk) * 2 * np.pi          # conjugate variable to k (a length)
    hw = path_difference / 2 if sideband_halfwidth is None else sideband_halfwidth
    mask = np.abs(f - path_difference) < hw
    analytic = np.fft.ifft(F * mask)
    phi = np.unwrap(np.angle(analytic))
    resid = phi - k * path_difference
    resid -= 2 * np.pi * np.round((resid[n // 2]) / (2 * np.pi))
    return resid / (2 * k)


def steam3d_scan(n_lines=200, n_k=1024, field_of_view=50e-6, path_difference=6e-3, snr_db=30.0,
                 step_height=3.1e-10, seed=0):
    """Simulate a 3D scan: one interferogram per pulse (line), lines spaced by flow, heights retrieved.
    Returns dict with x, y, true height, retrieved height (each line referenced to its own left edge)."""
    if n_lines < 2 or n_k < 64:
        raise ValueError("need n_lines >= 2 and n_k >= 64")
    rng = np.random.default_rng(seed)
    k = wavenumber_grid(n_k)
    x = spectral_encoding_x(k, field_of_view)
    y = np.linspace(-field_of_view / 2, field_of_view / 2, n_lines)
    H = demo_surface(x, y, step_height=step_height)
    R = np.empty_like(H)
    for j in range(n_lines):
        I = interferogram(k, H[j], path_difference, snr_db=snr_db, rng=rng)
        R[j] = retrieve_height_line(k, I, path_difference)
    # absolute piston per line is unobservable in single-line FT analysis; reference each line
    # to a known flat substrate strip. The strip starts INSIDE the band: Fourier fringe analysis
    # distorts the first/last few percent of pixels (Gibbs-like edge effects).
    edge = slice(n_k // 50, n_k // 8)
    R = R - R[:, edge].mean(axis=1, keepdims=True) + H[:, edge].mean(axis=1, keepdims=True)
    return dict(k=k, x=x, y=y, H=H, R=R)


def height_precision(lam, phase_noise_rad, n_avg=1):
    """delta h = lambda delta phi / (4 pi sqrt(n_avg)) for reflection interferometry."""
    _pos("lam", lam); _pos("phase_noise_rad", phase_noise_rad); _pos("n_avg", n_avg)
    return lam * phase_noise_rad / (4 * np.pi * np.sqrt(n_avg))


# ---------------------------------------------------------------- 2. the limit h -> 0 on a computer

def second_difference_error(h, x0=0.7):
    """|[f(x+h) - 2 f(x) + f(x-h)]/h^2 - f''(x)| for f = sin, in double precision."""
    h = np.asarray(h, float)
    fd = (np.sin(x0 + h) - 2 * np.sin(x0) + np.sin(x0 - h)) / h ** 2
    return np.abs(fd + np.sin(x0))


def error_model(h, x0=0.7, eps=np.finfo(float).eps):
    """Truncation h^2 |f''''| / 12 plus round-off 4 eps |f| / h^2 (f = sin)."""
    h = np.asarray(h, float)
    return h ** 2 * abs(np.sin(x0)) / 12 + 4 * eps * abs(np.sin(x0)) / h ** 2


def optimal_step(eps=np.finfo(float).eps):
    """h* minimizing the model: (48 eps)^{1/4} (for |f| = |f''''|)."""
    return (48 * eps) ** 0.25


# ---------------------------------------------------------------- 3. Schrodinger particle classifier

def lowest_levels(V, dx, mass_ratio, n_levels=4):
    """Lowest eigenvalues (eV) of -hbar^2/2m d2/dx2 + V on a uniform grid (angstrom), Dirichlet walls."""
    V = np.asarray(V, float)
    if V.ndim != 1 or V.size < n_levels + 2:
        raise ValueError("V must be 1D with more points than requested levels")
    _pos("dx", dx); _pos("mass_ratio", mass_ratio)
    c = HBAR2_2ME / mass_ratio / dx ** 2
    return eigh_tridiagonal(V + 2 * c, -c * np.ones(V.size - 1), select="i",
                            select_range=(0, n_levels - 1), eigvals_only=True)


def box_exact(n, L, mass_ratio):
    """Infinite well E_n = (hbar^2/2m) (n pi / L)^2 in eV, L in angstrom."""
    return HBAR2_2ME / mass_ratio * (np.asarray(n) * np.pi / L) ** 2


def linear_exact(n_levels, F, mass_ratio):
    """V = F|x|: even states at zeros of Ai', odd at zeros of Ai, E = |zero| (hbar^2 F^2 / 2m)^{1/3}."""
    from scipy.special import ai_zeros
    a, ap, _, _ = ai_zeros(n_levels)
    z = np.sort(np.abs(np.concatenate([a, ap])))[:n_levels]
    return z * (HBAR2_2ME / mass_ratio * F ** 2) ** (1 / 3)


def harmonic_exact(n, k_spring, mass_ratio):
    """E_n = (n + 1/2) hbar omega, hbar omega = sqrt(2 (hbar^2/2m) k) in eV (k in eV/angstrom^2)."""
    return (np.asarray(n) + 0.5) * np.sqrt(2 * HBAR2_2ME / mass_ratio * k_spring)


NOMINAL = dict(box_L=4.0, harmonic_k=10.0, linear_F=5.0)


def spectrum(particle, potential, params=None, n_grid=1200, n_levels=4):
    """First n_levels energies (eV) for a particle in a potential with parameters
    (box: L; harmonic: k; linear: F, V = F|x|). Grids adapt to the particle's length scale."""
    if particle not in MASS:
        raise ValueError(f"particle must be one of {list(MASS)}")
    if potential not in POTENTIALS:
        raise ValueError(f"potential must be one of {POTENTIALS}")
    p = dict(NOMINAL, **(params or {}))
    mr = MASS[particle]
    if potential == "box":
        x = np.linspace(0, p["box_L"], n_grid + 2)[1:-1]
        V = np.zeros_like(x)
    elif potential == "harmonic":
        s = (2 * HBAR2_2ME / mr / p["harmonic_k"]) ** 0.25          # ground-state length
        x = np.linspace(-10 * s, 10 * s, n_grid)
        V = 0.5 * p["harmonic_k"] * x ** 2
    else:
        ell = (HBAR2_2ME / mr / p["linear_F"]) ** (1 / 3)            # Airy length
        x = np.linspace(-12 * ell, 12 * ell, n_grid)
        V = p["linear_F"] * np.abs(x)
    return lowest_levels(V, x[1] - x[0], mr, n_levels)


def features(E):
    """[log10 E1, E2/E1, E3/E1, E4/E1] from the first four levels."""
    E = np.asarray(E, float)
    return np.array([np.log10(E[0]), E[1] / E[0], E[2] / E[0], E[3] / E[0]])


FEATURE_NAMES = ["log10 E1 (eV)", "E2/E1", "E3/E1", "E4/E1"]


def make_dataset(n_per_class=60, jitter=0.10, meas_noise=0.01, particles=tuple(MASS), seed=0):
    """Feature matrix X, labels y ('particle|potential') for jittered devices and noisy level readout."""
    if not 0 <= jitter < 1 or meas_noise < 0:
        raise ValueError("need 0 <= jitter < 1 and meas_noise >= 0")
    rng = np.random.default_rng(seed)
    X, y = [], []
    for part in particles:
        for pot in POTENTIALS:
            for _ in range(n_per_class):
                j = lambda v: v * (1 + jitter * rng.uniform(-1, 1))
                params = dict(box_L=j(NOMINAL["box_L"]), harmonic_k=j(NOMINAL["harmonic_k"]),
                              linear_F=j(NOMINAL["linear_F"]))
                E = spectrum(part, pot, params) * (1 + meas_noise * rng.standard_normal(4))
                X.append(features(E))
                y.append(f"{part}|{pot}")
    return np.array(X), np.array(y)


def train_classifier(X, y, seed=0, test_size=0.3):
    """Standardize + multinomial logistic regression; returns (model, X_test, y_test, accuracy)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=10.0))
    model.fit(Xtr, ytr)
    return model, Xte, yte, float(model.score(Xte, yte))


def physics_rule(Xrow):
    """Hand-written baseline: potential from E2/E1 (linear 2.30, harmonic 3, box 4), then
    particle from log E1 against the nominal-geometry prediction."""
    logE1, r21 = Xrow[0], Xrow[1]
    pot = "linear" if r21 < 2.65 else ("harmonic" if r21 < 3.5 else "box")
    best, part = np.inf, None
    for p in MASS:
        ref = np.log10(spectrum(p, pot)[0])
        if abs(ref - logE1) < best:
            best, part = abs(ref - logE1), p
    return f"{part}|{pot}"
