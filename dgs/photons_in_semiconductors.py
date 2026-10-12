"""Photons in semiconductors (Saleh & Teich, Fundamentals of Photonics, the
"Photons in Semiconductors" chapter) -- the parts this repo did not have yet.

1. CARRIERS = IMPROPER INTEGRALS
   n = int_{Ec}^{inf} rho_c(E) f(E) dE = N_c F_{1/2}(eta),  eta = (E_F - E_c)/kT
   F_{1/2}(eta) = (2/sqrt(pi)) int_0^inf sqrt(x) / (1 + e^{x - eta}) dx
   Boltzmann limit (eta << 0): the integral is Gamma(3/2) e^{eta} = (sqrt(pi)/2) e^eta,
   so n = N_c e^{eta}. SymPy does that integral exactly; quad does the full one.
   Mass-action law n p = n_i^2 and charge neutrality n + N_a = p + N_d fix E_F.

2. OPTICAL TRANSITIONS (direct gap, parabolic bands)
   joint DOS  rho(nu) ~ sqrt(h nu - E_g)
   LED spectrum  S ~ sqrt(h nu - E_g) exp(-(h nu - E_g)/kT): peak E_g + kT/2, FWHM ~ 1.8 kT
   gain  ~ sqrt(h nu - E_g) [f_c(E2) - f_v(E1)]  > 0  iff  E_g < h nu < E_Fc - E_Fv
   (Bernard-Duraffourg condition -- the reason a laser diode needs inversion).

3. DETECTION: photons are Poisson, so photocurrent shot noise is 2 q I B.
   Two lasers on one photodiode beat:  i = R[P1 + P2 + 2 sqrt(P1 P2) cos(2 pi df t)].
   Many channels against one local oscillator -> one beat tone per channel
   (coherent WDM demultiplexing by FFT), simulated in parallel with torch
   (CUDA if available) and checked against the closed-form tone amplitudes.
Reuses dgs.quantum_statistics.fermi_dirac and dgs.transimpedance_amplifier.responsivity.
"""
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.optimize import brentq

from dgs.quantum_statistics import fermi_dirac
from dgs.transimpedance_amplifier import responsivity

K_B_EV = 8.617333262e-5          # eV/K
H = 6.62607015e-34
M_E = 9.1093837015e-31
Q = 1.602176634e-19
K_B = 1.380649e-23

# Density-of-states effective masses (relative to m_e) and gaps at 300 K
# (Sze & Ng, Physics of Semiconductor Devices). GaAs hole mass is the
# heavy+light combined DOS value, rounded.
MATERIALS = {
    "Si":   dict(Eg=1.12, me=1.08, mh=0.81, direct=False),
    "GaAs": dict(Eg=1.42, me=0.067, mh=0.48, direct=True),
    "InP":  dict(Eg=1.344, me=0.077, mh=0.60, direct=True),
}


def _mat(material):
    if isinstance(material, dict):
        return material
    if material not in MATERIALS:
        raise ValueError(f"material must be one of {list(MATERIALS)} or a dict")
    return MATERIALS[material]


# ── 1. carriers as improper integrals ────────────────────────────────

def effective_density_cm3(m_rel, T=300.0):
    """N = 2 (2 pi m kT / h^2)^{3/2}, in cm^-3."""
    if m_rel <= 0 or T <= 0:
        raise ValueError("m_rel and T must be positive")
    return 2 * (2 * np.pi * m_rel * M_E * K_B * T / H ** 2) ** 1.5 * 1e-6


def boltzmann_limit_symbolic():
    """SymPy: (2/sqrt(pi)) int_0^oo sqrt(x) e^{-(x - eta)} dx = e^eta (exact)."""
    xs, eta = sp.symbols("x eta", real=True)
    xs = sp.Symbol("x", positive=True)
    val = 2 / sp.sqrt(sp.pi) * sp.integrate(sp.sqrt(xs) * sp.exp(-(xs - eta)), (xs, 0, sp.oo))
    return sp.simplify(val)


def fermi_dirac_half(eta):
    """F_{1/2}(eta) = (2/sqrt(pi)) int_0^inf sqrt(x)/(1+e^{x-eta}) dx  (improper)."""
    def one(e):
        f = lambda x: np.sqrt(x) * np.exp(-np.logaddexp(0.0, x - e))   # overflow-safe
        upper = max(e, 0.0) + 60.0
        val, _ = integrate.quad(f, 0, upper, limit=200, epsabs=0, epsrel=1e-11)
        tail, _ = integrate.quad(f, upper, np.inf, epsabs=0, epsrel=1e-11)
        return 2 / np.sqrt(np.pi) * (val + tail)
    return np.vectorize(one)(np.asarray(eta, dtype=float))


def fermi_dirac_half_mapped(eta, n=4000):
    """Same improper integral via x = u/(1-u), mapping [0, inf) to [0, 1).

    dx = du/(1-u)^2. A plain midpoint rule then works on a finite interval --
    the standard trick for turning an improper integral into a proper one.
    """
    u = (np.arange(n) + 0.5) / n
    x = u / (1 - u)
    w = 1 / (1 - u) ** 2 / n
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    vals = [np.sum(np.sqrt(x) * np.exp(-np.logaddexp(0.0, x - e)) * w) for e in eta]
    return 2 / np.sqrt(np.pi) * np.array(vals)


def electron_density(eta, material="GaAs", T=300.0, boltzmann=False):
    """n = N_c F_{1/2}(eta) (or N_c e^eta in the Boltzmann approximation), cm^-3."""
    m = _mat(material)
    Nc = effective_density_cm3(m["me"], T)
    return Nc * (np.exp(eta) if boltzmann else fermi_dirac_half(eta))


def intrinsic_density(material="GaAs", T=300.0):
    """n_i = sqrt(N_c N_v) exp(-E_g / 2kT), cm^-3."""
    m = _mat(material)
    kT = K_B_EV * T
    return np.sqrt(effective_density_cm3(m["me"], T) * effective_density_cm3(m["mh"], T)) \
        * np.exp(-m["Eg"] / (2 * kT))


def fermi_level(Nd=0.0, Na=0.0, material="GaAs", T=300.0):
    """Solve n + N_a = p + N_d (full ionisation, full Fermi-Dirac integrals).

    Energies in eV measured from the valence-band edge (E_v = 0, E_c = E_g).
    Returns dict E_F, n, p.
    """
    if Nd < 0 or Na < 0:
        raise ValueError("doping densities must be non-negative")
    m = _mat(material)
    kT = K_B_EV * T
    Nc, Nv = effective_density_cm3(m["me"], T), effective_density_cm3(m["mh"], T)
    n = lambda EF: Nc * fermi_dirac_half((EF - m["Eg"]) / kT)
    p = lambda EF: Nv * fermi_dirac_half((0.0 - EF) / kT)
    g = lambda EF: np.log(n(EF) + Na) - np.log(p(EF) + Nd)     # monotone, well scaled
    EF = brentq(g, -0.5, m["Eg"] + 0.5, xtol=1e-13)
    return {"E_F": EF, "n": float(n(EF)), "p": float(p(EF))}


# ── 2. optical transitions ───────────────────────────────────────────

def led_spectrum(hnu_eV, Eg, T=300.0):
    """Spontaneous emission shape sqrt(h nu - Eg) exp(-(h nu - Eg)/kT), unnormalised."""
    x = np.asarray(hnu_eV, dtype=float) - Eg
    kT = K_B_EV * T
    return np.where(x > 0, np.sqrt(np.clip(x, 0, None)) * np.exp(-x / kT), 0.0)


def led_peak_symbolic():
    """SymPy: d/dx [sqrt(x) e^{-x/kT}] = 0  ->  x = kT/2."""
    x, kT = sp.symbols("x kT", positive=True)
    return sp.solve(sp.diff(sp.sqrt(x) * sp.exp(-x / kT), x), x)[0]


def led_fwhm_over_kT():
    """FWHM of sqrt(y) e^{-y} (y = x/kT): roots of sqrt(y)e^{-y} = (1/2) peak."""
    f = lambda y: np.sqrt(y) * np.exp(-y)
    half = f(0.5) / 2
    lo = brentq(lambda y: f(y) - half, 1e-9, 0.5)
    hi = brentq(lambda y: f(y) - half, 0.5, 20)
    return hi - lo


def gain_spectrum(hnu_eV, EFc, EFv, material="GaAs", T=300.0):
    """Gain shape sqrt(h nu - Eg) [f_c(E2) - f_v(E1)] (E_v = 0, E_c = Eg).

    Direct transition at fixed k:  E2 - E_c = (m_r/m_e)(h nu - Eg),
    E1 = E2 - h nu,  1/m_r = 1/m_e + 1/m_h.  Positive = gain, negative = absorption.
    """
    m = _mat(material)
    hnu = np.asarray(hnu_eV, dtype=float)
    mr = 1 / (1 / m["me"] + 1 / m["mh"])
    x = hnu - m["Eg"]
    E2 = m["Eg"] + (mr / m["me"]) * np.clip(x, 0, None)
    E1 = E2 - hnu
    kT = K_B_EV * T
    occ = fermi_dirac(E2, EFc, kT) - fermi_dirac(E1, EFv, kT)
    return np.where(x > 0, np.sqrt(np.clip(x, 0, None)) * occ, 0.0)


def bernard_duraffourg_symbolic():
    """f_c(E2) - f_v(E1) > 0  <=>  E2 - E1 < E_Fc - E_Fv. Returns the simplified
    numerator sign expression (should be exp(...) - exp(...))."""
    E1, E2, Fc, Fv, kT = sp.symbols("E1 E2 F_c F_v kT", real=True)
    fc = 1 / (1 + sp.exp((E2 - Fc) / kT))
    fv = 1 / (1 + sp.exp((E1 - Fv) / kT))
    num = sp.factor(sp.together(fc - fv))
    return sp.simplify(sp.numer(num))


# ── 3. detection: Poisson photons, beat notes, coherent WDM ──────────

def photon_counts(P_watt, wavelength_nm, dt, eta=1.0, n=10000, seed=0):
    """Detected photoelectron counts per interval dt: Poisson(eta P dt / h nu)."""
    if P_watt < 0 or dt <= 0 or not 0 < eta <= 1:
        raise ValueError("P >= 0, dt > 0, 0 < eta <= 1")
    mean = eta * P_watt * dt * wavelength_nm * 1e-9 / (H * 299792458.0)
    return np.random.default_rng(seed).poisson(mean, size=n), mean


def beat_photocurrent(P1, P2, df, t, wavelength_nm=1550.0, eta=0.8):
    """i(t) = R [P1 + P2 + 2 sqrt(P1 P2) cos(2 pi df t)] for two co-polarised lasers."""
    R = responsivity(wavelength_nm, eta)
    return R * (P1 + P2 + 2 * np.sqrt(P1 * P2) * np.cos(2 * np.pi * df * np.asarray(t)))


def coherent_wdm_batch(P_ch, f_if, P_lo=1e-3, fs=20e9, n=4096, trials=64,
                       wavelength_nm=1550.0, eta=0.8, B_noise=None, R_load=None, T=300.0,
                       seed=0, device=None):
    """Batched coherent WDM receiver in torch.

    Each channel k (power P_ch[k]) beats with one local oscillator at
    intermediate frequency f_if[k]. The photocurrent of every trial is
        i = R [P_lo + sum P_k + sum 2 sqrt(P_lo P_k) cos(2 pi f_k t + phi_k)] + shot noise
    (channel-channel beats ignored: P_k << P_lo). Random phases per trial.
    Noise: shot 2 q I_dc B, plus thermal 4 k T B / R_load if R_load is given.
    With thermal noise, raising P_lo lifts the tones (~sqrt(P_lo)) above the
    fixed thermal floor until shot noise (~sqrt(P_lo) too) dominates -- the
    shot-noise-limited coherent receiver.
    Recovered channel power = (|FFT at f_k| * 2/n / (2 R))^2 / P_lo.

    Returns dict: recovered (trials x channels, numpy), expected tone amplitude,
    device used.
    """
    import torch
    P_ch = np.asarray(P_ch, dtype=float)
    f_if = np.asarray(f_if, dtype=float)
    if P_ch.shape != f_if.shape or np.any(P_ch < 0) or P_lo <= 0:
        raise ValueError("P_ch, f_if same shape; powers non-negative; P_lo > 0")
    if np.any(f_if <= 0) or np.any(f_if >= fs / 2):
        raise ValueError("intermediate frequencies must be in (0, fs/2)")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = responsivity(wavelength_nm, eta)
    # snap IFs to FFT bins so tones are leakage-free
    bins = np.round(f_if / fs * n).astype(int)
    f_snap = bins * fs / n
    t = torch.arange(n, dtype=torch.float64, device=dev) / fs
    phi = (2 * np.pi * torch.rand(trials, len(P_ch), 1, generator=g, dtype=torch.float64)).to(dev)
    amp = torch.tensor(2 * R * np.sqrt(P_lo * P_ch), dtype=torch.float64, device=dev)[None, :, None]
    fk = torch.tensor(f_snap, dtype=torch.float64, device=dev)[None, :, None]
    i = R * (P_lo + P_ch.sum()) + (amp * torch.cos(2 * np.pi * fk * t + phi)).sum(dim=1)
    if B_noise is None:
        B_noise = fs / 2
    I_dc = R * (P_lo + P_ch.sum())
    var = 2 * Q * I_dc * B_noise
    if R_load is not None:
        if R_load <= 0:
            raise ValueError("R_load must be positive")
        var += 4 * K_B * T * B_noise / R_load
    sigma = np.sqrt(var)
    i = i + sigma * torch.randn(trials, n, generator=g, dtype=torch.float64).to(dev)
    spec = torch.fft.rfft(i, dim=1).abs() * 2 / n
    tone = spec[:, bins]
    recovered = (tone / (2 * R)) ** 2 / P_lo
    return {"recovered": recovered.cpu().numpy(), "tone_expected": 2 * R * np.sqrt(P_lo * P_ch),
            "tone_measured": tone.cpu().numpy(), "device": str(dev), "f_snapped": f_snap,
            "shot_sigma": sigma}
