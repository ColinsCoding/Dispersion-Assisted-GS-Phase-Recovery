"""A $650 light-scattering flow cell: gravity flow, pulse timing, Mie pulse heights, budget.

In plain words: a laser beam crosses a thin glass capillary with water and beads flowing
through it. Each bead that passes makes a short flash of scattered light; a photodiode +
transimpedance amplifier (TIA) turns the flash into a voltage pulse, and a Raspberry Pi
Pico's ADC records it. Bigger beads scatter more -- but not smoothly: Mie resonances make
the pulse height wiggle with size, which is why real instruments calibrate with known
beads. This is a minimal flow cytometer, the slow cousin of Jalali's STEAM imaging flow
cytometer.

The chain, each step with a closed form:
  1. Gravity flow (no pump): dP = rho g h, Hagen-Poiseuille R = 8 mu L / (pi r^4),
     Q = dP / R. A reservoir of area A drains with time constant tau = R A / (rho g).
  2. Pulse timing: Gaussian beam I ~ exp(-2 y^2 / w^2) crossed at speed v gives a pulse
     of FWHM w sqrt(2 ln 2) / v. Parabolic (Poiseuille) flow v = 2 vbar (1 - r^2/R^2)
     spreads the pulse widths; Poisson statistics set the bead dilution.
  3. Mie scattering (Bohren & Huffman algorithm): efficiencies and angular amplitudes
     S1, S2 for a sphere of relative index m = n_p / n_med and size x = pi d n_med / lambda.
     Collected power = peak irradiance x the cross-section integrated over the detector's
     angular window.
  4. Detector: V = P_collected x responsivity x R_f; 12-bit ADC counts; TIA bandwidth
     ~ sqrt(GBW / (2 pi R_f C_d)).
  5. Budget: a parts list that fits $650 with tax and a reserve. Prices are rough
     estimates written from memory -- check vendors before ordering.

SI units throughout unless a name says otherwise. NumPy. Education / bench planning.
"""

import numpy as np

from dgs.microfluidics import reynolds_number

RHO_WATER = 998.0       # kg/m^3
MU_WATER = 1.0e-3       # Pa s (20 C)
G = 9.81                # m/s^2
N_WATER = 1.333
N_POLYSTYRENE = 1.59    # near 532-650 nm


def _pos(name, v):
    if not np.all(np.isfinite(v)) or np.any(np.asarray(v) <= 0):
        raise ValueError(f"{name} must be positive, got {v}")


# ---------------------------------------------------------------- 1. gravity flow

def tube_resistance(L, r, mu=MU_WATER):
    """Hagen-Poiseuille hydraulic resistance of a round tube: 8 mu L / (pi r^4) [Pa s / m^3]."""
    _pos("L", L); _pos("r", r); _pos("mu", mu)
    return 8 * mu * L / (np.pi * r ** 4)


def gravity_flow(h, R_total, rho=RHO_WATER):
    """Volumetric flow Q = rho g h / R_total [m^3/s] (resistances in series add)."""
    _pos("R_total", R_total)
    if np.any(np.asarray(h) < 0):
        raise ValueError("head height h must be >= 0")
    return rho * G * np.asarray(h, float) / R_total


def drain_time_constant(R_total, A_reservoir, rho=RHO_WATER):
    """tau = R A / (rho g): the head decays as h(t) = h0 exp(-t / tau)."""
    _pos("R_total", R_total); _pos("A_reservoir", A_reservoir)
    return R_total * A_reservoir / (rho * G)


def head_vs_time(h0, t, tau):
    """h(t) = h0 exp(-t/tau) for a reservoir draining through a fixed resistance."""
    _pos("tau", tau)
    return h0 * np.exp(-np.asarray(t, float) / tau)


def mean_velocity(Q, r):
    """vbar = Q / (pi r^2)."""
    _pos("r", r)
    return np.asarray(Q, float) / (np.pi * r ** 2)


def channel_reynolds(Q, r, rho=RHO_WATER, mu=MU_WATER):
    """Re based on mean velocity and diameter (uses dgs.microfluidics.reynolds_number)."""
    return float(reynolds_number(rho, mean_velocity(Q, r), 2 * r, mu)["Re"])


def poiseuille_velocity(rr, r, vbar):
    """Parabolic profile v(rr) = 2 vbar (1 - rr^2 / r^2), zero at the wall."""
    _pos("r", r)
    rr = np.asarray(rr, float)
    if np.any(np.abs(rr) > r):
        raise ValueError("radial position must be inside the tube")
    return 2 * vbar * (1 - rr ** 2 / r ** 2)


# ---------------------------------------------------------------- 2. pulse timing

def pulse_shape(t, v, w, peak=1.0):
    """Signal of a point bead crossing a Gaussian beam (1/e^2 radius w) at speed v."""
    _pos("v", v); _pos("w", w)
    return peak * np.exp(-2 * (v * np.asarray(t, float)) ** 2 / w ** 2)


def pulse_fwhm(w, v):
    """FWHM in time: w sqrt(2 ln 2) / v."""
    _pos("v", v); _pos("w", w)
    return w * np.sqrt(2 * np.log(2)) / v


def transit_width_samples(n, r, vbar, w, seed=0):
    """Pulse FWHMs for n beads at uniformly random positions over the tube cross-section
    (no hydrodynamic focusing). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    rr = r * np.sqrt(rng.uniform(0, 1, n)) * 0.98          # uniform in area, off the wall
    return pulse_fwhm(w, poiseuille_velocity(rr, r, vbar))


def coincidence_probability(conc_per_mL, Q, window):
    """P(two or more beads in the beam | at least one), Poisson with lambda = c Q window."""
    _pos("window", window)
    lam = conc_per_mL * 1e6 * np.asarray(Q, float) * window
    lam = np.maximum(lam, 1e-300)
    return (1 - np.exp(-lam) - lam * np.exp(-lam)) / (-np.expm1(-lam))


def max_concentration(Q, window, target=0.01):
    """Largest concentration (beads/mL) keeping the coincidence probability below target."""
    lo, hi = 1e-3, 1e12
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if coincidence_probability(mid, Q, window) > target:
            hi = mid
        else:
            lo = mid
    return lo


# ---------------------------------------------------------------- 3. Mie scattering (Bohren & Huffman)

def size_parameter(d, wavelength, n_med=N_WATER):
    """x = pi d n_med / lambda0."""
    _pos("d", d); _pos("wavelength", wavelength)
    return np.pi * d * n_med / wavelength


def mie_coefficients(m, x):
    """Mie coefficients a_n, b_n (n = 1..nstop) for relative index m and size parameter x."""
    _pos("x", x)
    m = complex(m)
    nstop = int(np.round(x + 4 * x ** (1 / 3) + 2))
    y = m * x
    nmx = int(max(nstop, abs(y)) + 16)
    D = np.zeros(nmx + 1, complex)
    for n in range(nmx, 0, -1):                       # logarithmic derivative, downward
        D[n - 1] = n / y - 1 / (D[n] + n / y)
    a, b = np.zeros(nstop, complex), np.zeros(nstop, complex)
    psi0, psi1 = np.cos(x), np.sin(x)                 # psi_{-1}, psi_0
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = psi1 - 1j * chi1
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) / x * psi1 - psi0
        chi = (2 * n - 1) / x * chi1 - chi0
        xi = psi - 1j * chi
        da, db = D[n] / m + n / x, m * D[n] + n / x
        a[n - 1] = (da * psi - psi1) / (da * xi - xi1)
        b[n - 1] = (db * psi - psi1) / (db * xi - xi1)
        psi0, psi1, chi0, chi1, xi1 = psi1, psi, chi1, chi, xi
    return a, b


def mie_efficiencies(m, x):
    """(Q_ext, Q_sca, Q_back)."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, a.size + 1)
    qext = 2 / x ** 2 * np.sum((2 * n + 1) * (a + b).real)
    qsca = 2 / x ** 2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    qback = np.abs(np.sum((2 * n + 1) * (-1) ** n * (a - b))) ** 2 / x ** 2
    return float(qext), float(qsca), float(qback)


def mie_amplitudes(m, x, theta):
    """Scattering amplitudes S1(theta), S2(theta)."""
    a, b = mie_coefficients(m, x)
    mu = np.cos(np.atleast_1d(np.asarray(theta, float)))
    S1 = np.zeros(mu.size, complex)
    S2 = np.zeros(mu.size, complex)
    pi_prev, pi_n = np.zeros_like(mu), np.ones_like(mu)
    for n in range(1, a.size + 1):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        f = (2 * n + 1) / (n * (n + 1))
        S1 += f * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        S2 += f * (a[n - 1] * tau_n + b[n - 1] * pi_n)
        pi_prev, pi_n = pi_n, ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n
    return S1, S2


def collected_cross_section(d, wavelength, theta_min, theta_max, phi_fraction=1.0,
                            n_particle=N_POLYSTYRENE, n_med=N_WATER, n_theta=400):
    """Cross-section (m^2) scattered into theta_min..theta_max (rad) for unpolarized light,
    times the azimuthal fraction collected: (phi_fraction 2 pi / k^2) int (|S1|^2+|S2|^2)/2 sin(theta) dtheta."""
    if not 0 <= theta_min < theta_max <= np.pi:
        raise ValueError("need 0 <= theta_min < theta_max <= pi")
    if not 0 < phi_fraction <= 1:
        raise ValueError("phi_fraction must be in (0, 1]")
    m = n_particle / n_med
    x = size_parameter(d, wavelength, n_med)
    k = 2 * np.pi * n_med / wavelength
    th = np.linspace(theta_min, theta_max, n_theta)
    S1, S2 = mie_amplitudes(m, x, th)
    integrand = 0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2) * np.sin(th)
    return float(phi_fraction * 2 * np.pi / k ** 2 * np.trapezoid(integrand, th))


def rayleigh_qsca(m, x):
    """Small-particle limit Q_sca = (8/3) x^4 |(m^2 - 1)/(m^2 + 2)|^2."""
    return 8 / 3 * x ** 4 * abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2


# ---------------------------------------------------------------- 4. detector

def peak_irradiance(P, w, w_cross=None):
    """Peak irradiance of a Gaussian beam 2 P / (pi w w_cross) [W/m^2]; w_cross = w for a round
    spot, larger for a cylindrical-lens light sheet (thin along flow, wide across the channel)."""
    w_cross = w if w_cross is None else w_cross
    _pos("P", P); _pos("w", w); _pos("w_cross", w_cross)
    return 2 * P / (np.pi * w * w_cross)


def pulse_height_volts(sigma_collected, P_laser, w, responsivity, R_f, w_cross=None):
    """V_peak = I0 sigma_col responsivity R_f (bead through the beam center)."""
    _pos("responsivity", responsivity); _pos("R_f", R_f)
    return peak_irradiance(P_laser, w, w_cross) * np.asarray(sigma_collected, float) * responsivity * R_f


def simulate_events(n, diameters, v_center_height, r, vbar, w_flow, w_cross, rate_hz, seed=0):
    """n beads with sizes drawn from `diameters`, positions uniform over the tube cross-section.

    v_center_height maps diameter -> pulse height (V) for a bead through the beam center.
    The beam propagates along y, so only the x offset matters: height x exp(-2 x^2 / w_cross^2).
    Speed follows the Poiseuille profile, so off-axis beads give longer, weaker pulses.
    Arrivals are Poisson at rate_hz. Returns a dict of arrays."""
    if n < 1:
        raise ValueError("n must be >= 1")
    _pos("rate_hz", rate_hz)
    rng = np.random.default_rng(seed)
    d = rng.choice(np.asarray(diameters, float), n)
    rad = r * np.sqrt(rng.uniform(0, 1, n)) * 0.98
    ang = rng.uniform(0, 2 * np.pi, n)
    xo = rad * np.cos(ang)
    v = poiseuille_velocity(rad, r, vbar)
    h = np.array([v_center_height[float(di)] for di in d]) * np.exp(-2 * xo ** 2 / w_cross ** 2)
    t0 = np.cumsum(rng.exponential(1 / rate_hz, n))
    return dict(d=d, x_offset=xo, v=v, height=h, fwhm=pulse_fwhm(w_flow, v), t0=t0)


def adc_trace(events, fs, w_flow, bits=12, v_ref=3.3, noise_v=0.0, seed=1):
    """Sampled detector voltage: sum of Gaussian pulses + white noise, through an ideal ADC.
    Returns (t, volts_before_adc, adc_codes)."""
    _pos("fs", fs)
    rng = np.random.default_rng(seed)
    T = float(events["t0"][-1] + 5 * events["fwhm"].max())
    t = np.arange(0, T, 1 / fs)
    V = np.zeros_like(t)
    for h, v, t0 in zip(events["height"], events["v"], events["t0"]):
        sl = np.abs(t - t0) < 4 * w_flow / v
        V[sl] += pulse_shape(t[sl] - t0, v, w_flow, h)
    V = V + noise_v * rng.standard_normal(t.size)
    return t, V, adc_counts(np.clip(V, 0, None), bits, v_ref)


def adc_counts(V, bits=12, v_ref=3.3):
    """Ideal ADC code, clipped to full scale."""
    if int(bits) != bits or bits < 1:
        raise ValueError("bits must be a positive integer")
    _pos("v_ref", v_ref)
    return np.clip(np.round(np.asarray(V, float) / v_ref * (2 ** bits - 1)), 0, 2 ** bits - 1).astype(int)


def tia_bandwidth(gbw, R_f, C_d):
    """Approximate TIA -3 dB bandwidth sqrt(GBW / (2 pi R_f C_d)) [Hz] (with optimal feedback C)."""
    _pos("gbw", gbw); _pos("R_f", R_f); _pos("C_d", C_d)
    return float(np.sqrt(gbw / (2 * np.pi * R_f * C_d)))


# ---------------------------------------------------------------- 5. budget (estimates from memory)

PARTS = [
    dict(item="Laser safety glasses, OD4+ at the laser wavelength", category="safety", qty=1, unit_usd=40,
         why="required before the laser is ever powered"),
    dict(item="Laser diode module, <=1 mW class 2 (532 or 650 nm)", category="optics", qty=1, unit_usd=25,
         why="1 mW already gives ~0.1-1 V pulses (Sec. 4)"),
    dict(item="Lens kit: cylindrical (light sheet) + plano-convex (collection)", category="optics", qty=1,
         unit_usd=40, why="25 x 300 um sheet; collect forward/side light"),
    dict(item="ND filter / beam block", category="optics", qty=1, unit_usd=25, why="block the direct beam"),
    dict(item="Borosilicate capillaries, ~0.5 mm ID", category="fluidics", qty=1, unit_usd=40, why="flow cell"),
    dict(item="Tubing, Luer fittings, syringes, reservoir", category="fluidics", qty=1, unit_usd=30,
         why="gravity-driven flow, no pump"),
    dict(item="Polystyrene calibration beads, 2-3 sizes", category="samples", qty=1, unit_usd=150,
         why="known sizes to calibrate pulse height"),
    dict(item="BPW34 photodiodes", category="electronics", qty=5, unit_usd=2, why="detectors (spares)"),
    dict(item="Op-amps, resistors, capacitors, breadboard", category="electronics", qty=1, unit_usd=25,
         why="transimpedance amplifier"),
    dict(item="Raspberry Pi Pico 2", category="electronics", qty=2, unit_usd=7, why="12-bit ADC + USB logging"),
    dict(item="USB microscope", category="alignment", qty=1, unit_usd=40, why="see beam on capillary"),
    dict(item="3D-printed mounts + base plate", category="mechanics", qty=1, unit_usd=40, why="hold everything still"),
]


def budget_summary(parts=PARTS, tax_rate=0.0875, shipping=35.0, cap=650.0):
    """Subtotal, sales tax, shipping, total and remaining reserve under the cap."""
    sub = float(sum(p["qty"] * p["unit_usd"] for p in parts))
    tax = round(sub * tax_rate, 2)
    total = round(sub + tax + shipping, 2)
    return dict(subtotal=sub, tax=tax, shipping=shipping, total=total, reserve=round(cap - total, 2),
                within_cap=total <= cap)
