"""Atomic clocks: Ramsey fringes, quantum-limited stability, relativity, IEEE 754
floating point, loops vs vectorized Allan deviation, and a random forest that
identifies clock noise.

  * RAMSEY. pi/2 pulse -> free evolution for T at detuning Delta -> pi/2 pulse:
    P = (1 + cos(Delta T)) / 2 (hard pulses). Fringe FWHM = 1 / (2T) in Hz.
  * QUANTUM PROJECTION NOISE. Each cycle measures N atoms: phase noise 1/sqrt(N),
    so sigma_y(tau) = 1 / (2 pi nu0 T sqrt(N)) * sqrt(T_c / tau).
  * HOLDOVER. A fractional frequency offset y makes the time error grow as
    x(t) = y t: how much time a clock "loses" once GPS is gone.
  * RELATIVITY. A GPS satellite clock runs fast by GM/c^2 (1/R - 1/r) from
    gravity and slow by v^2/(2c^2) from speed: net ~ +38 microseconds/day.
  * IEEE 754. A double carries ~16 significant digits; optical clocks resolve
    1e-18. Accumulating time in floating point loses it ("lost in the past"):
    the 1991 Patriot clock bug summed 0.1 s in 24-bit fixed point.
  * ALLAN DEVIATION. The clock-stability statistic, written as an explicit
    loop and as vectorized NumPy (cumulative sums), and timed.
  * ML. Power-law noise types leave fingerprints in ADEV/MDEV slopes; a
    decision tree and a random forest learn to classify them -- and show why
    ADEV alone cannot separate white from flicker phase noise.

Clock grades and cost tiers are ORDER-OF-MAGNITUDE classes, not product specs.
Synthetic data. Education.
"""
import numpy as np
import sympy as sp

NU_CS = 9_192_631_770            # Hz, defines the SI second (exact)
NU_SR = 429_228_004_229_873.0    # Hz, Sr-87 optical clock transition (approximate value used for illustration)
C = 299792458.0
GM_EARTH = 3.986004418e14
R_EARTH = 6.371e6
DAY = 86400.0

CLOCK_GRADES = {   # (fractional frequency offset after calibration, rough cost tier)
    "quartz TCXO": (1e-7, "dollars to tens of dollars"),
    "oven quartz (OCXO)": (1e-9, "hundreds of dollars"),
    "chip-scale atomic (CSAC)": (1e-11, "thousands of dollars"),
    "rubidium": (1e-11, "thousands of dollars"),
    "cesium beam": (1e-13, "tens of thousands of dollars"),
    "hydrogen maser": (1e-15, "hundreds of thousands of dollars"),
    "optical lattice (lab)": (1e-18, "research lab, millions in infrastructure"),
}


# ── 1. Ramsey interrogation ─────────────────────────────────────────

def ramsey_symbolic():
    """Hard pi/2 pulses (rotation about x) around free evolution exp(-i Delta T sigma_z / 2).
    Returns P(excited) after starting in |g>."""
    D, T = sp.symbols("Delta T", real=True)
    c, s = sp.cos(sp.pi / 4), sp.sin(sp.pi / 4)
    Rx = sp.Matrix([[c, -sp.I * s], [-sp.I * s, c]])           # exp(-i (pi/2) sigma_x / 2)
    U0 = sp.diag(sp.exp(-sp.I * D * T / 2), sp.exp(sp.I * D * T / 2))
    psi = Rx * U0 * Rx * sp.Matrix([1, 0])
    P = sp.simplify(sp.expand_complex(sp.Abs(psi[1]) ** 2))
    return sp.simplify(sp.trigsimp(P.rewrite(sp.cos)))


def ramsey_fwhm_symbolic():
    """Detuning (in Hz, Delta = 2 pi delta_f) where P falls to 1/2: FWHM = 1/(2T)."""
    df, T = sp.symbols("delta_f T", positive=True)
    P = (1 + sp.cos(2 * sp.pi * df * T)) / 2
    half = sp.solve(sp.Eq(P, sp.Rational(1, 2)), df)
    return 2 * min(half, key=lambda e: e.subs(T, 1))


def ramsey_fringe(detuning_hz, T, pulse_time=None):
    """P(excited) vs detuning. With finite pulses (Rabi frequency set so that
    each pulse is pi/2 on resonance) the fringes sit under a sinc-like envelope."""
    d = 2 * np.pi * np.asarray(detuning_hz, float)
    if pulse_time is None:
        return 0.5 * (1 + np.cos(d * T))
    Om = np.pi / (2 * pulse_time)
    W = np.sqrt(Om**2 + d**2)
    out = []
    for dd, ww in zip(np.atleast_1d(d), np.atleast_1d(W)):
        H = 0.5 * np.array([[-dd, Om], [Om, dd]])
        ev, V = np.linalg.eigh(H)
        Up = V @ np.diag(np.exp(-1j * ev * pulse_time)) @ V.conj().T
        Uf = np.diag([np.exp(1j * dd * T / 2), np.exp(-1j * dd * T / 2)])
        psi = Up @ Uf @ Up @ np.array([1, 0])
        out.append(abs(psi[1]) ** 2)
    return np.array(out)


# ── 2. quantum projection noise and Allan deviation limit ───────────

def qpn_allan_symbolic():
    """Per cycle: phase uncertainty 1/sqrt(N) -> frequency uncertainty
    1/(2 pi T sqrt(N)); fractional: divide by nu0; average tau/T_c cycles."""
    nu0, T, N, Tc, tau = sp.symbols("nu_0 T N T_c tau", positive=True)
    per_cycle = 1 / (2 * sp.pi * T * sp.sqrt(N)) / nu0
    return sp.simplify(per_cycle / sp.sqrt(tau / Tc))


def qpn_allan(tau, nu0, T, N, Tc):
    return 1 / (2 * np.pi * nu0 * T * np.sqrt(N)) * np.sqrt(Tc / np.asarray(tau, float))


# ── 3. holdover: time lost without GPS ───────────────────────────────

def holdover_error(t_s, frac_offset, aging_per_day=0.0):
    """x(t) = y t + (1/2) D t^2 with D the linear frequency drift (aging)."""
    t = np.asarray(t_s, float)
    return frac_offset * t + 0.5 * aging_per_day / DAY * t**2


def holdover_table(t_s=DAY):
    return {k: (float(holdover_error(t_s, y)), cost) for k, (y, cost) in CLOCK_GRADES.items()}


# ── 4. relativity for GPS ────────────────────────────────────────────

def gps_relativity_symbolic():
    """Fractional rate of a satellite clock (radius r, circular speed sqrt(GM/r))
    relative to the ground (radius R, Earth rotation neglected):
        gravity: + GM/c^2 (1/R - 1/r);  velocity: - GM/(2 r c^2)."""
    G, c, R, r = sp.symbols("GM c R r", positive=True)
    grav = G / c**2 * (1 / R - 1 / r)
    vel = -(G / r) / (2 * c**2)
    return sp.simplify(grav), sp.simplify(vel), sp.simplify(grav + vel)


def gps_relativity_us_per_day(r=26_561_750.0):
    grav, vel, tot = gps_relativity_symbolic()
    G, c, R, rr = sp.symbols("GM c R r", positive=True)
    sub = {G: GM_EARTH, c: C, R: R_EARTH, rr: r}
    return tuple(float(e.subs(sub)) * DAY * 1e6 for e in (grav, vel, tot))


def height_shift(dh_m, g=9.80665):
    """Gravitational redshift between clocks dh apart: g dh / c^2."""
    return g * dh_m / C**2


# ── 5. IEEE 754 and time accumulation ────────────────────────────────

def float_resolution(value, dtype=np.float64):
    """Gap to the next representable number at `value` (ULP)."""
    return float(np.spacing(dtype(value)))


def accumulate_time(dt=0.1, n=3_600_000, dtype=np.float64):
    """Sum dt n times in the given float type (sequential, like a clock tick)."""
    acc = dtype(0)
    step = dtype(dt)
    for _ in range(n):
        acc = dtype(acc + step)
    return float(acc)


def accumulate_time_fast(dt=0.1, n=3_600_000, dtype=np.float64):
    """Same sequential rounding, vectorized: np.cumsum accumulates left to right in dtype."""
    return float(np.cumsum(np.full(n, dt, dtype=dtype), dtype=dtype)[-1])


def patriot_chop_error(hours=100.0, bits_after_point=23):
    """0.1 s stored by truncation in a fixed-point register with 23 fractional
    bits (24-bit register): the per-tick error times the tick count.
    Returns (per-tick error s, accumulated error s)."""
    tenth = sp.Rational(1, 10)
    chopped = sp.floor(tenth * 2**bits_after_point) / sp.Integer(2) ** bits_after_point
    per_tick = tenth - chopped
    ticks = int(hours * 3600 * 10)
    return float(per_tick), float(per_tick * ticks)


def exact_sum_symbolic(dt="1/10", n=3_600_000):
    """Exact rational arithmetic in SymPy: no rounding at all."""
    return sp.Rational(dt) * n


# ── 6. power-law clock noise and Allan deviation: loops vs vectorized ─

NOISE_BETA = {"white PM": 2, "flicker PM": 1, "white FM": 0, "flicker FM": -1, "random walk FM": -2}


def powerlaw_frequency_noise(n, tau0, beta, rng=None):
    """Fractional-frequency samples y with PSD S_y(f) ~ f^beta (FFT shaping)."""
    rng = np.random.default_rng(rng)
    f = np.fft.rfftfreq(n, tau0)
    f[0] = f[1]
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) * f ** (beta / 2)
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return y / np.std(y)


def phase_from_frequency(y, tau0):
    """Time-error series x (seconds): x_k = tau0 * sum_{j<k} y_j."""
    return np.concatenate([[0.0], np.cumsum(y) * tau0])


def adev_loop(x, tau0, m):
    """Overlapping Allan deviation at tau = m tau0, explicit Python loop:
    sigma^2 = 1/(2 tau^2 (N-2m)) sum (x_{i+2m} - 2 x_{i+m} + x_i)^2."""
    N = len(x)
    tau = m * tau0
    s = 0.0
    for i in range(N - 2 * m):
        d = x[i + 2 * m] - 2 * x[i + m] + x[i]
        s += d * d
    return np.sqrt(s / (2 * tau**2 * (N - 2 * m)))


def adev(x, tau0, m):
    """Same statistic, vectorized (slicing)."""
    x = np.asarray(x, float)
    d = x[2 * m:] - 2 * x[m:-m] + x[:-2 * m]
    return np.sqrt(np.mean(d * d) / (2 * (m * tau0) ** 2))


def mdev(x, tau0, m):
    """Modified Allan deviation (vectorized with cumulative sums): averages the
    phase over m samples before differencing, which separates white PM
    (slope -3/2) from flicker PM (slope -1)."""
    x = np.asarray(x, float)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    N = len(x)
    n_terms = N - 3 * m + 1
    if n_terms < 1:
        raise ValueError("series too short for this m")
    j = np.arange(n_terms)
    s1 = cs[j + m] - cs[j]
    s2 = cs[j + 2 * m] - cs[j + m]
    s3 = cs[j + 3 * m] - cs[j + 2 * m]
    d = (s3 - 2 * s2 + s1) / m
    return np.sqrt(np.mean(d * d) / (2 * (m * tau0) ** 2))


def adev_symbolic_second_difference():
    """The Allan variance is half the mean square of the second difference of
    phase over tau, divided by tau^2. For a pure frequency offset y (x = y t)
    the second difference vanishes -- ADEV ignores constant offsets."""
    t, tau, y, x0 = sp.symbols("t tau y x_0", real=True)
    x = x0 + y * t
    return sp.simplify(x.subs(t, t + 2 * tau) - 2 * x.subs(t, t + tau) + x)


def slope_features(x, tau0, ms=(1, 2, 4, 8, 16, 32, 64)):
    """log-log slopes of ADEV and MDEV between successive averaging factors."""
    ms = np.asarray(ms)
    a = np.array([adev(x, tau0, m) for m in ms])
    md = np.array([mdev(x, tau0, m) for m in ms])
    la, lm, lt = np.log(a), np.log(md), np.log(ms * tau0)
    return np.concatenate([np.diff(la) / np.diff(lt), np.diff(lm) / np.diff(lt)])


def make_noise_dataset(n_per_class=60, n=4096, tau0=1.0, rng=0):
    """Feature matrix (ADEV slopes, MDEV slopes) and labels for the 5 noise types."""
    rng = np.random.default_rng(rng)
    X, yl = [], []
    for label, beta in NOISE_BETA.items():
        for _ in range(n_per_class):
            yf = powerlaw_frequency_noise(n, tau0, beta, rng)
            X.append(slope_features(phase_from_frequency(yf, tau0), tau0))
            yl.append(label)
    return np.array(X), np.array(yl)


def train_classifiers(X, y, adev_only=False, seed=0):
    """Train/test a decision tree and a random forest. adev_only=True drops the
    MDEV slope features. Returns dict of test accuracies and the forest's
    confusion matrix."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    nfeat = X.shape[1] // 2
    Xu = X[:, :nfeat] if adev_only else X
    Xtr, Xte, ytr, yte = train_test_split(Xu, y, test_size=0.3, random_state=seed, stratify=y)
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(Xtr, ytr)
    forest = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr, ytr)
    labels = list(NOISE_BETA)
    return {"tree_acc": float(tree.score(Xte, yte)), "forest_acc": float(forest.score(Xte, yte)),
            "confusion": confusion_matrix(yte, forest.predict(Xte), labels=labels),
            "labels": labels, "tree": tree, "forest": forest}
