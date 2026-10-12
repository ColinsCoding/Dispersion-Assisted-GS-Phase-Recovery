"""Statistical thermodynamics for quantum information science, derived in SymPy.

One object -- the partition function Z = sum_i exp(-beta E_i) -- generates
everything:  U = -d ln Z / d beta,  F = -k T ln Z,  S = k (ln Z + beta U),
C = dU/dT,  <dE^2> = k T^2 C.

  * TWO-LEVEL SYSTEM = a thermal qubit. Populations, the Schottky heat-capacity
    peak, and the Bloch vector length tanh(beta eps / 2). Gibbs entropy equals
    the von Neumann entropy of rho = exp(-beta H)/Z and Shannon entropy x k ln 2.
  * HARMONIC OSCILLATOR = a microwave cavity / transmon readout mode.
    Bose-Einstein occupation n = 1/(e^{beta hbar omega} - 1): why superconducting
    qubits live in dilution refrigerators at ~10-20 mK.
  * PARTICLE IN A 3D BOX = the ideal gas. Degeneracy counting (loop vs
    vectorized), the sqrt(E) density of states from an octant of a sphere,
    Z = V / lambda_th^3, and the Sackur-Tetrode entropy, checked against argon.
  * INFORMATION IS PHYSICAL. One bit is k ln 2 of entropy; erasing it costs at
    least k T ln 2 of heat (Landauer). A one-molecule Szilard engine extracts
    exactly k T ln 2 from one bit of knowledge.

Constants are CODATA/SI exact where defined. Education.
"""
import numpy as np
import sympy as sp

K_B = 1.380649e-23
H = 6.62607015e-34
HBAR = H / (2 * np.pi)
N_A = 6.02214076e23
AMU = 1.66053906660e-27

beta, eps, T, k = sp.symbols("beta epsilon T k_B", positive=True)


# ── the partition-function machine ───────────────────────────────────

def thermodynamics_from_Z(Z, b=beta):
    """U, F, S, C from Z(beta) with beta = 1/(k T). Returns dict of SymPy expressions in T."""
    lnZ = sp.log(Z)
    U = sp.simplify(-sp.diff(lnZ, b))
    sub = {b: 1 / (k * T)}
    U_T = sp.simplify(U.subs(sub))
    F_T = sp.simplify((-lnZ / b).subs(sub))
    S_T = sp.simplify((k * (lnZ + b * U)).subs(sub))
    C_T = sp.simplify(sp.diff(U_T, T))
    return {"Z": Z, "U": U_T, "F": F_T, "S": S_T, "C": C_T}


def two_level_symbolic():
    """Energies 0 and eps."""
    Z = 1 + sp.exp(-beta * eps)
    return thermodynamics_from_Z(Z)


def gibbs_equals_shannon_two_level():
    """S_thermo - (-k sum p ln p) for the two-level system: should simplify to 0."""
    th = two_level_symbolic()
    p1 = 1 / (1 + sp.exp(eps / (k * T)))
    p0 = 1 - p1
    S_gibbs = -k * (p0 * sp.log(p0) + p1 * sp.log(p1))
    return sp.simplify(sp.expand_log(th["S"] - S_gibbs, force=True))


def fluctuation_dissipation_two_level():
    """<E^2> - <E>^2 - k T^2 C: should be 0."""
    th = two_level_symbolic()
    p1 = 1 / (1 + sp.exp(eps / (k * T)))
    var = eps**2 * p1 - (eps * p1) ** 2
    return sp.simplify((var - k * T**2 * th["C"]).rewrite(sp.exp))


def schottky_peak():
    """Temperature (in units of eps/k) where the two-level heat capacity peaks:
    solves dC/dT = 0, i.e. x tanh(x/2) = 2 with x = eps/(kT) -> kT ~ 0.417 eps."""
    x = sp.symbols("x", positive=True)
    root = sp.nsolve(x * sp.tanh(x / 2) - 2, x, 2.4)
    return float(1 / root)


def thermal_qubit(freq_hz, T_K):
    """Thermal state of a qubit with splitting h f at temperature T.
    Returns excited population, Bloch length tanh(h f / 2kT), entropy in bits."""
    if freq_hz <= 0 or T_K <= 0:
        raise ValueError("freq_hz and T_K must be positive")
    x = H * freq_hz / (K_B * T_K)
    p1 = 1 / (1 + np.exp(x))
    p0 = 1 - p1
    S_bits = -sum(p * np.log2(p) for p in (p0, p1) if p > 0)
    return {"p_excited": p1, "bloch_length": np.tanh(x / 2), "entropy_bits": S_bits}


def thermal_density_matrix_symbolic():
    """rho = exp(-beta H)/Z for H = (eps/2) sigma_z, its Bloch z-component, and
    von Neumann entropy minus the thermodynamic entropy (in units of k): 0."""
    Hm = sp.diag(eps / 2, -eps / 2)
    rho = sp.simplify((-beta * Hm).exp() / ((-beta * Hm).exp()).trace())
    bz = sp.simplify(rho[0, 0] - rho[1, 1])
    p = [rho[0, 0], rho[1, 1]]
    S_vn = -sum(pi * sp.log(pi) for pi in p)
    Z = sp.exp(beta * eps / 2) + sp.exp(-beta * eps / 2)
    U = -sp.diff(sp.log(Z), beta)
    S_th = sp.log(Z) + beta * U
    # substitute e^{beta eps / 2} = y so every exponential becomes a power of y
    y = sp.symbols("y", positive=True)
    diff = (S_vn - S_th).rewrite(sp.exp).subs(beta, 2 * sp.log(y) / eps)
    diff = sp.simplify(sp.expand_log(sp.simplify(diff), force=True))
    return rho, sp.simplify(bz.rewrite(sp.tanh)), diff


# ── harmonic oscillator: cavities and readout modes ─────────────────

def oscillator_symbolic():
    """Quantum oscillator E_n = hbar w (n + 1/2): Z = e^{-x/2} sum r^n with
    r = e^{-beta hbar w} < 1 -- a convergent geometric series."""
    w, hb = sp.symbols("omega hbar", positive=True)
    n = sp.symbols("n", integer=True, nonnegative=True)
    r = sp.symbols("r", positive=True)
    geo = sp.summation(r**n, (n, 0, sp.oo))
    geo = geo.args[0][0] if isinstance(geo, sp.Piecewise) else geo      # branch |r| < 1
    Z = sp.simplify(sp.exp(-beta * hb * w / 2) * geo.subs(r, sp.exp(-beta * hb * w)))
    U = sp.simplify(-sp.diff(sp.log(Z), beta))
    nbar = sp.simplify(U / (hb * w) - sp.Rational(1, 2))
    return Z, U, nbar


def bose_occupation(freq_hz, T_K):
    x = H * np.asarray(freq_hz, float) / (K_B * np.asarray(T_K, float))
    with np.errstate(over="ignore"):                 # deep freeze-out: e^x overflows, occupation -> 0
        return 1 / np.expm1(x)


# ── particle in a 3D box: degeneracy, density of states, ideal gas ──

def count_states_loop(E_max):
    """Number of (nx, ny, nz >= 1) with nx^2 + ny^2 + nz^2 <= E_max (box units), triple loop."""
    count = 0
    n = int(np.sqrt(E_max)) + 1
    for nx in range(1, n + 1):
        for ny in range(1, n + 1):
            for nz in range(1, n + 1):
                if nx * nx + ny * ny + nz * nz <= E_max:
                    count += 1
    return count


def count_states_vectorized(E_max):
    n = int(np.sqrt(E_max)) + 1
    r = np.arange(1, n + 1)
    s = r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2
    return int(np.count_nonzero(s <= E_max))


def degeneracies(E_max):
    """Map energy (nx^2+ny^2+nz^2) -> number of states, vectorized."""
    n = int(np.sqrt(E_max)) + 1
    r = np.arange(1, n + 1)
    s = (r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2).ravel()
    s = s[s <= E_max]
    vals, counts = np.unique(s, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def octant_count_symbolic():
    """States with n^2 <= E: one octant of a sphere of radius sqrt(E):
    N(E) ~ (1/8)(4/3) pi E^{3/2};  density of states dN/dE = (pi/4) sqrt(E)."""
    E = sp.symbols("E", positive=True)
    N = sp.Rational(1, 8) * sp.Rational(4, 3) * sp.pi * E ** sp.Rational(3, 2)
    return N, sp.diff(N, E), E


def translational_Z_symbolic():
    """Z_1 = (L / lambda)^3 from the Gaussian integral over each momentum
    component (classical limit), lambda = h / sqrt(2 pi m k T)."""
    m, L, h, p = sp.symbols("m L h p", positive=True)
    Z1d = L / h * sp.integrate(sp.exp(-p**2 / (2 * m * k * T)), (p, -sp.oo, sp.oo))
    lam = h / sp.sqrt(2 * sp.pi * m * k * T)
    return sp.simplify(Z1d**3), sp.simplify(Z1d**3 - (L / lam) ** 3)


def sackur_tetrode_symbolic():
    """S = k ln(Z_1^N / N!) + U/T with Stirling ln N! ~ N ln N - N, U = 3/2 N k T:
    S = N k [ ln(V / (N lambda^3)) + 5/2 ]."""
    N, V, lam = sp.symbols("N V lambda", positive=True)
    lnQ = N * sp.log(V / lam**3) - (N * sp.log(N) - N)
    S = k * lnQ + sp.Rational(3, 2) * N * k
    target = N * k * (sp.log(V / (N * lam**3)) + sp.Rational(5, 2))
    return sp.simplify(sp.expand_log(S - target, force=True))


def thermal_wavelength(mass_kg, T_K):
    return H / np.sqrt(2 * np.pi * mass_kg * K_B * T_K)


def sackur_tetrode_molar(mass_amu, T_K=298.15, P_Pa=1e5):
    """Molar entropy [J/(mol K)] of a monatomic ideal gas."""
    lam = thermal_wavelength(mass_amu * AMU, T_K)
    v_per = K_B * T_K / P_Pa
    return N_A * K_B * (np.log(v_per / lam**3) + 2.5)


def maxwell_speed_symbolic():
    """3D speed distribution f(v) = 4 pi v^2 (m/2 pi k T)^{3/2} exp(-m v^2 / 2kT):
    normalization, mean speed, rms speed, most probable speed."""
    m, v = sp.symbols("m v", positive=True)
    f = 4 * sp.pi * v**2 * (m / (2 * sp.pi * k * T)) ** sp.Rational(3, 2) * sp.exp(-m * v**2 / (2 * k * T))
    norm = sp.simplify(sp.integrate(f, (v, 0, sp.oo)))
    mean = sp.simplify(sp.integrate(v * f, (v, 0, sp.oo)))
    rms = sp.simplify(sp.sqrt(sp.integrate(v**2 * f, (v, 0, sp.oo))))
    vp = [s for s in sp.solve(sp.diff(f, v), v) if s.is_positive][0]
    return norm, mean, rms, sp.simplify(vp)


# ── information is physical ─────────────────────────────────────────

def bit_entropy_symbolic():
    """Entropy drop when a fair bit (2 equally likely states) is reset to 1 state: k ln 2."""
    return sp.simplify(k * sp.log(2) - k * sp.log(1))


def szilard_work_symbolic():
    """One molecule, isothermal expansion V/2 -> V: W = int p dV with p V = k T."""
    V, Vp = sp.symbols("V V'", positive=True)
    return sp.simplify(sp.integrate(k * T / Vp, (Vp, V / 2, V)))


def landauer_limit(T_K):
    """Minimum heat to erase one bit: k T ln 2 [J]."""
    if np.any(np.asarray(T_K, float) <= 0):
        raise ValueError("temperature must be positive")
    return K_B * np.asarray(T_K, float) * np.log(2)


# ── QIS-in-practice: the cryogenic engineering ──────────────────────

def required_temperature(freq_hz, max_excited=1e-3):
    """Temperature at which a qubit/cavity mode's thermal excited population
    drops to max_excited (two-level: p1 = 1/(1+e^x))."""
    if not 0 < max_excited < 0.5:
        raise ValueError("max_excited must be in (0, 0.5)")
    x = np.log(1 / max_excited - 1)
    return H * freq_hz / (K_B * x)
