"""NV-diamond quantum magnetometer, end to end: spin physics, optical mounts, op-amp readout.

  * SPIN PHYSICS. The NV- center's ground state is spin 1:
        H = D S_z^2 + gamma_e B . S,  D = 2.870 GHz, gamma_e = 28.025 GHz/T.
    Microwaves drive m_s = 0 -> +/-1; on resonance the green-pumped red
    fluorescence dips (optically detected magnetic resonance, ODMR). For B along
    the NV axis the dips sit at D +/- gamma_e B: their splitting measures B.
  * SENSITIVITY. A Lorentzian dip of contrast C and FWHM Gamma has maximum slope
    3 sqrt(3) C / (4 Gamma). With photon rate R (shot noise) the field resolution
    is  eta = 4 Gamma / (3 sqrt(3) gamma_e C sqrt(R))  [T/sqrt(Hz)].
  * VECTOR. Diamond hosts NVs along the four <111> directions; each measures the
    projection of B on its own axis. Four projections -> least squares -> B.
  * OPTICAL MOUNTS. A kinematic mirror mount turns a screw turn into an angle
    p/L; a mirror doubles it in the beam. Temperature drift tilts the mirror the
    same way. Beam-pointing drift moves the laser spot on the diamond and
    changes the fluorescence -- intensity noise that mimics a field signal.
  * OP-AMP. The fluorescence lands on a photodiode feeding a transimpedance
    amplifier. SymPy derives its closed-loop transfer function with a
    single-pole op-amp and the C_f that makes it maximally flat, then checks the
    approximation used in dgs.transimpedance_amplifier.

Synthetic numbers; typical literature magnitudes, not a specific device. Education.
"""
import numpy as np
import sympy as sp

from dgs import transimpedance_amplifier as tia

D_ZFS = 2.870e9            # zero-field splitting [Hz]
GAMMA_E = 28.025e9         # electron gyromagnetic ratio [Hz/T]
Q_E = 1.602176634e-19
K_B = 1.380649e-23

NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)


# ── 1. spin-1 Hamiltonian ────────────────────────────────────────────

def spin1_matrices():
    """S_x, S_y, S_z for spin 1 (hbar = 1), basis m = +1, 0, -1."""
    r = 1 / sp.sqrt(2)
    Sx = r * sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    Sy = r * sp.Matrix([[0, -sp.I, 0], [sp.I, 0, -sp.I], [0, sp.I, 0]])
    Sz = sp.diag(1, 0, -1)
    return Sx, Sy, Sz


def spin1_algebra_check():
    """[S_x, S_y] = i S_z and S^2 = s(s+1) = 2: zero matrices if correct."""
    Sx, Sy, Sz = spin1_matrices()
    return sp.simplify(Sx * Sy - Sy * Sx - sp.I * Sz), sp.simplify(Sx**2 + Sy**2 + Sz**2 - 2 * sp.eye(3))


def nv_hamiltonian_symbolic(axial_only=True):
    """H = D S_z^2 + gamma (B_z S_z [+ B_x S_x]). Returns (H, eigenvalues)."""
    D, g, Bz, Bx = sp.symbols("D gamma_e B_z B_x", real=True)
    Sx, Sy, Sz = spin1_matrices()
    H = D * Sz**2 + g * Bz * Sz
    if not axial_only:
        H += g * Bx * Sx
    return H, list(H.eigenvals().keys())


def odmr_frequencies_symbolic():
    """Transition frequencies m=0 -> +/-1 for axial field: D +/- gamma B."""
    H, ev = nv_hamiltonian_symbolic(True)
    e0 = H[1, 1]                                               # the m = 0 level (energy 0)
    return sorted([sp.simplify(e - e0) for e in ev if sp.simplify(e - e0) != 0], key=str)


def odmr_frequencies(B_vec, axis=(0, 0, 1)):
    """Numeric transition frequencies [Hz] for an arbitrary field vector B [T]
    seen by an NV along `axis` (exact 3x3 diagonalization)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    B = np.asarray(B_vec, float)
    # rotate B into the NV frame: z' = axis
    tmp = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    x_ax = np.cross(axis, tmp)
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(axis, x_ax)
    bx, by, bz = B @ x_ax, B @ y_ax, B @ axis
    r = 1 / np.sqrt(2)
    Sx = r * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex)
    Sy = r * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    Sz = np.diag([1.0, 0, -1.0]).astype(complex)
    H = D_ZFS * Sz @ Sz + GAMMA_E * (bx * Sx + by * Sy + bz * Sz)
    w = np.linalg.eigvalsh(H)
    return np.sort(w[1:] - w[0])


# ── 2. ODMR line shape and sensitivity ───────────────────────────────

def lorentzian_dip(f, f0, contrast, fwhm):
    """Normalized fluorescence 1 - C (Gamma/2)^2 / ((f - f0)^2 + (Gamma/2)^2)."""
    hw = fwhm / 2
    return 1 - contrast * hw**2 / ((np.asarray(f, float) - f0) ** 2 + hw**2)


def max_slope_symbolic():
    """Maximum |dS/df| of the Lorentzian dip and where it occurs."""
    f, C, G = sp.symbols("f C Gamma", positive=True)
    S = 1 - C * (G / 2) ** 2 / (f**2 + (G / 2) ** 2)
    dS = sp.diff(S, f)
    f_star = [s for s in sp.solve(sp.diff(dS, f), f) if s.is_positive][0]
    return sp.simplify(f_star), sp.simplify(dS.subs(f, f_star))


def sensitivity_symbolic():
    """eta = 1 / (gamma * max slope * sqrt(R)): shot-noise-limited field resolution."""
    C, G, g, R = sp.symbols("C Gamma gamma_e R", positive=True)
    _, slope = max_slope_symbolic()
    return sp.simplify(1 / (g * slope * sp.sqrt(R)))


def sensitivity(fwhm_hz, contrast, photon_rate):
    """eta [T/sqrt(Hz)] = 4 Gamma / (3 sqrt(3) gamma_e C sqrt(R))."""
    for n, v in (("fwhm_hz", fwhm_hz), ("contrast", contrast), ("photon_rate", photon_rate)):
        if v <= 0:
            raise ValueError(f"{n} must be positive")
    if contrast > 1:
        raise ValueError("contrast must be <= 1")
    return 4 * fwhm_hz / (3 * np.sqrt(3) * GAMMA_E * contrast * np.sqrt(photon_rate))


def simulate_odmr(B_vec, freqs, contrast=0.03, fwhm=8e6, counts_per_point=2e5, rng=None, axes=NV_AXES):
    """Ensemble ODMR: sum of dips from all four NV orientations (each with its
    two transitions), Poisson photon noise. Returns normalized counts."""
    rng = np.random.default_rng(rng)
    s = np.ones_like(np.asarray(freqs, float))
    for ax in axes:
        for f0 in odmr_frequencies(B_vec, ax):
            s *= lorentzian_dip(freqs, f0, contrast, fwhm)
    return rng.poisson(s * counts_per_point) / counts_per_point


def find_dips(freqs, signal, n_dips=8, min_sep=6e6):
    """Greedy minimum picking with a separation guard, then a 3-point
    parabolic refinement of each minimum."""
    freqs, signal = np.asarray(freqs), np.asarray(signal)
    sm = np.convolve(np.pad(signal, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    order = np.argsort(sm)
    picked = []
    for i in order:
        if all(abs(freqs[i] - freqs[j]) > min_sep for j in picked):
            picked.append(i)
        if len(picked) == n_dips:
            break
    out = []
    for i in sorted(picked):
        i = min(max(i, 1), len(freqs) - 2)
        y0, y1, y2 = sm[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        out.append(freqs[i] + shift * (freqs[1] - freqs[0]))
    return np.array(out)


def vector_from_projections(projections_T, axes=NV_AXES):
    """Least-squares B from four unlabeled magnitudes |B . n_k|.
    A spectrum alone does not say WHICH axis produced which splitting, nor the
    sign: every field related to the true one by the crystal's tetrahedral
    symmetry (and B -> -B) gives the same spectrum. This search over all axis
    assignments and signs returns ONE member of that family -- correct only up
    to symmetry. Use recover_vector_with_bias for a unique answer."""
    from itertools import permutations
    proj = np.asarray(projections_T, float)
    best = None
    for perm in permutations(range(4)):
        for signs in np.array(np.meshgrid(*[[1, -1]] * 4)).T.reshape(-1, 4):
            y = signs * proj[list(perm)]
            b = np.linalg.lstsq(axes, y, rcond=None)[0]
            r = np.sum((axes @ b - y) ** 2)
            if best is None or r < best[0] - 1e-30:
                best = (r, b)
    return best[1]


def symmetry_equivalent(b1, b2, axes=NV_AXES, rtol=2e-2):
    """True if b1 and b2 give the same set of |projections| (same ODMR spectrum)."""
    p1 = np.sort(np.abs(axes @ np.asarray(b1, float)))
    p2 = np.sort(np.abs(axes @ np.asarray(b2, float)))
    return bool(np.allclose(p1, p2, rtol=rtol, atol=rtol * np.max(p2)))


def recover_vector_with_bias(B_unknown, B_bias, freqs, rng=None, **odmr_kw):
    """Apply a KNOWN bias field chosen so its four projections are distinct and
    larger than the unknown's. Then the dip pairs, sorted by splitting, are
    labeled by the bias ordering and their signs by the bias signs. Solve for
    the total field by least squares and subtract the bias: a unique B."""
    B_bias = np.asarray(B_bias, float)
    s = simulate_odmr(np.asarray(B_unknown, float) + B_bias, freqs, rng=rng, **odmr_kw)
    proj = projections_from_dips(find_dips(freqs, s))          # sorted large -> small
    bias_proj = NV_AXES @ B_bias
    order = np.argsort(-np.abs(bias_proj))                       # axes from largest to smallest bias projection
    y = np.empty(4)
    y[order] = proj * np.sign(bias_proj[order])
    B_total = np.linalg.lstsq(NV_AXES, y, rcond=None)[0]
    return B_total - B_bias


def projections_from_dips(dips_hz):
    """Pair the 8 sorted dips symmetrically about D: |B_k| ~ (f_hi - f_lo) / (2 gamma)."""
    d = np.sort(np.asarray(dips_hz))
    return (d[::-1][:4] - d[:4]) / (2 * GAMMA_E)


# ── 3. optical mounts ────────────────────────────────────────────────

def mount_symbolic():
    """Kinematic mount: screw pitch p, lever arm L -> mirror tilt p/L per turn;
    a reflected beam turns by twice that; at distance d the spot moves 2 d p / L.
    Thermal: differential expansion (alpha_s - alpha_m) h dT over lever L tilts
    by (alpha_s - alpha_m) h dT / L. Returns dict of expressions."""
    p, L, d, n, a_s, a_m, h, dT = sp.symbols("p L d n alpha_s alpha_m h DeltaT", positive=True)
    tilt = n * p / L
    return {"tilt_per_turns": tilt, "spot_shift": 2 * d * tilt,
            "thermal_tilt": (a_s - a_m) * h * dT / L, "thermal_spot": 2 * d * (a_s - a_m) * h * dT / L}


def tpi_to_pitch_m(tpi):
    """Threads per inch -> pitch in meters."""
    if tpi <= 0:
        raise ValueError("tpi must be positive")
    return 25.4e-3 / tpi


def spot_shift(turns, tpi, lever_m, distance_m):
    return 2 * distance_m * turns * tpi_to_pitch_m(tpi) / lever_m


def thermal_spot_drift(dT, screw_length_m, lever_m, distance_m, alpha_screw=17.3e-6, alpha_mount=23.1e-6):
    """Spot drift from a steel screw in an aluminum body (defaults: stainless vs Al)."""
    return 2 * distance_m * abs(alpha_screw - alpha_mount) * screw_length_m * dT / lever_m


def fluorescence_vs_offset(offset_m, spot_waist_m):
    """Collected fluorescence relative to perfect alignment when a Gaussian
    pump spot (1/e^2 radius w) is displaced by delta from a small diamond:
    exp(-2 delta^2 / w^2)."""
    return np.exp(-2 * np.asarray(offset_m, float) ** 2 / spot_waist_m**2)


def false_field_from_pointing(offset_m, d_offset_m, spot_waist_m, fwhm_hz, contrast):
    """Intensity change from a pointing jitter d_offset at operating offset
    delta, divided by the ODMR slope -> the magnetic field it masquerades as
    (if not normalized out). |dI/I| = 4 delta d_delta / w^2."""
    dI = 4 * offset_m * d_offset_m / spot_waist_m**2
    slope_per_T = GAMMA_E * 3 * np.sqrt(3) * contrast / (4 * fwhm_hz)
    return dI / slope_per_T


# ── 4. op-amp transimpedance readout ─────────────────────────────────

def tia_closed_loop_symbolic():
    """KCL at the inverting node with photodiode capacitance C_d, feedback
    R_f || C_f, and a single-pole op-amp A(s) = w_t / s:
        I = v s C_d + (v - V_o) / Z_f,   V_o = -A v.
    Returns (V_o / I, omega_0, Q) of the resulting second-order system."""
    s, Rf, Cf, Cd, wt, I = sp.symbols("s R_f C_f C_d omega_t I", positive=True)
    v, Vo = sp.symbols("v V_o")
    Zf = Rf / (1 + s * Rf * Cf)
    A = wt / s
    sol = sp.solve([sp.Eq(I, v * s * Cd + (v - Vo) / Zf), sp.Eq(Vo, -A * v)], [v, Vo], dict=True)[0]
    Ht = sp.simplify(sol[Vo] / I)
    num, den = sp.fraction(sp.together(Ht))
    den = sp.Poly(sp.expand(den), s)
    a2, a1, a0 = [den.coeff_monomial(s**k) for k in (2, 1, 0)]
    w0 = sp.sqrt(a0 / a2)
    Q = sp.simplify(sp.sqrt(a0 * a2) / a1)
    return Ht, sp.simplify(w0), Q


def tia_flat_cf_symbolic():
    """Solve Q(C_f) = 1/sqrt(2) exactly (Butterworth). Returns the positive root."""
    _, _, Q = tia_closed_loop_symbolic()
    Rf, Cf, Cd, wt = sp.symbols("R_f C_f C_d omega_t", positive=True)
    sols = sp.solve(sp.Eq(Q**2, sp.Rational(1, 2)), Cf)
    return [sp.simplify(x) for x in sols if x.is_positive is not False]


def tia_flat_cf(R_f, C_d, f_gbw):
    """Exact Butterworth C_f from the symbolic solution, evaluated numerically."""
    Rf, Cd, wt = sp.symbols("R_f C_d omega_t", positive=True)
    vals = [float(e.subs({Rf: R_f, Cd: C_d, wt: 2 * np.pi * f_gbw})) for e in tia_flat_cf_symbolic()]
    vals = [v for v in vals if v > 0]
    return min(vals)


def tia_response(f, R_f, C_f, C_d, f_gbw):
    """|V_o / I| [ohm] of the closed-loop TIA (single-pole op-amp)."""
    s = 2j * np.pi * np.asarray(f, float)
    wt = 2 * np.pi * f_gbw
    Zf = R_f / (1 + s * R_f * C_f)
    A = wt / s
    # from the KCL solution: V_o/I = -A Zf / (1 + A + s C_d Zf)
    return np.abs(-A * Zf / (1 + A + s * C_d * Zf))


def shot_noise_limited_rf(photocurrent, T=300.0):
    """Johnson current noise of R_f equals photocurrent shot noise when
    4 k T / R_f = 2 q I  ->  R_f = 2 k T / (q I); larger R_f is shot-noise limited.
    Equivalent statement: I R_f > 2 kT/q ~ 52 mV at 300 K."""
    if photocurrent <= 0:
        raise ValueError("photocurrent must be positive")
    return 2 * K_B * T / (Q_E * photocurrent)


def readout_noise_budget(P_opt_W, wavelength_nm, R_f, C_d, B_hz, qe=0.8, e_n=5e-9):
    """Input-referred noise currents [A] from dgs.transimpedance_amplifier."""
    resp = tia.responsivity(wavelength_nm, qe)
    I = tia.photocurrent(P_opt_W, resp)
    return {"photocurrent": I,
            "shot": float(tia.shot_noise_current(I, B_hz)),
            "thermal": float(tia.thermal_noise_current(R_f, B_hz)),
            "amplifier": float(tia.amplifier_noise_current(e_n, C_d, B_hz))}
