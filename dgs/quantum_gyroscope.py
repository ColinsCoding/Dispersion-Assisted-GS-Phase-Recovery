"""Quantum gyroscopes: the Sagnac effect for light and for atoms, interferometers in
transfer-matrix and qubit notation, the Heisenberg limit, and the CAD/CAM of a
fiber-gyro coil.

  * SAGNAC (light). Two beams circle a loop rotating at Omega. Their transit
    times differ by dt ~ 4 A Omega / c^2, a phase  dphi = 8 pi A Omega / (lambda c).
  * SAGNAC (matter). For massive particles  dphi = 2 m A Omega / hbar, from
    Stokes: closed-loop integral of m (Omega x r) . dl = 2 m Omega A. The ratio
    to light is m c^2 / (2 hbar omega) ~ 1e10 (often quoted without the 2) --
    the reason atoms make exquisite gyroscopes.
  * ATOM GYRO. In the rotating frame an atom moving at v feels the Coriolis
    acceleration -2 Omega x v. Plugging it into the Mach-Zehnder phase
    k_eff a T^2 gives  phi = -2 k_eff . (Omega x v) T^2.
  * NOTATION. An interferometer is a product of 2x2 transfer matrices
    (beam splitter, phase, beam splitter). In QIS notation the same product is
    the circuit H R_z(phi) H on a qubit; P(output) = (1 + cos phi)/2.
  * QUANTUM INFORMATION TECHNOLOGY. N independent atoms: dphi = 1/sqrt(N)
    (shot noise). N atoms in a GHZ state: dphi = 1/N (Heisenberg limit).
  * CAD/CAM. A spiral coil's Sagnac scale factor is its enclosed area,
    computed from the toolpath by Green's theorem (the shoelace formula). The
    same path is exported as DXF (CAD) and G-code (CAM).

Civilian education; synthetic numbers.
"""
import numpy as np
import sympy as sp

HBAR = 1.054571817e-34
H_PLANCK = 6.62607015e-34
C = 299792458.0
AMU = 1.66053906660e-27
M_RB87 = 86.909180527 * AMU
M_CS133 = 132.905451961 * AMU
OMEGA_EARTH = 7.2921150e-5          # rad/s, sidereal
RB87_WAVELENGTH = 780.241e-9


# ── 1. Sagnac derivations ────────────────────────────────────────────

def sagnac_time_symbolic():
    """Ring of radius R rotating at Omega. Co-rotating light must go farther:
    c t_plus = 2 pi R + R Omega t_plus; counter: c t_minus = 2 pi R - R Omega t_minus.
    Returns (exact dt, first-order dt in A = pi R^2)."""
    R, W, c, t = sp.symbols("R Omega c t", positive=True)
    tp = sp.solve(sp.Eq(c * t, 2 * sp.pi * R + R * W * t), t)[0]
    tm = sp.solve(sp.Eq(c * t, 2 * sp.pi * R - R * W * t), t)[0]
    dt = sp.simplify(tp - tm)
    first = sp.series(dt, W, 0, 2).removeO()
    A = sp.symbols("A", positive=True)
    return dt, sp.simplify(first.subs(R, sp.sqrt(A / sp.pi)))


def sagnac_phase_light_symbolic():
    """dphi = omega dt = (2 pi c / lambda) * 4 A Omega / c^2 = 8 pi A Omega / (lambda c)."""
    A, W, c, lam = sp.symbols("A Omega c lambda", positive=True)
    _, dt = sagnac_time_symbolic()
    return sp.simplify(2 * sp.pi * c / lam * dt.subs(sp.Symbol("A", positive=True), A))


def rotating_frame_circulation_symbolic():
    """curl(Omega x r) = 2 Omega (rigid rotation about z), and its flux through
    a disk of radius R is 2 Omega pi R^2. Returns (curl vector, circulation)."""
    x, y, z = sp.symbols("x y z", real=True)
    W, R, rr, th = sp.symbols("Omega R r theta", positive=True)
    v = sp.Matrix([0, 0, W]).cross(sp.Matrix([x, y, z]))
    curl = sp.Matrix([sp.diff(v[2], y) - sp.diff(v[1], z),
                      sp.diff(v[0], z) - sp.diff(v[2], x),
                      sp.diff(v[1], x) - sp.diff(v[0], y)])
    # direct line integral around the circle, as a check on Stokes
    xs, ys = R * sp.cos(th), R * sp.sin(th)
    integrand = (v[0].subs({x: xs, y: ys}) * sp.diff(xs, th) + v[1].subs({x: xs, y: ys}) * sp.diff(ys, th))
    circ = sp.simplify(sp.integrate(integrand, (th, 0, 2 * sp.pi)))
    return curl, circ


def sagnac_phase_matter_symbolic():
    """dphi = (1/hbar) loop-integral of m (Omega x r) . dl = 2 m Omega A / hbar."""
    m, hb, A = sp.symbols("m hbar A", positive=True)
    _, circ = rotating_frame_circulation_symbolic()
    R, W = sp.symbols("R Omega", positive=True)
    return sp.simplify((m * circ / hb).subs(R, sp.sqrt(A / sp.pi)))


def matter_light_ratio_symbolic():
    """(matter phase)/(light phase) for the same area: m c^2 / (2 hbar omega).
    Returns (ratio, ratio - m c^2/(2 hbar omega)) -- the second entry is 0."""
    m, hb, c, lam, A, W = sp.symbols("m hbar c lambda A Omega", positive=True)
    ratio = sagnac_phase_matter_symbolic() / sagnac_phase_light_symbolic()
    omega = 2 * sp.pi * c / lam
    return sp.simplify(ratio), sp.simplify(ratio - m * c**2 / (2 * hb * omega))


def atom_gyro_phase_symbolic():
    """Coriolis acceleration a = -2 Omega x v inserted into the three-pulse
    phase k_eff [x(0) - 2 x(T) + x(2T)]. Returns phi (vectors along x, y, z
    chosen as k_eff = k x_hat, v = v y_hat, Omega = Omega z_hat)."""
    k, v, W, T, t = sp.symbols("k_eff v Omega T t", positive=True)
    kv = sp.Matrix([k, 0, 0])
    vel = sp.Matrix([0, v, 0])
    Om = sp.Matrix([0, 0, W])
    a = -2 * Om.cross(vel)
    pos = lambda tt: vel * tt + a * tt**2 / 2
    phi = kv.dot(pos(0) - 2 * pos(T) + pos(2 * T))
    return sp.simplify(phi)


# ── numbers ──────────────────────────────────────────────────────────

def _pos(**kw):
    for n, v in kw.items():
        if np.any(np.asarray(v, float) <= 0):
            raise ValueError(f"{n} must be positive")


def sagnac_phase_light(area_m2, omega, wavelength, n_turns=1):
    _pos(area_m2=area_m2, wavelength=wavelength, n_turns=n_turns)
    return 8 * np.pi * n_turns * area_m2 * omega / (wavelength * C)


def sagnac_phase_matter(area_m2, omega, mass):
    _pos(area_m2=area_m2, mass=mass)
    return 2 * mass * area_m2 * omega / HBAR


def matter_light_gain(mass, wavelength):
    """Sagnac phase ratio, matter / light, same area: m c^2 / (2 hbar omega) = m c lambda / (2 h)."""
    _pos(mass=mass, wavelength=wavelength)
    return mass * C * wavelength / (2 * H_PLANCK)


def atom_gyro_phase(omega, v, T, wavelength=RB87_WAVELENGTH):
    """|phi| = 2 k_eff v Omega T^2 for perpendicular k, v, Omega."""
    _pos(v=v, T=T, wavelength=wavelength)
    return 2 * (4 * np.pi / wavelength) * v * omega * T**2


def atom_gyro_area(v, T, mass=M_RB87, wavelength=RB87_WAVELENGTH):
    """Enclosed area: arms separate at the recoil velocity hbar k_eff / m for
    time T, while the atoms travel at v: A = v_rec v T^2."""
    v_rec = HBAR * (4 * np.pi / wavelength) / mass
    return v_rec * v * T**2


def gyro_sensitivity(scale_factor_rad_per_rad_s, n_particles, contrast=1.0):
    """Shot-noise-limited rotation resolution per shot: 1/(C sqrt(N) S)."""
    _pos(scale_factor_rad_per_rad_s=scale_factor_rad_per_rad_s, n_particles=n_particles)
    return 1.0 / (contrast * np.sqrt(n_particles) * scale_factor_rad_per_rad_s)


def deg_per_hour(rad_per_s):
    return np.asarray(rad_per_s) * 180 / np.pi * 3600


# ── 2. transfer-matrix notation <-> qubit notation ──────────────────

def beam_splitter_symbolic():
    """Symmetric 50/50 beam splitter transfer matrix (lossless, unitary)."""
    return sp.Matrix([[1, sp.I], [sp.I, 1]]) / sp.sqrt(2)


def phase_symbolic(phi=None):
    phi = sp.symbols("phi", real=True) if phi is None else phi
    return sp.Matrix([[sp.exp(sp.I * phi), 0], [0, 1]])


def mach_zehnder_symbolic():
    """Transfer matrix B P(phi) B and the output-port probabilities for light
    (or an atom) entering port 0."""
    phi = sp.symbols("phi", real=True)
    M = sp.simplify(beam_splitter_symbolic() * phase_symbolic(phi) * beam_splitter_symbolic())
    out = M * sp.Matrix([1, 0])
    P = [sp.simplify(sp.expand_complex(sp.Abs(o) ** 2).rewrite(sp.cos)) for o in out]
    return M, [sp.simplify(sp.trigsimp(p)) for p in P]


def mz_equals_qubit_circuit():
    """The transfer-matrix Mach-Zehnder B P(phi) B and the qubit circuit
    H Rz(phi) H give the same port probabilities once the output ports are
    relabeled (an X gate). Returns probs(MZ) - probs(X H Rz H): the zero matrix."""
    phi = sp.symbols("phi", real=True)
    Hd = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    Rz = sp.Matrix([[sp.exp(-sp.I * phi / 2), 0], [0, sp.exp(sp.I * phi / 2)]])
    X = sp.Matrix([[0, 1], [1, 0]])
    M, _ = mach_zehnder_symbolic()
    probs = lambda A: A.applyfunc(lambda e: sp.simplify(sp.trigsimp(sp.expand_complex(sp.Abs(e) ** 2).rewrite(sp.cos))))
    return sp.simplify(probs(M) - probs(X * Hd * Rz * Hd))


def mz_port_probabilities_qubit():
    """P(0), P(1) for H Rz(phi) H |0>."""
    phi = sp.symbols("phi", real=True)
    Hd = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    Rz = sp.Matrix([[sp.exp(-sp.I * phi / 2), 0], [0, sp.exp(sp.I * phi / 2)]])
    out = Hd * Rz * Hd * sp.Matrix([1, 0])
    return [sp.simplify(sp.trigsimp(sp.expand_complex(sp.Abs(o) ** 2))) for o in out]


# ── 3. Heisenberg limit with GHZ states ──────────────────────────────

def ghz_phase_state(n, phi):
    """(|0...0> + e^{i n phi} |1...1>)/sqrt(2): each qubit picked up phi."""
    v = np.zeros(2**n, complex)
    v[0] = 1 / np.sqrt(2)
    v[-1] = np.exp(1j * n * phi) / np.sqrt(2)
    return v


def product_phase_state(n, phi):
    """n independent qubits each in (|0> + e^{i phi}|1>)/sqrt(2)."""
    one = np.array([1, np.exp(1j * phi)]) / np.sqrt(2)
    v = np.array([1.0 + 0j])
    for _ in range(n):
        v = np.kron(v, one)
    return v


def quantum_fisher_information(state_fn, n, phi=0.3, h=1e-5):
    """Pure-state QFI F_Q = 4 (<d psi|d psi> - |<psi|d psi>|^2), by central
    differences. Cramer-Rao: dphi >= 1/sqrt(F_Q)."""
    psi = state_fn(n, phi)
    d = (state_fn(n, phi + h) - state_fn(n, phi - h)) / (2 * h)
    return float(4 * (np.vdot(d, d).real - abs(np.vdot(psi, d)) ** 2))


def ghz_parity_fringe(n, phis):
    """<X^{(x)n}> for the GHZ state = cos(n phi): the fringe oscillates n times faster."""
    out = []
    for p in np.atleast_1d(phis):
        v = ghz_phase_state(n, p)
        # X^{(x)n} maps |0..0> <-> |1..1>
        out.append(2 * np.real(np.conj(v[0]) * v[-1]))
    return np.array(out)


# ── 4. CAD/CAM: spiral coil, Green's theorem, DXF and G-code ─────────

def spiral_path(r_inner, pitch, n_turns, pts_per_turn=360):
    """Archimedean spiral r = r_inner + pitch theta / (2 pi), closed back to
    the start by a straight return lead (so the loop encloses a definite area)."""
    _pos(r_inner=r_inner, pitch=pitch, n_turns=n_turns, pts_per_turn=pts_per_turn)
    th = np.linspace(0, 2 * np.pi * n_turns, int(pts_per_turn * n_turns) + 1)
    r = r_inner + pitch * th / (2 * np.pi)
    xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return np.vstack([xy, xy[:1]])


def shoelace_area(xy):
    """Signed area 1/2 sum (x_i y_{i+1} - x_{i+1} y_i): Green's theorem
    A = 1/2 loop-integral (x dy - y dx) on a polygon. For a self-overlapping
    spiral this is the winding-number-weighted area = the Sagnac area."""
    x, y = np.asarray(xy, float).T
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def path_length(xy):
    return float(np.sum(np.hypot(*np.diff(np.asarray(xy, float), axis=0).T)))


def spiral_area_symbolic():
    """Winding-weighted area of the spiral: integral of r^2/2 dtheta over N turns
    (the straight return lead sweeps no extra area about the origin)."""
    r0, p, N, th = sp.symbols("r_0 p N theta", positive=True)
    r = r0 + p * th / (2 * sp.pi)
    return sp.simplify(sp.integrate(r**2 / 2, (th, 0, 2 * sp.pi * N)))


def to_dxf(xy, units_mm=True):
    """Minimal ASCII DXF (R12 subset): one LINE entity per segment, layer COIL.
    Readable by common CAD programs; no external library."""
    s = 1e3 if units_mm else 1.0
    out = ["0", "SECTION", "2", "ENTITIES"]
    for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
        out += ["0", "LINE", "8", "COIL", "10", f"{x1*s:.4f}", "20", f"{y1*s:.4f}", "30", "0.0",
                "11", f"{x2*s:.4f}", "21", f"{y2*s:.4f}", "31", "0.0"]
    out += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(out) + "\n"


def read_dxf_lines(text):
    """Parse back the LINE entities written by to_dxf -> array of segments (mm)."""
    tok = [t.strip() for t in text.splitlines()]
    segs = []
    for i, t in enumerate(tok):
        if t == "LINE" and tok[i - 1] == "0":
            g = dict(zip(tok[i + 1:i + 17:2], tok[i + 2:i + 18:2]))
            segs.append([float(g["10"]), float(g["20"]), float(g["11"]), float(g["21"])])
    return np.array(segs)


def to_svg(xy, size_px=500):
    x, y = np.asarray(xy, float).T
    span = max(np.ptp(x), np.ptp(y))
    sx = (x - x.min()) / span * (size_px - 20) + 10
    sy = (y.max() - y) / span * (size_px - 20) + 10
    d = "M " + " L ".join(f"{a:.2f},{b:.2f}" for a, b in zip(sx, sy))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}">'
            f'<title>Sagnac coil toolpath</title><path d="{d}" fill="none" stroke="#0072B2" stroke-width="1"/></svg>')


def to_gcode(xy, depth_mm=0.15, safe_z_mm=2.0, feed_mm_min=300.0, plunge_mm_min=60.0):
    """G-code for engraving the coil groove: mm, absolute, one plunge, one
    continuous cut, retract. Returns (gcode text, cut length in mm)."""
    _pos(depth_mm=depth_mm, safe_z_mm=safe_z_mm, feed_mm_min=feed_mm_min, plunge_mm_min=plunge_mm_min)
    p = np.asarray(xy, float) * 1e3
    g = ["(Sagnac coil groove -- generated by dgs.quantum_gyroscope)", "G21 (mm)", "G90 (absolute)",
         f"G0 Z{safe_z_mm:.3f}", f"G0 X{p[0,0]:.4f} Y{p[0,1]:.4f}",
         f"G1 Z{-depth_mm:.3f} F{plunge_mm_min:.0f}"]
    g += [f"G1 X{a:.4f} Y{b:.4f} F{feed_mm_min:.0f}" if i == 0 else f"G1 X{a:.4f} Y{b:.4f}"
          for i, (a, b) in enumerate(p[1:])]
    g += [f"G0 Z{safe_z_mm:.3f}", "M2"]
    return "\n".join(g) + "\n", path_length(p)


def gcode_moves(text):
    """Extract (x, y) of every G1 cutting move with an XY target."""
    pts = []
    for line in text.splitlines():
        if line.startswith("G1") and "X" in line:
            f = dict((w[0], float(w[1:])) for w in line.split() if w[0] in "XY")
            pts.append((f["X"], f["Y"]))
    return np.array(pts)


def coil_scale_factor(xy, wavelength=1550e-9):
    """Sagnac scale factor S = dphi / dOmega = 8 pi A_eff / (lambda c) for light
    guided along the toolpath; A_eff from Green's theorem."""
    return 8 * np.pi * abs(shoelace_area(xy)) / (wavelength * C)
