"""Applied quantum mechanics at engineering-implementation depth.

The numerical toolkit an engineer actually reaches for: a finite-difference
eigensolver for the time-independent Schrodinger equation (quantum wells), a
scattering integrator for tunnelling (tunnel diodes, STM, flash memory), the
harmonic-oscillator ladder (phonons, photons), two-level Rabi dynamics
(qubits, modulators), and the split-operator propagator -- which is bit-for-bit
the split-step Fourier method used for pulse propagation in fibre.

Natural units (hbar = m = 1) by default; pass SI values to recover real
numbers. Companion to the rest of the `griffiths` physics engine.
"""

import numpy as np


# ── time-independent Schrodinger equation (bound states) ────────────
def solve_tise(x, V, n_states=6, hbar=1.0, m=1.0):
    """Finite-difference eigensolver for H psi = E psi on a 1-D grid.

    H = -hbar^2/(2m) d^2/dx^2 + V(x), discretised with Dirichlet walls at the
    grid ends (i.e. an infinite wall just outside [x0, xN]).  Returns
    (energies[:n_states], psi[:, :n_states]) with psi normalised so that
    sum |psi|^2 dx = 1.
    """
    x = np.asarray(x, dtype=float)
    V = np.asarray(V, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a 1-D grid of at least 3 points")
    if V.shape != x.shape:
        raise ValueError(f"V must match x shape; got {V.shape} vs {x.shape}")
    if n_states < 1 or n_states > x.size:
        raise ValueError(f"n_states must be in [1, {x.size}], got {n_states}")
    dx = x[1] - x[0]
    coeff = hbar**2 / (2 * m * dx**2)
    main = 2 * coeff + V
    off = -coeff * np.ones(x.size - 1)
    H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    E, psi = np.linalg.eigh(H)
    psi = psi / np.sqrt(dx)               # discrete -> continuum normalisation
    return E[:n_states], psi[:, :n_states]


def infinite_well_energies(n, L, hbar=1.0, m=1.0):
    """Analytic levels E_n = n^2 pi^2 hbar^2 / (2 m L^2), n = 1, 2, ... ."""
    n = np.asarray(n)
    if np.any(n < 1):
        raise ValueError("quantum number n must be >= 1")
    return n**2 * np.pi**2 * hbar**2 / (2 * m * L**2)


def harmonic_energies(n, omega, hbar=1.0):
    """Analytic levels E_n = hbar omega (n + 1/2), n = 0, 1, 2, ... ."""
    n = np.asarray(n)
    if np.any(n < 0):
        raise ValueError("quantum number n must be >= 0")
    return hbar * omega * (n + 0.5)


# ── tunnelling / scattering ─────────────────────────────────────────
def transmission(x, V, E, hbar=1.0, m=1.0):
    """Transmission coefficient T(E) through a 1-D barrier V(x).

    Integrates the Schrodinger ODE from the right lead (pure outgoing wave)
    leftward with RK4, then reads off the incident amplitude.  Requires flat,
    equal leads: V[0] == V[-1], and E above the lead floor (a propagating
    scattering state).
    """
    x = np.asarray(x, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.shape != x.shape:
        raise ValueError(f"V must match x shape; got {V.shape} vs {x.shape}")
    if not np.isclose(V[0], V[-1]):
        raise ValueError("leads must be equal: V[0] != V[-1]")
    if E <= V[0]:
        raise ValueError(f"E={E} must exceed the lead floor V_lead={V[0]:g}")
    k = np.sqrt(2 * m * (E - V[0])) / hbar
    fac = 2 * m / hbar**2

    def Vf(xx):
        return np.interp(xx, x, V)

    def deriv(xx, y):
        psi, phi = y
        return np.array([phi, fac * (Vf(xx) - E) * psi])

    # right-lead boundary: psi = e^{ikx}, transmitted amplitude set to 1
    y = np.array([np.exp(1j * k * x[-1]), 1j * k * np.exp(1j * k * x[-1])],
                 dtype=complex)
    xs = x[::-1]
    for i in range(xs.size - 1):
        h = xs[i + 1] - xs[i]                 # negative step (integrating left)
        xi = xs[i]
        k1 = deriv(xi, y)
        k2 = deriv(xi + 0.5 * h, y + 0.5 * h * k1)
        k3 = deriv(xi + 0.5 * h, y + 0.5 * h * k2)
        k4 = deriv(xi + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    psi0, phi0 = y
    # left lead: psi = A e^{ikx} + B e^{-ikx}; extract |A|
    A = 0.5 * (psi0 + phi0 / (1j * k)) * np.exp(-1j * k * x[0])
    return float(1.0 / np.abs(A)**2)         # T = |t/A|^2 with t = 1, equal leads


def rectangular_barrier_T(E, V0, a, hbar=1.0, m=1.0):
    """Analytic T(E) for a rectangular barrier of height V0, width a."""
    if E <= 0 or V0 <= 0 or a <= 0:
        raise ValueError("require E, V0, a > 0")
    if E < V0:
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar
        return 1.0 / (1 + (V0**2 * np.sinh(kappa * a)**2) / (4 * E * (V0 - E)))
    if E > V0:
        k2 = np.sqrt(2 * m * (E - V0)) / hbar
        return 1.0 / (1 + (V0**2 * np.sin(k2 * a)**2) / (4 * E * (E - V0)))
    return 1.0 / (1 + (m * V0 * a**2) / (2 * hbar**2))   # E == V0 limit


# ── harmonic oscillator ladder ──────────────────────────────────────
def ladder_operators(n_max):
    """Truncated annihilation/creation matrices a, a_dag on {|0>,...,|n_max>}."""
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    diag = np.sqrt(np.arange(1, n_max + 1))
    a = np.diag(diag, 1)
    return a, a.conj().T


def coherent_state(alpha, n_max):
    """Coherent state |alpha> = e^{-|alpha|^2/2} sum alpha^n/sqrt(n!) |n>."""
    n = np.arange(n_max + 1)
    logc = -0.5 * np.abs(alpha)**2 + n * np.log(alpha + 0j) - 0.5 * _lgamma(n + 1)
    return np.exp(logc)


def _lgamma(z):
    from math import lgamma
    return np.array([lgamma(v) for v in np.atleast_1d(z)]).reshape(np.shape(z))


# ── two-level system (Rabi) ─────────────────────────────────────────
def rabi_evolution(t, Omega, delta=0.0):
    """Excited-state population of a driven two-level system, start in ground.

    Omega = Rabi frequency, delta = detuning.  P_e(t) =
    Omega^2/(Omega^2+delta^2) sin^2( sqrt(Omega^2+delta^2) t / 2 ).
    """
    t = np.asarray(t, dtype=float)
    Omega_gen = np.sqrt(Omega**2 + delta**2)
    if Omega_gen == 0:
        return np.zeros_like(t)
    return (Omega**2 / Omega_gen**2) * np.sin(Omega_gen * t / 2)**2


# ── split-operator (= split-step Fourier) time evolution ────────────
def split_step(psi0, x, V, dt, steps, hbar=1.0, m=1.0, store_every=1):
    """Symmetric split-operator propagation of the time-dependent Schrodinger eq.

    psi(t+dt) = e^{-iV dt/2hbar} F^{-1}[ e^{-i hbar k^2 dt/2m} F[ e^{-iV dt/2hbar} psi ] ].

    The half-potential / full-kinetic / half-potential ordering is the exact
    same Strang splitting used by the split-step Fourier method for the
    nonlinear Schrodinger equation in fibre optics (dispersion <-> kinetic
    term, nonlinearity <-> potential).  Returns the stored frames as a
    (n_frames, N) complex array.
    """
    psi0 = np.asarray(psi0, dtype=complex)
    x = np.asarray(x, dtype=float)
    V = np.asarray(V, dtype=float)
    if psi0.shape != x.shape or V.shape != x.shape:
        raise ValueError("psi0, x, V must share the same 1-D shape")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    N = x.size
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    half_V = np.exp(-1j * V * dt / (2 * hbar))
    full_K = np.exp(-1j * hbar * k**2 * dt / (2 * m))
    psi = psi0.copy()
    frames = [psi.copy()]
    for s in range(1, steps + 1):
        psi = half_V * psi
        psi = np.fft.ifft(full_K * np.fft.fft(psi))
        psi = half_V * psi
        if s % store_every == 0:
            frames.append(psi.copy())
    return np.array(frames)


def gaussian_packet(x, x0, k0, sigma):
    """Normalised Gaussian wavepacket centred at x0 with mean momentum k0."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)
    dx = x[1] - x[0]
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)


# ── the hydrogen atom: levels, orbitals, spin, and photons ──────────
RYDBERG_EV = 13.605693      # ground-state binding energy of hydrogen, |E_1|


def hydrogen_energy(n, Z=1):
    """Bohr level energy E_n = -Z^2 * 13.6 eV / n^2."""
    if n < 1:
        raise ValueError("principal quantum number n must be >= 1")
    return -Z**2 * RYDBERG_EV / n**2


def hydrogen_radial(n, l):
    """Symbolic radial wavefunction R_{nl}(r) (Bohr radii), via sympy.physics.hydrogen."""
    import sympy as sp
    from sympy.physics.hydrogen import R_nl
    if not (0 <= l < n):
        raise ValueError(f"need 0 <= l < n; got n={n}, l={l}")
    r = sp.Symbol("r", positive=True)
    return sp.simplify(R_nl(n, l, r)), r


def transition_wavelength(n_hi, n_lo, Z=1):
    """Rydberg formula: wavelength (nm) of the photon emitted in n_hi -> n_lo.

    1/lambda = Z^2 R_inf (1/n_lo^2 - 1/n_hi^2). The level transition IS the
    photon -- this is where quantum levels become photonics.
    """
    if n_hi <= n_lo:
        raise ValueError("n_hi must exceed n_lo for emission")
    R_inf_per_nm = 1.0973731568e-2          # Rydberg constant in 1/nm
    inv_lambda = Z**2 * R_inf_per_nm * (1 / n_lo**2 - 1 / n_hi**2)
    return 1.0 / inv_lambda


def selection_rule_allowed(l1, l2, dm=0):
    """Electric-dipole selection rules: allowed iff |dl| = 1 and |dm| <= 1."""
    return abs(l1 - l2) == 1 and abs(dm) <= 1


def real_orbital_angular(orbital, theta, phi):
    """Real angular factor |Y| of the named orbital, for surface rendering.

    Returns the (signed) real spherical-harmonic angular function evaluated on
    theta, phi arrays. Supports 's', 'pz','px','py', 'dz2','dxz','dx2y2'.
    """
    t, p = np.asarray(theta), np.asarray(phi)
    table = {
        "s":     lambda: 0.5 / np.sqrt(np.pi) * np.ones_like(t),
        "pz":    lambda: np.sqrt(3 / (4 * np.pi)) * np.cos(t),
        "px":    lambda: np.sqrt(3 / (4 * np.pi)) * np.sin(t) * np.cos(p),
        "py":    lambda: np.sqrt(3 / (4 * np.pi)) * np.sin(t) * np.sin(p),
        "dz2":   lambda: np.sqrt(5 / (16 * np.pi)) * (3 * np.cos(t)**2 - 1),
        "dxz":   lambda: np.sqrt(15 / (4 * np.pi)) * np.sin(t) * np.cos(t) * np.cos(p),
        "dx2y2": lambda: np.sqrt(15 / (16 * np.pi)) * np.sin(t)**2 * np.cos(2 * p),
    }
    if orbital not in table:
        raise ValueError(f"orbital must be one of {sorted(table)}, got {orbital!r}")
    return table[orbital]()


def spin_state(theta, phi=0.0):
    """Spin-1/2 spinor pointing along direction (theta, phi) on the Bloch sphere:
    |chi> = (cos(theta/2), e^{i phi} sin(theta/2)). theta=0 is spin-up."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


# ── the statistics under QM: expectation, spread, Heisenberg (hbar=1) ──
# Born's rule makes |psi(x)|^2 a probability density, so a wavefunction is a
# statistical object: a mean position <x>, a spread sigma_x, a mean momentum <p>,
# obeying the uncertainty principle sigma_x sigma_p >= 1/2. This is the
# "statistics class that comes before Griffiths Ch.1 QM".
def expectation_x(psi, x):
    """<x> = integral x |psi|^2 dx (probability-weighted mean position)."""
    x = np.asarray(x, dtype=float)
    p = np.abs(psi)**2
    return float(np.trapezoid(x * p, x) / np.trapezoid(p, x))


def sigma_x(psi, x):
    """Position spread sqrt(<x^2> - <x>^2) -- the standard deviation of position."""
    x = np.asarray(x, dtype=float)
    p = np.abs(psi)**2
    norm = np.trapezoid(p, x)
    mean = np.trapezoid(x * p, x) / norm
    x2 = np.trapezoid(x**2 * p, x) / norm
    return float(np.sqrt(max(x2 - mean**2, 0.0)))


def expectation_p(psi, x):
    """<p> = integral psi* (-i d/dx) psi dx  (hbar=1); real for any state."""
    x = np.asarray(x, dtype=float)
    dpsi = np.gradient(np.asarray(psi, dtype=complex), x)
    num = np.trapezoid(np.conj(psi) * (-1j * dpsi), x)
    return float(np.real(num / np.trapezoid(np.abs(psi)**2, x)))


def sigma_p(psi, x):
    """Momentum spread sqrt(<p^2> - <p>^2); <p^2> = integral |dpsi/dx|^2 dx."""
    x = np.asarray(x, dtype=float)
    dpsi = np.gradient(np.asarray(psi, dtype=complex), x)
    norm = np.trapezoid(np.abs(psi)**2, x)
    p2 = np.real(np.trapezoid(np.abs(dpsi)**2, x) / norm)
    return float(np.sqrt(max(p2 - expectation_p(psi, x)**2, 0.0)))


def uncertainty_product(psi, x):
    """sigma_x * sigma_p. Heisenberg: always >= 1/2 (hbar/2), with equality only
    for a Gaussian -- the minimum-uncertainty (most 'classical') state."""
    return sigma_x(psi, x) * sigma_p(psi, x)
