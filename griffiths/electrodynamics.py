"""Electrodynamics -- Griffiths Ch. 7-9, wired straight into this repo.

The whole dispersion-GS receiver rests on one operator, H(f) = exp(i pi D f^2):
a phase that is *quadratic in frequency*. This module shows where that comes
from -- it is electrodynamics, end to end:

    Maxwell's equations            (the four laws)
      -> 1-D wave equation         (curl-curl in a medium)
      -> plane wave  e^{i(kz-wt)}  (the complex-exponential field)
      -> dispersion relation k(w)  (how the medium delays each colour)
      -> group-velocity dispersion (Taylor: the quadratic term beta_2)
      -> H(f) = exp(i pi D f^2)    (exactly disperse() in this repo)

Plus the supporting cast Griffiths gives you: Ohm's law in conductivity-tensor
form J = sigma . E (anisotropic media), the wave impedance Z = sqrt(mu/eps)
linking the electric and magnetic components, the Poynting flux, and optical
power expressed in decibels. SymPy for the analytics, NumPy for the numeric
bridge to disperse(). Civilian optical metrology / education.
"""

import numpy as np
import sympy as sp

# physical-ish symbols (positive so sqrt simplifies cleanly)
mu, eps, mu0, eps0, c = sp.symbols("mu epsilon mu_0 epsilon_0 c", positive=True)
omega, k, z, t = sp.symbols("omega k z t", real=True)


# ── 1. Maxwell's equations (source of everything) ────────────────────
def maxwell_equations(medium=True):
    """The four Maxwell equations as readable sympy Eq objects.

    Uses abstract div/curl placeholders so the *structure* is the lesson:
    two divergence laws (sources) + two curl laws (dynamics). In a linear
    medium D = eps E and H = B / mu. Returns a dict name -> Eq.
    """
    rho = sp.Symbol("rho")
    divE, divB, curlE, curlB = sp.symbols("div_E div_B curl_E curl_B")
    dB_dt, dE_dt = sp.symbols("dB/dt dE/dt")
    J = sp.Symbol("J")
    e = eps if medium else eps0
    m = mu if medium else mu0
    return {
        "Gauss_E":   sp.Eq(divE, rho / e),               # charges make diverging E
        "Gauss_B":   sp.Eq(divB, 0),                     # no magnetic monopoles
        "Faraday":   sp.Eq(curlE, -dB_dt),               # changing B curls E
        "Ampere":    sp.Eq(curlB, m * J + m * e * dE_dt),  # current + displacement
    }


# ── 2. Maxwell -> 1-D wave equation (curl-curl) ──────────────────────
def wave_equation_1d():
    """Derive d^2E/dz^2 = mu*eps * d^2E/dt^2 for a 1-D wave in a medium.

    Take E = E(z,t) x-hat, B = B(z,t) y-hat, no free charge/current. The two
    curl equations reduce to coupled first-order PDEs; cross-differentiating
    eliminates B and gives the wave equation. Returns (wave_eq, steps_dict).
    """
    E = sp.Function("E")(z, t)
    B = sp.Function("B")(z, t)
    # Faraday  (curl E)_y = -dB/dt  ->  dE/dz = -dB/dt
    faraday = sp.Eq(sp.diff(E, z), -sp.diff(B, t))
    # Ampere   (curl B)_y = mu eps dE/dt  ->  -dB/dz = mu eps dE/dt
    ampere = sp.Eq(-sp.diff(B, z), mu * eps * sp.diff(E, t))
    # d/dz of Faraday, then substitute dB/dz from Ampere
    d2E = sp.diff(faraday.lhs, z)                      # d2E/dz2
    rhs = -sp.diff(sp.diff(B, z), t)                   # -d/dt(dB/dz)
    rhs = rhs.subs(sp.diff(B, z), -mu * eps * sp.diff(E, t))  # Ampere: dB/dz = -mu eps dE/dt
    wave = sp.Eq(d2E, sp.simplify(rhs))
    return wave, {"faraday": faraday, "ampere": ampere}


# ── 3. plane wave + dispersion relation ──────────────────────────────
def plane_wave_dispersion():
    """Substitute the complex plane wave into the wave equation.

    E(z,t) = exp(i(k z - omega t)) solves it iff  k^2 = mu*eps*omega^2.
    Returns (dispersion_relation_Eq, k_of_omega, refractive_index n).
    """
    E = sp.exp(sp.I * (k * z - omega * t))
    lhs = sp.diff(E, z, 2)
    rhs = mu * eps * sp.diff(E, t, 2)
    rel = sp.simplify((lhs - rhs) / E)                 # = 0 condition
    disp = sp.Eq(k**2, mu * eps * omega**2)            # the dispersion relation
    k_w = omega * sp.sqrt(mu * eps)                    # k = omega / v
    n = sp.sqrt(mu * eps / (mu0 * eps0))               # n = c/v
    assert sp.simplify(rel + k**2 - mu * eps * omega**2) == 0
    return disp, k_w, n


def wave_impedance(medium=True):
    """Z = E/H = sqrt(mu/eps): the ratio of the electric to magnetic field.

    The complex-exponential E and B are not independent -- the medium fixes
    their ratio. Vacuum gives Z0 = sqrt(mu0/eps0) ~ 377 ohm.
    """
    return sp.sqrt(mu / eps) if medium else sp.sqrt(mu0 / eps0)


# ── 4. Ohm's law, tensor (anisotropic) form ──────────────────────────
def ohms_law_tensor(sigma=None):
    """J = sigma . E with a 3x3 conductivity tensor (anisotropic medium).

    Pass a 3x3 sympy Matrix; default is the isotropic sigma*I, for which
    J is just sigma*E. Returns (J_vector, sigma_matrix).
    """
    s = sp.Symbol("sigma")
    Sig = sp.eye(3) * s if sigma is None else sp.Matrix(sigma)
    if Sig.shape != (3, 3):
        raise ValueError("sigma must be 3x3")
    Ex, Ey, Ez = sp.symbols("E_x E_y E_z")
    Evec = sp.Matrix([Ex, Ey, Ez])
    return Sig * Evec, Sig


def drude_conductivity():
    """Frequency-dependent conductivity sigma(omega) = sigma0 / (1 + i omega tau).

    The complex sigma is *why* media are dispersive: the electrons cannot
    follow the field instantly, so different colours see different sigma --
    and hence different k(omega). Returns the sympy expression.
    """
    sigma0, tau = sp.symbols("sigma_0 tau", positive=True)
    return sigma0 / (1 + sp.I * omega * tau)


# ── 5. Poynting flux + power in decibels ─────────────────────────────
def time_average_poynting(E0, medium=True):
    """Time-averaged energy flux <S> = E0^2 / (2 Z) of a plane wave (W/m^2).

    The intensity a square-law detector measures. Z is the wave impedance.
    """
    return E0**2 / (2 * wave_impedance(medium))


def to_decibels(power, ref=1.0, kind="power"):
    """Express optical power (or amplitude) in dB relative to ref.

    kind='power'     -> 10*log10(P/ref)   (intensities, |E|^2)
    kind='amplitude' -> 20*log10(A/ref)   (field amplitudes, |E|)
    Accepts scalars or arrays; non-positive values map to -inf.
    """
    p = np.asarray(power, dtype=float)
    factor = 10.0 if kind == "power" else 20.0 if kind == "amplitude" else None
    if factor is None:
        raise ValueError("kind must be 'power' or 'amplitude'")
    with np.errstate(divide="ignore"):
        out = factor * np.log10(np.where(p > 0, p / ref, np.nan))
    return np.where(p > 0, out, -np.inf)


# ── 6. the bridge: GVD  ->  H(f) = exp(i pi D f^2) ───────────────────
def gvd_transfer(f, beta2, L):
    """The medium's quadratic spectral phase over length L (the GVD operator).

    Expand k(omega) about a carrier: the constant and linear terms are a phase
    offset and a group delay (removable). The first term that *reshapes* the
    pulse is the quadratic one, beta_2 = d^2k/domega^2:
        H(omega) = exp( i * (1/2) * beta_2 * L * (2*pi*f)^2 ),   omega = 2*pi*f.
    This is exactly the dispersion operator the GS receiver inverts.
    """
    f = np.asarray(f, dtype=float)
    return np.exp(1j * 0.5 * beta2 * L * (2 * np.pi * f) ** 2)

def dispersion_param_D(beta2, L):
    """Map physical GVD (beta_2, length L) to this repo's normalized D.

    Matching exp(i*pi*D*f^2) to exp(i*(1/2)*beta_2*L*(2*pi*f)^2) gives
        D = 2*pi*beta_2*L,
    so disperse(x, D) and gvd_transfer(f, beta_2, L) are the same operator.
    """
    return 2 * np.pi * beta2 * L


# ── 7. absorption & dispersion: the Lorentz oscillator (Griffiths 9.4.3) ──
def lorentz_epsilon(omega, omega0, gamma, omega_p):
    """Complex relative permittivity of a single-resonance Lorentz medium.

    Bound electrons are damped oscillators driven by the wave:
        eps_r(w) = 1 + w_p^2 / (w0^2 - w^2 - i*gamma*w)
    w0 = resonance, gamma = damping (linewidth), w_p = plasma frequency
    (oscillator strength). The imaginary part is *absorption*, the real part
    is *dispersion* -- two faces of the same resonance.
    """
    w = np.asarray(omega, dtype=float)
    if gamma < 0 or omega_p < 0:
        raise ValueError("gamma and omega_p must be non-negative")
    return 1.0 + omega_p**2 / (omega0**2 - w**2 - 1j * gamma * w)


def complex_index(eps_r):
    """Complex refractive index n~ = n + i*kappa = sqrt(eps_r) (passive: kappa>=0).

    n (real part) sets the phase velocity -> *dispersion*; kappa (imaginary part)
    sets the decay -> *absorption*.
    """
    n = np.sqrt(np.asarray(eps_r, dtype=complex))
    n = np.where(n.imag < 0, -n, n)           # passive-medium branch (kappa >= 0)
    return n


def absorption_coefficient(omega, kappa, c=1.0):
    """Beer-Lambert intensity absorption coefficient alpha = 2*omega*kappa/c.

    Intensity falls as I(z) = I0 exp(-alpha z): the field amplitude decays like
    exp(-omega*kappa*z/c), so the intensity (its square) decays twice as fast.
    """
    return 2.0 * np.asarray(omega, dtype=float) * np.asarray(kappa, dtype=float) / c


def beer_lambert(I0, alpha, z):
    """Transmitted intensity I0 * exp(-alpha * z) through length z."""
    return np.asarray(I0, dtype=float) * np.exp(-np.asarray(alpha, dtype=float) * z)


def is_anomalous(omega, n_real):
    """Boolean mask where dn/domega < 0 -- the *anomalous* dispersion band.

    Normally n rises with frequency (normal dispersion); right across an
    absorption line it briefly *falls* -- anomalous dispersion, the price the
    Kramers-Kronig relations charge for the absorption peak.
    """
    return np.gradient(np.asarray(n_real, dtype=float), np.asarray(omega, dtype=float)) < 0


def _hilbert(g):
    """Hilbert transform H[g] via FFT (analytic signal = g + i H[g])."""
    g = np.asarray(g, dtype=float)
    N = len(g)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(g) * h).imag


def kramers_kronig(chi_imag):
    """Reconstruct the real part of the susceptibility from its imaginary part.

    Absorption (Im chi) and dispersion (Re chi) are not independent -- causality
    (no response before the cause) ties them by a Hilbert transform. Given the
    absorption spectrum on a uniform grid, returns the implied dispersion
    spectrum (up to the discrete-Hilbert sign/edge conventions).
    """
    return -_hilbert(np.asarray(chi_imag, dtype=float))


# ── 8. Ohm + Gauss + continuity -> charge relaxation ────────────────
def charge_relaxation_ode():
    """Combine the two laws (and charge conservation) into one decay equation.

        continuity:  d(rho)/dt + div J = 0
        Ohm:         J = sigma E
        Gauss:       div E = rho / eps
      =>  d(rho)/dt = -(sigma/eps) rho   =>   rho(t) = rho_0 e^{-t/tau},  tau = eps/sigma.

    Free charge in a conductor dissolves exponentially: it flees to the surface
    (Ohm pushes the current, Gauss says the divergence is the charge). Returns
    (ode, solution, tau) symbolically.
    """
    t = sp.Symbol("t", real=True)
    rho = sp.Function("rho")
    eps_, sig, rho0 = sp.symbols("epsilon sigma rho_0", positive=True)
    ode = sp.Eq(rho(t).diff(t), -(sig / eps_) * rho(t))
    sol = sp.dsolve(ode, rho(t), ics={rho(0): rho0})
    return ode, sol, eps_ / sig


def charge_relaxation_time(eps, sigma):
    """tau = eps/sigma: the time for free charge in a conductor to die off e-fold.

    Tiny for metals (~1e-19 s for copper -- so small the simple model breaks down,
    but the point stands: charge relaxes essentially instantly, which is why a
    conductor is an equipotential and why the receiver's metal screens fast).
    """
    if eps <= 0 or sigma <= 0:
        raise ValueError("eps and sigma must be > 0")
    return eps / sigma


def charge_decay(rho0, t, eps, sigma):
    """Free-charge density rho(t) = rho0 * exp(-t / tau), tau = eps/sigma."""
    tau = charge_relaxation_time(eps, sigma)
    return np.asarray(rho0, dtype=float) * np.exp(-np.asarray(t, dtype=float) / tau)


# ── 9. EM waves in conductors: complex k~ and skin depth (Griffiths 9.4.1) ──
_EPS0 = 8.8541878128e-12     # F/m
_MU0 = 4e-7 * np.pi          # H/m


def conductor_wavenumber(omega, sigma, eps=_EPS0, mu=_MU0):
    """Complex wavenumber k~ = k + i*kappa in a conductor (Griffiths 9.125-9.126).

    Once Ohm's law adds the i*mu*sigma*omega term, the dispersion relation is
    k~^2 = mu eps omega^2 + i mu sigma omega, so k~ is complex:
        k     = omega sqrt(mu eps/2) [ sqrt(1+(sigma/eps omega)^2) + 1 ]^(1/2)
        kappa = omega sqrt(mu eps/2) [ sqrt(1+(sigma/eps omega)^2) - 1 ]^(1/2)
    The wave propagates with k and decays as exp(-kappa z). Returns k + i*kappa.
    """
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0) or sigma < 0:
        raise ValueError("omega > 0 and sigma >= 0 required")
    base = omega * np.sqrt(mu * eps / 2.0)
    s = np.sqrt(1.0 + (sigma / (eps * omega))**2)
    k = base * np.sqrt(s + 1.0)
    kappa = base * np.sqrt(s - 1.0)
    return k + 1j * kappa


def skin_depth(omega, sigma, eps=_EPS0, mu=_MU0):
    """Skin depth d = 1/kappa: the distance over which the amplitude falls by 1/e
    (Griffiths 9.128). For a good conductor d ~ sqrt(2/(omega mu sigma))."""
    kappa = conductor_wavenumber(omega, sigma, eps, mu).imag
    if np.any(kappa <= 0):
        raise ValueError("no attenuation (sigma = 0?): skin depth undefined")
    return 1.0 / kappa


if __name__ == "__main__":
    sp.init_printing()
    disp, k_w, n = plane_wave_dispersion()
    print("dispersion relation:", disp)
    print("k(omega) =", k_w, "   n =", n)
    wave, _ = wave_equation_1d()
    print("1-D wave equation:", wave)
