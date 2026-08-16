"""thz_waveguide_dispersion_relation.py -- a photon in a waveguide obeys
EXACTLY the relativistic dispersion relation of a MASSIVE particle, not
just "something dispersive." This module makes that identity explicit and
applies it to a THz-band radio-over-fiber/waveguide link -- the
"dispersion-assisted" theme this entire repo is organized around, applied
here to a FutureG (6G/THz) radio application (dgs/sbir_portfolio.py).

THE IDENTITY (verified below with SymPy, not asserted): a waveguide's
cutoff frequency omega_c (dgs.cylindrical_waveguide_resonance's TE11
result, k_c=1.8412/a) makes its dispersion relation
    omega^2 = c^2 k^2 + omega_c^2.
The relativistic dispersion relation for a massive particle
(dgs.compton_scattering's E^2=(pc)^2+(mc^2)^2) becomes EXACTLY this same
equation under E=hbar*omega, p=hbar*k, m=hbar*omega_c/c^2 -- a photon
propagating below a waveguide's cutoff behaves as if it acquired an
EFFECTIVE MASS m_eff=hbar*omega_c/c^2, purely from confinement, with no
actual rest mass anywhere.

CONSEQUENCES that follow directly from the SAME algebra as a relativistic
particle: phase velocity v_p=omega/k EXCEEDS c (no causality violation --
no energy/information travels at v_p, exactly as for a relativistic
particle's phase velocity); group velocity v_g=domega/dk stays BELOW c
always; and v_p*v_g=c^2 EXACTLY (the same identity a de Broglie matter
wave satisfies). GVD (group velocity dispersion, beta_2=d^2k/domega^2,
this repo's core physical quantity via dgs.dispersive_fourier) causes a
THz pulse to spread as it propagates -- the actual bandwidth-limiting
mechanism in a dispersion-assisted THz radio link.
"""

from __future__ import annotations
import numpy as np
import sympy as sp

from dgs.cylindrical_waveguide_resonance import waveguide_cutoff_frequency

HBAR = 1.054571817e-34   # J*s
C_LIGHT = 299792458.0   # m/s


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. The identity: waveguide dispersion IS relativistic dispersion ───────

def verify_waveguide_matches_relativistic_dispersion() -> bool:
    """CHECKED: substituting E=hbar*omega, p=hbar*k, m=hbar*omega_c/c^2
    into the relativistic dispersion relation E^2=(pc)^2+(mc^2)^2 produces
    EXACTLY omega^2=c^2 k^2+omega_c^2 -- not an analogy, an algebraic
    identity between dgs.compton_scattering's relation and the waveguide
    dispersion relation dgs.cylindrical_waveguide_resonance's cutoff
    frequency work implies."""
    omega, k, c, omega_c = sp.symbols("omega k c omega_c", positive=True)
    E, p, m, hbar = sp.symbols("E p m hbar", positive=True)

    relativistic_lhs_minus_rhs = E**2 - ((p * c)**2 + (m * c**2)**2)
    substituted = relativistic_lhs_minus_rhs.subs({E: hbar * omega, p: hbar * k, m: hbar * omega_c / c**2})
    substituted_over_hbar2 = sp.simplify(substituted / hbar**2)

    waveguide_lhs_minus_rhs = omega**2 - (c**2 * k**2 + omega_c**2)
    diff = sp.simplify(substituted_over_hbar2 - waveguide_lhs_minus_rhs)
    if diff != 0:
        raise AssertionError(f"substituted relativistic dispersion does not match waveguide dispersion: leftover {diff}")
    return True


def effective_photon_mass(omega_c: float) -> float:
    """m_eff = hbar*omega_c/c^2 -- the EFFECTIVE rest mass a photon
    acquires purely from waveguide confinement (no actual rest mass
    anywhere; this is the SAME role m plays in E^2=(pc)^2+(mc^2)^2)."""
    _validate_positive(omega_c=omega_c)
    return HBAR * omega_c / C_LIGHT**2


# ── 2. Phase velocity, group velocity, and v_p*v_g=c^2 ──────────────────────

def waveguide_wavenumber(omega: float, omega_c: float, c: float = C_LIGHT) -> float:
    """k(omega) = sqrt(omega^2 - omega_c^2)/c, valid for omega > omega_c
    (propagating regime; below cutoff k is imaginary -- evanescent, no
    real wavenumber, handled by waveguide_propagation_constant in
    dgs.cylindrical_waveguide_resonance for that regime)."""
    _validate_positive(omega_c=omega_c, c=c)
    if omega <= omega_c:
        raise ValueError(f"omega={omega} must exceed the cutoff omega_c={omega_c} "
                         f"for a real (propagating) wavenumber")
    return np.sqrt(omega**2 - omega_c**2) / c


def phase_velocity(omega: float, omega_c: float, c: float = C_LIGHT) -> float:
    """v_p = omega/k = c*omega/sqrt(omega^2-omega_c^2) -- EXCEEDS c, same
    as a relativistic particle's phase velocity v_p=E/p=c^2/v_particle."""
    k = waveguide_wavenumber(omega, omega_c, c)
    return omega / k


def group_velocity(omega: float, omega_c: float, c: float = C_LIGHT) -> float:
    """v_g = domega/dk = c*sqrt(omega^2-omega_c^2)/omega -- stays below c
    always, the actual signal/energy propagation speed."""
    _validate_positive(omega_c=omega_c, c=c)
    if omega <= omega_c:
        raise ValueError(f"omega={omega} must exceed the cutoff omega_c={omega_c}")
    return c * np.sqrt(omega**2 - omega_c**2) / omega


def verify_phase_group_velocity_product(omega: float, omega_c: float,
                                        c: float = C_LIGHT, rtol: float = 1e-9) -> bool:
    """CHECKED: v_p * v_g = c^2 EXACTLY -- the same identity a de Broglie
    matter wave satisfies (v_phase * v_group = c^2 there too), now shown
    to hold for a confined photon as well, from the identical dispersion-
    relation algebra."""
    v_p = phase_velocity(omega, omega_c, c)
    v_g = group_velocity(omega, omega_c, c)
    product = v_p * v_g
    rel_err = abs(product - c**2) / c**2
    if rel_err > rtol:
        raise AssertionError(f"v_p*v_g = {product}, expected exactly c^2={c**2}, relative error {rel_err:.2e}")
    return True


# ── 3. GVD and THz pulse broadening: the "dispersion-assisted" mechanism ────

def group_velocity_dispersion(omega: float, omega_c: float, c: float = C_LIGHT) -> float:
    """beta_2 = d^2k/domega^2, evaluated analytically from k(omega) --
    this repo's central quantity (dgs.dispersive_fourier's GVD), here
    derived for the waveguide/relativistic dispersion relation instead of
    a fiber's material dispersion."""
    _validate_positive(omega_c=omega_c, c=c)
    if omega <= omega_c:
        raise ValueError(f"omega={omega} must exceed the cutoff omega_c={omega_c}")
    # k(w) = sqrt(w^2-wc^2)/c; dk/dw = w/(c*sqrt(w^2-wc^2));
    # d2k/dw2 = -wc^2 / (c*(w^2-wc^2)^1.5)
    return -omega_c**2 / (c * (omega**2 - omega_c**2)**1.5)


def thz_pulse_broadening(L_m: float, bandwidth_rad_s: float, omega0: float,
                         omega_c: float, c: float = C_LIGHT) -> float:
    """Estimated pulse-width GROWTH (s) after propagating distance L_m,
    from GVD spreading a pulse of angular-frequency bandwidth
    `bandwidth_rad_s` centered at omega0: Delta_t ~ |beta_2| * L * Delta_omega
    -- the same linear-in-(L, bandwidth, beta_2) estimate
    dgs.dispersive_fourier's stretch-factor reasoning uses, applied here
    to a THz waveguide link instead of a fiber."""
    _validate_positive(L_m=L_m, bandwidth_rad_s=bandwidth_rad_s)
    beta2 = group_velocity_dispersion(omega0, omega_c, c)
    return abs(beta2) * L_m * bandwidth_rad_s


# ── 4. Problems 1 & 2 (CSUS deliverable): GVD sign is FIXED, so there is ───
# ──    no zero-dispersion point, and no two-segment "dispersion            ──
# ──    compensation" is possible from this mechanism alone                 ──

def verify_gvd_sign_is_fixed() -> bool:
    """CHECKED (SymPy, not asserted): beta_2(omega) =
    -omega_c^2 / (c*(omega^2-omega_c^2)^1.5) is STRICTLY NEGATIVE for every
    omega > omega_c > 0, c > 0 -- proven here by expanding
    omega^2-omega_c^2 at omega=omega_c+delta (delta>0) into
    delta^2+2*delta*omega_c, which SymPy confirms is strictly positive, so
    the whole expression's numerator (-omega_c^2, strictly negative) over a
    strictly positive denominator is strictly negative for ANY omega_c > 0.

    Problem 2 (zero-dispersion point): solving beta_2(omega)=0 for omega
    directly (not substituting delta) returns no solution -- the numerator
    -omega_c^2 never vanishes for omega_c>0 and never depends on omega, so
    there is no finite omega, at ANY cutoff, where GVD crosses zero. This
    is UNLIKE a fiber, where material and waveguide dispersion can have
    opposite signs and a real zero-dispersion wavelength exists.

    Problem 1 (two-segment dispersion compensation): because beta_2 keeps
    the SAME sign for every valid (omega, omega_c) pair, a two-segment link
    built from this SAME hollow-waveguide mechanism (different radii ->
    different omega_c, but the same sign-fixed formula) can only ADD
    negative contributions along its length -- beta_2_1*L1 + beta_2_2*L2 is
    always more negative than either term alone, never zero, for L1,L2>0.
    Compensating this dispersion needs a DIFFERENT mechanism with opposite-
    sign GVD (e.g. material/fiber dispersion), not a second waveguide
    segment of this same type.
    """
    omega, omega_c, c, delta = sp.symbols("omega omega_c c delta", positive=True)
    beta2 = -omega_c**2 / (c * (omega**2 - omega_c**2)**sp.Rational(3, 2))

    # Problem 2: no finite omega solves beta_2(omega) = 0.
    zero_dispersion_solutions = sp.solve(sp.Eq(beta2, 0), omega)
    if zero_dispersion_solutions:
        raise AssertionError(f"expected NO finite zero-dispersion point, "
                             f"SymPy found {zero_dispersion_solutions}")

    # Problem 1: beta_2 is strictly negative for every omega=omega_c+delta,
    # delta>0 -- so it can never flip sign to cancel a same-type segment.
    base_expanded = sp.expand((omega**2 - omega_c**2).subs(omega, omega_c + delta))
    if not base_expanded.is_positive:
        raise AssertionError(f"expected omega^2-omega_c^2 strictly positive "
                             f"for omega=omega_c+delta, delta>0, got {base_expanded}")
    beta2_rebuilt = -omega_c**2 / (c * base_expanded**sp.Rational(3, 2))
    if not beta2_rebuilt.is_negative:
        raise AssertionError(f"expected beta_2 strictly negative for all "
                             f"omega>omega_c>0, got {beta2_rebuilt}")

    # As omega -> infinity, beta_2 -> 0 but never REACHES it at finite omega.
    limit_at_infinity = sp.limit(beta2, omega, sp.oo)
    if limit_at_infinity != 0:
        raise AssertionError(f"expected beta_2 -> 0 as omega -> infinity, got {limit_at_infinity}")

    return True


# ── 5. Problem 3 (CSUS deliverable): rank TE11/TM01/TE21 by how much each ──
# ──    disperses a shared-carrier THz pulse -- confirms TE11 (already the ──
# ──    dominant mode by cutoff, dgs.cylindrical_waveguide_resonance) is    ──
# ──    ALSO the lowest-dispersion choice, not just the lowest-loss one    ──

def rank_modes_by_dispersion(a: float, L_m: float = 1.0,
                             bandwidth_rad_s: float = 2 * np.pi * 10e9,
                             omega_headroom: float = 1.5) -> dict:
    """For a circular waveguide of radius `a`, compute the cutoff, effective
    photon mass, and pulse-broadening (dgs.thz_pulse_broadening) of TE11,
    TM01, and TE21 -- the same three candidate modes
    dgs.cylindrical_waveguide_resonance.dominant_mode_cutoff compares by
    cutoff alone -- all driven at a SHARED carrier omega0 =
    `omega_headroom` * (highest of the three cutoffs), so every mode is
    safely in its propagating regime. Returns modes ranked from least to
    most dispersion, answering "which mode is the best choice for a
    dispersion-sensitive 6G link" quantitatively rather than by cutoff
    alone."""
    _validate_positive(a=a, L_m=L_m, bandwidth_rad_s=bandwidth_rad_s)
    if omega_headroom <= 1.0:
        raise ValueError(f"omega_headroom must be > 1 (must exceed every cutoff), got {omega_headroom}")

    mode_defs = {"TE11": (1, 1, "TE"), "TM01": (0, 1, "TM"), "TE21": (2, 1, "TE")}
    cutoffs = {name: 2 * np.pi * waveguide_cutoff_frequency(m, n, a, boundary)
              for name, (m, n, boundary) in mode_defs.items()}
    omega0 = omega_headroom * max(cutoffs.values())

    results = {}
    for name, omega_c in cutoffs.items():
        results[name] = {
            "cutoff_THz": omega_c / (2 * np.pi) / 1e12,
            "m_eff_kg": effective_photon_mass(omega_c),
            "broadening_ps": thz_pulse_broadening(L_m, bandwidth_rad_s, omega0, omega_c) * 1e12,
        }
    ranked = sorted(results, key=lambda name: results[name]["broadening_ps"])
    return {"omega0_THz": omega0 / (2 * np.pi) / 1e12, "modes": results, "ranked_best_to_worst": ranked}


# ── 6. Problem 4 (CSUS deliverable): does thermal expansion of the guide ───
# ──    meaningfully shift the pulse-broadening prediction?                ──

def thermal_broadening_shift(a0: float, omega0: float, L_m: float = 1.0,
                             bandwidth_rad_s: float = 2 * np.pi * 10e9,
                             alpha_per_K: float = 17e-6, delta_T_K: float = 60.0,
                             m: int = 1, n: int = 1, boundary: str = "TE") -> dict:
    """Real waveguide dimensions drift with temperature: a(T) = a0*(1 +
    alpha*Delta_T) (linear thermal expansion; default alpha=17e-6 /K is
    copper's coefficient, a standard THz waveguide wall material). This
    shifts omega_c (since omega_c ~ 1/a), which shifts GVD and therefore
    the predicted pulse broadening. Computes the FRACTIONAL change in both
    over a realistic delta_T_K=60 K swing (e.g. an outdoor -20C to +40C
    operating range) at a fixed operating omega0 -- answering whether
    section 5's pulse-broadening prediction needs a temperature correction
    for a real deployment, or whether the shift is negligible."""
    _validate_positive(a0=a0, omega0=omega0, L_m=L_m, bandwidth_rad_s=bandwidth_rad_s)
    a_hot = a0 * (1 + alpha_per_K * delta_T_K)
    omega_c_nominal = 2 * np.pi * waveguide_cutoff_frequency(m, n, a0, boundary)
    omega_c_hot = 2 * np.pi * waveguide_cutoff_frequency(m, n, a_hot, boundary)
    if omega0 <= max(omega_c_nominal, omega_c_hot):
        raise ValueError(f"omega0={omega0} must exceed both the nominal and "
                         f"thermally-shifted cutoffs to stay in the propagating band")

    broadening_nominal = thz_pulse_broadening(L_m, bandwidth_rad_s, omega0, omega_c_nominal)
    broadening_hot = thz_pulse_broadening(L_m, bandwidth_rad_s, omega0, omega_c_hot)
    return {
        "omega_c_frac_shift": (omega_c_hot - omega_c_nominal) / omega_c_nominal,
        "broadening_nominal_ps": broadening_nominal * 1e12,
        "broadening_hot_ps": broadening_hot * 1e12,
        "broadening_frac_shift": (broadening_hot - broadening_nominal) / broadening_nominal,
    }


# ── 7. Problem 5 (CSUS deliverable): is omega^2=c^2k^2+omega_c^2 a property ─
# ──    of ANY hollow waveguide, not just the circular case this module    ──
# ──    started from? Proven in general, then checked concretely for a     ──
# ──    RECTANGULAR guide (Griffiths Ch. 9's own worked example)           ──

def verify_dispersion_relation_is_geometry_independent() -> bool:
    """CHECKED (SymPy): starting from the 3D wave equation and the
    separation ansatz Psi(x,y,z,t) = f(x,y)*exp(i(k*z-omega*t)) -- valid
    for ANY transverse cross-section shape, circular or otherwise -- the
    z,t derivatives alone (computed here) reduce the wave equation to
        [laplacian_t(f)/f] - k^2 + omega^2/c^2 = 0.
    Defining k_c^2 = -laplacian_t(f)/f (the transverse Helmholtz
    eigenvalue -- whatever VALUE the boundary condition picks out, for
    ANY geometry) turns this into omega^2 = c^2*(k^2+k_c^2) purely from
    the z,t algebra -- the geometry only decides what k_c IS, never the
    FORM of the omega-k relation. Solving symbolically for omega confirms
    this."""
    w, k, c, kc, t, z = sp.symbols("omega k c k_c t z", positive=True)
    expo = sp.exp(sp.I * (k * z - w * t))
    d2_dz2_over_expo = sp.simplify(sp.diff(expo, z, 2) / expo)
    d2_dt2_over_expo = sp.simplify(sp.diff(expo, t, 2) / expo)
    if d2_dz2_over_expo != -k**2:
        raise AssertionError(f"expected d^2/dz^2 factor -k^2, got {d2_dz2_over_expo}")
    if d2_dt2_over_expo != -w**2:
        raise AssertionError(f"expected d^2/dt^2 factor -omega^2, got {d2_dt2_over_expo}")

    # wave eq / (f*expo): [lap_t(f)/f] + (d2/dz2 factor) - (1/c^2)*(d2/dt2 factor) = 0
    # substitute lap_t(f)/f = -k_c^2 (the definition of the transverse eigenvalue)
    wave_eq = -kc**2 + (-k**2) - (1 / c**2) * (-w**2)
    solved = sp.solve(sp.Eq(wave_eq, 0), w)
    expected = c * sp.sqrt(k**2 + kc**2)
    if sp.simplify(solved[0] - expected) != 0:
        raise AssertionError(f"expected omega = c*sqrt(k^2+k_c^2), got {solved}")

    # Concrete second example (Griffiths Ch. 9.5.2): a RECTANGULAR guide
    # (dimensions a x b) with f(x,y) = cos(m*pi*x/a)*cos(n*pi*y/b) -- a
    # completely different transverse solution (sines/cosines, not Bessel
    # functions) -- still satisfies laplacian_t(f) = -k_c^2*f with the
    # rectangular-specific k_c^2 = (m*pi/a)^2 + (n*pi/b)^2, confirming the
    # SAME omega^2=c^2(k^2+k_c^2) form applies, just with a different k_c
    # formula than the circular-guide case this module is built around.
    x, y, a, b = sp.symbols("x y a b", positive=True)
    m_idx, n_idx = sp.symbols("m n", positive=True, integer=True)
    f_rect = sp.cos(m_idx * sp.pi * x / a) * sp.cos(n_idx * sp.pi * y / b)
    lap_t_f_rect = sp.diff(f_rect, x, 2) + sp.diff(f_rect, y, 2)
    kc2_rect = (m_idx * sp.pi / a)**2 + (n_idx * sp.pi / b)**2
    residual = sp.simplify(lap_t_f_rect + kc2_rect * f_rect)
    if residual != 0:
        raise AssertionError(f"rectangular guide k_c^2 formula does not satisfy "
                             f"laplacian_t(f) = -k_c^2*f, residual {residual}")

    return True


def rectangular_waveguide_cutoff_frequency(m: int, n: int, a: float, b: float,
                                           c: float = C_LIGHT) -> float:
    """f_c = (c/2)*sqrt((m/a)^2+(n/b)^2) for a rectangular waveguide of
    width a, height b (Griffiths Ch. 9.5.2's own worked example) -- the
    concrete second geometry verify_dispersion_relation_is_geometry_
    independent checks against the circular-guide k_c=j'_{m,n}/a formula
    this module otherwise uses throughout."""
    if m < 0 or n < 0 or (m == 0 and n == 0):
        raise ValueError(f"m,n must be >= 0, not both 0, got m={m}, n={n}")
    _validate_positive(a=a, b=b, c=c)
    k_c = np.pi * np.sqrt((m / a)**2 + (n / b)**2)
    return c * k_c / (2 * np.pi)


if __name__ == "__main__":
    print("=== 1. Waveguide dispersion IS the relativistic dispersion relation ===")
    ok_identity = verify_waveguide_matches_relativistic_dispersion()
    print(f"  E^2=(pc)^2+(mc^2)^2  ->  omega^2=c^2k^2+omega_c^2 under E=hbar*omega, "
          f"p=hbar*k, m=hbar*omega_c/c^2: {ok_identity}")

    # a 0.3 mm radius circular waveguide's TE11 cutoff, reused from
    # dgs.cylindrical_waveguide_resonance (k_c*a=1.8412 for TE11)
    a = 0.3e-3   # m
    k_c_a = 1.8412
    omega_c = C_LIGHT * (k_c_a / a)
    m_eff = effective_photon_mass(omega_c)
    print(f"\n  waveguide radius a={a*1e3:.2f} mm  ->  f_c = {omega_c/(2*np.pi)/1e12:.4f} THz")
    print(f"  effective photon mass m_eff = hbar*omega_c/c^2 = {m_eff:.3e} kg "
          f"({m_eff/9.1093837015e-31:.3e} electron masses)")

    print("\n=== 2. Phase velocity > c, group velocity < c, product = c^2 exactly ===")
    omega_drive = 1.5 * omega_c   # operate at 1.5x cutoff, well into the propagating band
    v_p = phase_velocity(omega_drive, omega_c)
    v_g = group_velocity(omega_drive, omega_c)
    ok_product = verify_phase_group_velocity_product(omega_drive, omega_c)
    print(f"  operating at omega=1.5*omega_c ({omega_drive/(2*np.pi)/1e12:.4f} THz):")
    print(f"  v_phase = {v_p:.6e} m/s  ({v_p/C_LIGHT:.4f}c, exceeds c -- no signal travels this fast)")
    print(f"  v_group = {v_g:.6e} m/s  ({v_g/C_LIGHT:.4f}c, the real signal speed)")
    print(f"  v_phase * v_group = c^2, verified: {ok_product}")

    print("\n=== 3. GVD and THz pulse broadening over a real link length ===")
    L_link = 1.0   # m
    bandwidth = 2 * np.pi * 10e9   # 10 GHz signal bandwidth (a realistic 6G channel)
    beta2 = group_velocity_dispersion(omega_drive, omega_c)
    broadening = thz_pulse_broadening(L_link, bandwidth, omega_drive, omega_c)
    print(f"  GVD beta_2 = {beta2:.4e} s^2/m")
    print(f"  over L={L_link} m, {bandwidth/(2*np.pi)/1e9:.1f} GHz bandwidth: "
          f"pulse broadening ~ {broadening*1e12:.4f} ps")

    print("\nA THz waveguide/photonic link disperses a signal for the EXACT SAME")
    print("algebraic reason a massive relativistic particle has a nontrivial")
    print("dispersion relation -- confinement manufactures an effective mass, and")
    print("that mass is what group-velocity dispersion (this repo's core physics)")
    print("is built from, applied here to a FutureG (6G/THz) radio link.")

    print("\n=== 4. Problems 1 & 2 (CSUS): is there a zero-dispersion point? ===")
    ok_sign = verify_gvd_sign_is_fixed()
    print(f"  beta_2(omega) proven strictly negative for all omega>omega_c>0: {ok_sign}")
    print("  -> NO finite zero-dispersion point exists (unlike fiber's material/")
    print("     waveguide dispersion crossover), and NO two-segment link built from")
    print("     this same mechanism can cancel its own dispersion to zero.")

    print("\n=== 5. Problem 3 (CSUS): rank TE11/TM01/TE21 by dispersion, not just cutoff ===")
    ranking = rank_modes_by_dispersion(a)
    print(f"  shared carrier omega0 = {ranking['omega0_THz']:.4f} THz (1.5x the highest cutoff)")
    for name in ranking["ranked_best_to_worst"]:
        r = ranking["modes"][name]
        print(f"  {name}: f_c={r['cutoff_THz']:.4f} THz  broadening={r['broadening_ps']:.4f} ps")
    print(f"  best (least dispersion): {ranking['ranked_best_to_worst'][0]} -- "
          f"same mode dominant_mode_cutoff already picks by lowest cutoff, "
          f"now ALSO confirmed lowest-dispersion, not just lowest-loss.")

    print("\n=== 6. Problem 4 (CSUS): does thermal expansion matter? ===")
    thermal = thermal_broadening_shift(a, omega_drive)
    print(f"  60 K operating swing, copper wall (alpha=17e-6/K):")
    print(f"  omega_c fractional shift:    {thermal['omega_c_frac_shift']*100:+.4f}%")
    print(f"  broadening: nominal={thermal['broadening_nominal_ps']:.4f} ps, "
          f"hot={thermal['broadening_hot_ps']:.4f} ps "
          f"({thermal['broadening_frac_shift']*100:+.4f}% shift)")
    print("  -> negligible for a realistic outdoor temperature range; no thermal")
    print("     correction needed for section 3's pulse-broadening prediction.")

    print("\n=== 7. Problem 5 (CSUS): is omega^2=c^2k^2+omega_c^2 geometry-independent? ===")
    ok_general = verify_dispersion_relation_is_geometry_independent()
    print(f"  general separation-of-variables proof (any transverse shape): {ok_general}")
    a_rect, b_rect = 22.86e-3, 10.16e-3   # WR-90-style X-band rectangular guide, m
    f_c_rect = rectangular_waveguide_cutoff_frequency(1, 0, a_rect, b_rect)
    print(f"  concrete rectangular TE10 example (a={a_rect*1e3:.3f} mm, b={b_rect*1e3:.3f} mm): "
          f"f_c = {f_c_rect/1e9:.4f} GHz")
    print("  -> the SAME omega^2=c^2(k^2+k_c^2) form holds for circular AND rectangular")
    print("     guides (and, by the general proof, any hollow-guide cross-section);")
    print("     geometry only changes what k_c IS, never the relation's FORM.")
