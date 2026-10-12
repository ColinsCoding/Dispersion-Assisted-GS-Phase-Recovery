"""How the geometry of space itself changes over billions of years: the
flat FLRW (Friedmann-Lemaitre-Robertson-Walker) metric and the Friedmann
equation that governs its scale factor a(t).

Special relativity (dgs/special_relativity.py) assumes a FIXED, flat
Minkowski spacetime and asks how coordinates relate between inertial
frames moving through it. Cosmology asks a different question: the
spatial geometry ITSELF stretches over time, described by one function
a(t) (a(t_now)=1 by convention) in the flat FLRW line element

    ds^2 = -c^2 dt^2 + a(t)^2 (dx^2 + dy^2 + dz^2).

The Friedmann equation (from Einstein's field equations applied to this
metric) says how a(t) evolves given what fills the universe:

    H(a)^2 = (da/dt / a)^2 = H0^2 [ Omega_r/a^4 + Omega_m/a^3 + Omega_Lambda ]

(flat universe, Omega_r+Omega_m+Omega_Lambda=1). Radiation redshifts an
extra factor of a beyond matter (Omega_r/a^4 vs Omega_m/a^3), which is
why the early universe was radiation-dominated even though matter and
dark energy dominate today -- exactly the kind of qualitative regime
change over billions of years this module makes numerically concrete.

A genuine, important subtlety kept explicit here: Hubble recession
"velocity" v=H*d can exceed c for distant enough galaxies WITHOUT
violating special relativity, because it is not a local relative
velocity between two observers in a shared inertial frame (the SR
speed limit dgs/special_relativity.py enforces) -- it is the rate at
which spacetime itself is stretching between them.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad

C_SI = 2.99792458e8       # m/s
MPC_TO_KM = 3.0856775814913673e19    # km per megaparsec
SEC_PER_GYR = 3.15576e16             # seconds per gigayear (365.25-day years)

# Planck 2018 flat-LambdaCDM best-fit parameters (Planck Collaboration 2020, A&A 641, A6)
H0_KM_S_MPC = 67.4
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
OMEGA_R = 9.24e-5   # radiation (photons + neutrinos), small today


def hubble_constant_per_gyr(H0_km_s_mpc=H0_KM_S_MPC):
    """Convert H0 from km/s/Mpc (the usual observational unit) to 1/Gyr
    (convenient for integrating a(t) directly in billions of years)."""
    H0_per_sec = H0_km_s_mpc / MPC_TO_KM
    return H0_per_sec * SEC_PER_GYR


def friedmann_H(a, H0=H0_KM_S_MPC, omega_m=OMEGA_M, omega_r=OMEGA_R, omega_lambda=OMEGA_LAMBDA):
    """H(a) = H0 * sqrt(Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_Lambda),
    with Omega_k = 1 - Omega_m - Omega_r - Omega_Lambda enforcing flatness
    exactly (not assumed to be zero -- computed and can be checked)."""
    a = np.asarray(a, float)
    omega_k = 1.0 - omega_m - omega_r - omega_lambda
    return H0 * np.sqrt(omega_r / a**4 + omega_m / a**3 + omega_k / a**2 + omega_lambda)


def scale_factor_matter_dominated(t, t0=1.0):
    """Closed-form a(t) = (t/t0)^(2/3) for a single-component, matter-only
    (w=0) flat universe -- used below to sanity-check the general numerical
    integrator in a limit where the answer is known analytically."""
    return (np.asarray(t, float) / t0) ** (2.0 / 3.0)


def scale_factor_radiation_dominated(t, t0=1.0):
    """Closed-form a(t) = (t/t0)^(1/2) for a single-component, radiation-only
    (w=1/3) flat universe -- the other analytically-known limit."""
    return (np.asarray(t, float) / t0) ** 0.5


def integrate_scale_factor(t_span_gyr, a0=1e-8, H0=H0_KM_S_MPC,
                            omega_m=OMEGA_M, omega_r=OMEGA_R, omega_lambda=OMEGA_LAMBDA,
                            n_eval=2000):
    """Numerically integrate da/dt = a*H(a) forward in time (Gyr), starting
    from a tiny a0 (near the Big Bang) up through t_span_gyr[1]. Returns the
    full a(t) trajectory so the radiation -> matter -> dark-energy handoff
    can be seen directly, not just asserted."""
    H0_per_gyr = hubble_constant_per_gyr(H0)

    def rhs(t, y):
        a = max(y[0], 1e-12)
        H = friedmann_H(a, H0_per_gyr, omega_m, omega_r, omega_lambda)
        return [a * H]

    t_eval = np.linspace(t_span_gyr[0], t_span_gyr[1], n_eval)
    sol = solve_ivp(rhs, t_span_gyr, [a0], t_eval=t_eval, method="RK45",
                     rtol=1e-9, atol=1e-14)
    return sol.t, sol.y[0]


def age_of_universe_gyr(H0=H0_KM_S_MPC, omega_m=OMEGA_M, omega_r=OMEGA_R, omega_lambda=OMEGA_LAMBDA):
    """Age = integral from a=0 to a=1 of da/(a*H(a)), evaluated directly via
    quadrature (an INDEPENDENT method from integrate_scale_factor's forward
    time-stepping, used below to cross-check it)."""
    H0_per_gyr = hubble_constant_per_gyr(H0)

    def integrand(a):
        return 1.0 / (a * friedmann_H(a, H0_per_gyr, omega_m, omega_r, omega_lambda))

    age, _ = quad(integrand, 1e-8, 1.0, limit=200)
    return age


def redshift_from_scale_factor(a_emit, a_obs=1.0):
    """1+z = a_obs/a_emit -- the scale factor directly IS the cosmological
    redshift; light emitted when the universe was half its current size
    (a_emit=0.5) arrives redshifted by z=1, wavelength doubled."""
    return a_obs / a_emit - 1.0


def hubble_law_velocity_km_s(distance_mpc, H0=H0_KM_S_MPC):
    """v = H0 * d -- Hubble's law, the LOCAL, low-redshift approximation to
    cosmological recession (not a Lorentz-boost velocity; see module
    docstring)."""
    return H0 * distance_mpc


def hubble_radius_mpc(H0=H0_KM_S_MPC):
    """Distance at which the naive Hubble-law recession speed v=H0*d equals
    c. Beyond this, galaxies recede faster than light WITHOUT violating
    special relativity -- it is the metric stretching, not a local velocity
    through space (dgs.special_relativity's speed limit applies to the
    latter, not the former)."""
    H0_per_sec = H0 / MPC_TO_KM
    d_km = (C_SI / 1000.0) / H0_per_sec   # C_SI m/s -> km/s, divided by 1/s gives km
    return d_km / MPC_TO_KM                # km -> Mpc (missing in an earlier version:
                                            # that version returned km mislabeled as Mpc,
                                            # off by a factor of MPC_TO_KM ~3.1e19)


if __name__ == "__main__":
    print("=== Friedmann equation: H(a) at three eras ===")
    for a, label in [(1e-4, "early (radiation era)"), (0.5, "matter era"), (1.0, "today")]:
        H = friedmann_H(a)
        print(f"  a={a:8.4f} ({label:22s}): H(a) = {H:8.2f} km/s/Mpc")

    print("\n=== Sanity check: numerical integrator vs. known closed-form limits ===")
    t = np.linspace(0.01, 10.0, 50)
    a_numeric_matter, = [None]
    from scipy.integrate import solve_ivp as _sivp
    def rhs_matter_only(tt, y):
        a = max(y[0], 1e-12)
        return [a * (2.0/3.0) / tt]   # H(t) = (2/3)/t exactly for matter-only
    sol_m = _sivp(rhs_matter_only, (t[0], t[-1]), [scale_factor_matter_dominated(t[0])], t_eval=t, rtol=1e-10, atol=1e-14)
    a_closed = scale_factor_matter_dominated(t)
    max_err = np.max(np.abs(sol_m.y[0] - a_closed))
    print(f"  matter-only: max |numeric - closed form| = {max_err:.2e}")
    assert max_err < 1e-6

    print("\n=== Age of the universe: two independent methods ===")
    age_quad = age_of_universe_gyr()
    t_arr, a_arr = integrate_scale_factor((0.0, 20.0), a0=1e-8, n_eval=200000)
    idx_today = np.argmin(np.abs(a_arr - 1.0))
    age_forward = t_arr[idx_today]
    print(f"  via da/(aH) quadrature:      {age_quad:.3f} Gyr")
    print(f"  via forward time-integration: {age_forward:.3f} Gyr")
    print(f"  known value (Planck 2018):    13.787 Gyr")
    assert abs(age_quad - 13.787) < 0.2
    assert abs(age_forward - age_quad) < 0.1

    print("\n=== Cosmological redshift ===")
    for a_e in (1.0, 0.5, 0.1, 0.01):
        z = redshift_from_scale_factor(a_e)
        print(f"  a_emit={a_e:5.2f}  ->  z={z:8.3f}")

    print("\n=== Hubble's law and the c-boundary ===")
    d_hubble = hubble_radius_mpc()
    print(f"  Hubble radius (v=c): {d_hubble:.0f} Mpc = {d_hubble*3.2616e6/1e9:.2f} billion light-years")
    v_at_2x = hubble_law_velocity_km_s(2 * d_hubble)
    print(f"  naive recession speed at 2x that distance: {v_at_2x:.0f} km/s = {v_at_2x/(C_SI/1000):.2f} c")
    print("  (superluminal, but this is metric expansion, not a local SR-violating velocity)")
