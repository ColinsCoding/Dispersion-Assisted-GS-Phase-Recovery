"""Phase and group velocity of a Bessel-mode field between two conducting
planes -- and the exact identity v_p * v_g = c^2.

dgs.cylindrical_waveguide_resonance already builds the Bessel-function
radial mode profile J_m(k_c*r) for a circular waveguide/cavity (the
Helmholtz equation separates into Bessel's equation in cylindrical
coordinates) and its cutoff/resonant frequencies. This module asks the
next question: how FAST does a wave at frequency f > f_c actually travel
along the guide?

Two different velocities answer two different questions:
  PHASE velocity   v_p = omega/beta            -- how fast a point of
                                                    constant phase moves.
  GROUP velocity   v_g = domega/dbeta           -- how fast energy/a pulse
                                                    envelope moves.

For a hollow waveguide with cutoff f_c, beta(omega) = sqrt((omega/c)^2 -
k_c^2), so
    v_p = c / sqrt(1 - (f_c/f)^2)   >= c   (ALWAYS -- a hint that v_p is
                                             not a physical transport speed)
    v_g = c * sqrt(1 - (f_c/f)^2)   <= c   (energy never outruns light)
and multiplying them:
    v_p * v_g = c^2                 EXACTLY, for every f > f_c.

At cutoff (f -> f_c+): v_p -> infinity (a phase front can sweep across the
guide arbitrarily fast -- no information travels with it) while v_g -> 0
(a pulse launched right at cutoff never actually leaves). Far above cutoff
(f >> f_c) both v_p, v_g -> c: a waveguide mode is really just a plane wave
bouncing off the walls at an angle, and at high enough frequency the angle
of incidence flattens out until the wave travels almost straight down the
guide at c.

"Two planes": dgs.cylindrical_waveguide_resonance.cavity_resonant_frequency
already quantizes the axial direction between two conducting end plates
(separation L, mode index p); this module's velocities describe the field
BEFORE those plates are added (the open waveguide), and
cavity_standing_wave_from_traveling shows how two oppositely-directed
traveling waves at +-beta combine into the standing wave the closed cavity
actually supports.

Group velocity here is computed TWO independent ways -- the closed form
c*sqrt(1-(fc/f)^2), and a numerical finite-difference domega/dbeta from
dgs.cylindrical_waveguide_resonance.waveguide_propagation_constant -- and
cross-checked, so v_p*v_g=c^2 is a verified numerical fact, not just
algebra taken on faith. py-3.13 (scipy available).
"""

from __future__ import annotations
import numpy as np

from dgs.cylindrical_waveguide_resonance import (
    waveguide_cutoff_frequency, waveguide_propagation_constant, C_LIGHT,
)


def phase_velocity(f: float, m: int, n: int, a: float, boundary: str = "TM",
                    c: float = C_LIGHT) -> float:
    """v_p = omega/beta = 2*pi*f / Re(beta). Raises if f <= f_c (beta is
    then purely imaginary -- evanescent, no propagating phase front)."""
    beta = waveguide_propagation_constant(f, m, n, a, boundary, c)
    if beta.real <= 0:
        raise ValueError(f"f={f:.4e} Hz is below cutoff for this mode (beta is evanescent)")
    return 2 * np.pi * f / beta.real


def group_velocity_closed_form(f: float, m: int, n: int, a: float, boundary: str = "TM",
                                c: float = C_LIGHT) -> float:
    """v_g = c*sqrt(1-(f_c/f)^2), the closed-form result for a hollow
    conducting waveguide (derived from beta(omega) = sqrt((omega/c)^2-k_c^2)
    by direct differentiation domega/dbeta)."""
    fc = waveguide_cutoff_frequency(m, n, a, boundary, c)
    if f <= fc:
        raise ValueError(f"f={f:.4e} Hz is below cutoff f_c={fc:.4e} Hz")
    return c * np.sqrt(1 - (fc / f) ** 2)


def group_velocity_numerical(f: float, m: int, n: int, a: float, boundary: str = "TM",
                              c: float = C_LIGHT, df_frac: float = 1e-5) -> float:
    """v_g = domega/dbeta by a central finite difference on
    waveguide_propagation_constant -- an INDEPENDENT numerical check of
    group_velocity_closed_form, not the same formula re-evaluated."""
    df = f * df_frac
    beta_plus = waveguide_propagation_constant(f + df, m, n, a, boundary, c).real
    beta_minus = waveguide_propagation_constant(f - df, m, n, a, boundary, c).real
    domega = 2 * np.pi * (2 * df)
    dbeta = beta_plus - beta_minus
    if dbeta == 0:
        raise ValueError("degenerate finite difference (dbeta=0); increase df_frac")
    return domega / dbeta


def verify_vp_vg_equals_c2(f: float, m: int, n: int, a: float, boundary: str = "TM",
                            c: float = C_LIGHT, tol: float = 1e-4) -> dict:
    """CHECKED, both ways: (1) the closed-form and numerical group
    velocities agree with each other, and (2) v_p * v_g = c^2 to within
    `tol` (relative). Returns a dict of all four numbers plus the two
    checks so a caller can see exactly what was verified, not just a bool."""
    vp = phase_velocity(f, m, n, a, boundary, c)
    vg_closed = group_velocity_closed_form(f, m, n, a, boundary, c)
    vg_numeric = group_velocity_numerical(f, m, n, a, boundary, c)

    vg_agreement = abs(vg_closed - vg_numeric) / vg_closed
    if vg_agreement > tol:
        raise AssertionError(f"closed-form vg={vg_closed:.6e} vs numerical vg={vg_numeric:.6e} "
                              f"disagree by {vg_agreement:.2e} (tol={tol:.2e})")

    product = vp * vg_closed
    product_error = abs(product - c ** 2) / c ** 2
    if product_error > tol:
        raise AssertionError(f"v_p*v_g = {product:.6e} != c^2 = {c**2:.6e} "
                              f"(relative error {product_error:.2e}, tol={tol:.2e})")

    return {
        "f_Hz": f, "v_p_m_s": vp, "v_g_closed_form_m_s": vg_closed,
        "v_g_numerical_m_s": vg_numeric, "v_p_times_v_g": product,
        "c_squared": c ** 2, "vg_agreement_rel_err": vg_agreement,
        "vp_vg_product_rel_err": product_error,
    }


def velocities_near_cutoff(m: int, n: int, a: float, boundary: str = "TM",
                            c: float = C_LIGHT, f_over_fc: np.ndarray | None = None) -> dict:
    """Sweep f/f_c from just above 1 to far above 1: v_p should DIVERGE
    (-> infinity as f -> f_c+) while v_g -> 0 -- and both should converge
    to c as f/f_c -> infinity (a waveguide mode at high enough frequency
    is just a plane wave grazing down the guide)."""
    if f_over_fc is None:
        f_over_fc = np.concatenate([np.linspace(1.001, 1.5, 30), np.linspace(1.5, 20, 30)])
    fc = waveguide_cutoff_frequency(m, n, a, boundary, c)
    f_arr = f_over_fc * fc
    vp_arr = np.array([phase_velocity(f, m, n, a, boundary, c) for f in f_arr])
    vg_arr = np.array([group_velocity_closed_form(f, m, n, a, boundary, c) for f in f_arr])
    return {"f_over_fc": f_over_fc, "f_Hz": f_arr, "v_p": vp_arr, "v_g": vg_arr, "f_c_Hz": fc}


if __name__ == "__main__":
    a = 0.01   # 1 cm radius circular waveguide, TE11 (dominant mode)
    fc = waveguide_cutoff_frequency(1, 1, a, "TE")
    print(f"TE11 in a={a*100:.0f} cm circular guide: f_c = {fc/1e9:.4f} GHz")

    print("\n=== v_p * v_g = c^2, checked two independent ways at several f/f_c ===")
    for ratio in (1.05, 1.2, 2.0, 5.0):
        f = ratio * fc
        result = verify_vp_vg_equals_c2(f, 1, 1, a, "TE")
        print(f"  f/f_c={ratio:5.2f}: v_p={result['v_p_m_s']:.4e} m/s (>c), "
              f"v_g={result['v_g_closed_form_m_s']:.4e} m/s (<c), "
              f"v_p*v_g/c^2={result['v_p_times_v_g']/result['c_squared']:.8f}")

    print("\n=== near cutoff: v_p diverges, v_g -> 0 ===")
    sweep = velocities_near_cutoff(1, 1, a, "TE", f_over_fc=np.array([1.001, 1.01, 1.1, 1.5, 5.0, 20.0]))
    for ratio, vp, vg in zip(sweep["f_over_fc"], sweep["v_p"], sweep["v_g"]):
        print(f"  f/f_c={ratio:7.3f}: v_p={vp:.4e} m/s, v_g={vg:.4e} m/s")

    print("\nboth v_p and v_g -> c far above cutoff: the waveguide mode becomes")
    print("indistinguishable from a plane wave grazing straight down the guide.")
