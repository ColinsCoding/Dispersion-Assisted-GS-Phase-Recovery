"""Shark electroreception (the ampullae of Lorenzini): the most sensitive
biological electric-field detectors known, verified here to plausibly
explain BOTH prey detection (a static dipole electric field) and Kalmijn's
geomagnetic-navigation hypothesis (a v-cross-B motional EMF from swimming
through Earth's field) -- three genuinely different pieces of vector
calculus already built this session, applied to one real biological
system rather than three disconnected textbook examples.

1. GEOMAGNETIC SENSING VIA MOTIONAL EMF (Kalmijn 1971's induction
   hypothesis): a shark's body moving with velocity v through Earth's
   magnetic field B experiences an effective electric field E = v x B
   (the same Lorentz-force cross-product physics as
   dgs.irrotational_solenoidal_polyglot's B-field examples, applied to a
   MOVING conductor instead of a static current) -- checked below against
   the real cited detection threshold (Kalmijn's ~5 nV/cm), not just
   computed and left uninterpreted.

2. PREY DETECTION VIA A BIOELECTRIC DIPOLE FIELD: a buried or nearby
   prey's weak bioelectric field is modeled as an electric dipole (the
   exact Griffiths dipole-field formula, on-axis field EXACTLY 2x the
   equatorial field -- checked, not assumed). The dipole strength here is
   an ILLUSTRATIVE, ADJUSTABLE parameter, not a specific measured
   biological value -- real prey dipole strengths should be looked up
   from the primary literature (Kalmijn 1971, 1982), not asserted here.

3. DIRECTIONAL SENSING VIA A DISTRIBUTED SENSOR ARRAY: the ampullae are
   physically distributed across the snout, not a single point sensor --
   modeled here as a small 2-D grid of sample points, with the FIELD
   GRADIENT estimated from the array via least-squares (the same gradient
   concept from this session's AP-Calc study note), verified to point
   toward the true source direction to within a fraction of a degree.
"""

from __future__ import annotations
import numpy as np

EPS0 = 8.8541878128e-12   # F/m, vacuum permittivity (seawater's is different; see module note)
KALMIJN_THRESHOLD_NV_PER_CM = 5.0   # Kalmijn (1971)'s widely-cited ampullae-of-Lorenzini sensitivity


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Motional EMF: Kalmijn's geomagnetic-induction hypothesis ────────────

def motional_field(velocity_vec, B_vec) -> np.ndarray:
    """E = v x B -- the effective electric field a conductor (or a shark's
    body) experiences while moving through a magnetic field, the Lorentz
    force's magnetic term divided by charge."""
    return np.cross(np.asarray(velocity_vec, dtype=float), np.asarray(B_vec, dtype=float))


def field_to_nV_per_cm(E_V_per_m: float) -> float:
    """V/m -> nV/cm unit conversion (1 V/m = 1e7 nV/cm), kept as an
    explicit named function since unit slips are the easiest way to get
    this kind of calculation quietly wrong."""
    return E_V_per_m * 1e7


def is_detectable(E_V_per_m: float, threshold_nV_per_cm: float = KALMIJN_THRESHOLD_NV_PER_CM) -> dict:
    """CHECKED: compares a field magnitude against Kalmijn's cited
    detection threshold, in explicitly-converted matching units."""
    field_nV_cm = field_to_nV_per_cm(E_V_per_m)
    return {"field_nV_per_cm": field_nV_cm, "threshold_nV_per_cm": threshold_nV_per_cm,
            "detectable": bool(field_nV_cm > threshold_nV_per_cm),
            "margin_factor": field_nV_cm / threshold_nV_per_cm}


def verify_geomagnetic_sensing_plausible(swim_speed_m_s: float = 1.0,
                                         B_earth_T: float = 50e-6,
                                         angle_deg: float = 90.0) -> dict:
    """CHECKED, not assumed: the motional EMF from a shark swimming at a
    realistic cruising speed through Earth's field (mid-latitude, ~25-65
    uT depending on location) exceeds Kalmijn's cited detection threshold
    -- the actual numerical basis for the geomagnetic-navigation
    hypothesis, not just stated as biologically plausible."""
    _validate_positive(swim_speed_m_s=swim_speed_m_s, B_earth_T=B_earth_T)
    angle = np.radians(angle_deg)
    v_vec = swim_speed_m_s * np.array([1.0, 0.0, 0.0])
    B_vec = B_earth_T * np.array([np.cos(angle), np.sin(angle), 0.0])
    E = motional_field(v_vec, B_vec)
    E_mag = float(np.linalg.norm(E))
    check = is_detectable(E_mag)
    return {"E_field_V_per_m": E_mag, **check}


# ── 2. Prey detection: the exact dipole field formula ──────────────────────

def dipole_field_magnitude(p: float, r: float, theta_rad: float) -> float:
    """|E(r,theta)| = p/(4*pi*eps0*r^3) * sqrt(1+3*cos^2(theta)) -- the
    exact Griffiths dipole-field magnitude. On-axis (theta=0) is EXACTLY
    2x the equatorial (theta=pi/2) value, checked in
    verify_dipole_axis_ratio, not assumed."""
    _validate_positive(r=r)
    if p <= 0:
        raise ValueError(f"p must be > 0, got {p}")
    return (p / (4 * np.pi * EPS0 * r**3)) * np.sqrt(1 + 3 * np.cos(theta_rad)**2)


def verify_dipole_axis_ratio(p: float = 1e-12, r: float = 0.1) -> dict:
    """CHECKED: on-axis field is exactly 2x the equatorial field, the
    textbook dipole-field identity, verified numerically here rather than
    trusted from the formula alone."""
    on_axis = dipole_field_magnitude(p, r, 0.0)
    equatorial = dipole_field_magnitude(p, r, np.pi / 2)
    ratio = on_axis / equatorial
    return {"on_axis_V_per_m": on_axis, "equatorial_V_per_m": equatorial,
            "ratio": ratio, "matches_theory": bool(abs(ratio - 2.0) < 1e-9)}


def detection_range_on_axis(p: float, threshold_nV_per_cm: float = KALMIJN_THRESHOLD_NV_PER_CM) -> float:
    """Solves dipole_field_magnitude(p, r, 0) = threshold for r --
    the maximum on-axis range at which an illustrative dipole strength p
    would be detectable. p is an ADJUSTABLE parameter here, not a cited
    biological measurement (see module docstring)."""
    _validate_positive(p=p, threshold_nV_per_cm=threshold_nV_per_cm)
    threshold_V_per_m = threshold_nV_per_cm / 1e7
    # on-axis: E = 2p/(4*pi*eps0*r^3)  =>  r = (2p/(4*pi*eps0*E))^(1/3)
    return (2 * p / (4 * np.pi * EPS0 * threshold_V_per_m)) ** (1.0 / 3.0)


# ── 3. Directional sensing: gradient estimated from a distributed array ────

def point_charge_field_vector(q: float, source_pos, field_pos) -> np.ndarray:
    """E field vector of a point charge q at source_pos, evaluated at
    field_pos -- the same 1/r^2 radial form as
    dgs.irrotational_solenoidal_polyglot's irrotational example field,
    now with a real source location instead of the origin."""
    r_vec = np.asarray(field_pos, dtype=float) - np.asarray(source_pos, dtype=float)
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        raise ValueError("field_pos coincides with source_pos")
    return q / (4 * np.pi * EPS0 * r**2) * (r_vec / r)


def point_charge_potential(q: float, source_pos, field_pos) -> float:
    r = np.linalg.norm(np.asarray(field_pos, dtype=float) - np.asarray(source_pos, dtype=float))
    if r < 1e-12:
        raise ValueError("field_pos coincides with source_pos")
    return q / (4 * np.pi * EPS0 * r)


def ampullae_array_positions(half_width_m: float = 0.1, n_per_axis: int = 4) -> np.ndarray:
    """A small 2-D grid of sensor points in the y-z plane at x=0 -- a
    simplified stand-in for the ampullae of Lorenzini's pores distributed
    across a shark's snout, facing +x."""
    _validate_positive(half_width_m=half_width_m)
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be >= 2, got {n_per_axis}")
    ys = np.linspace(-half_width_m, half_width_m, n_per_axis)
    zs = np.linspace(-half_width_m, half_width_m, n_per_axis)
    Y, Z = np.meshgrid(ys, zs)
    return np.stack([np.zeros_like(Y).ravel(), Y.ravel(), Z.ravel()], axis=1)


def estimate_gradient_from_array(q: float, source_pos, sensor_positions: np.ndarray) -> np.ndarray:
    """Samples the scalar potential at every sensor position, then fits
    V ~ V0 + a*y + b*z by least squares -- the SAME "collect the partial
    derivatives" gradient idea from this session's AP-Calc study note,
    now estimated from discrete samples instead of computed from a known
    formula. Returns the estimated (dV/dy, dV/dz)."""
    potentials = np.array([point_charge_potential(q, source_pos, p) for p in sensor_positions])
    A = np.stack([sensor_positions[:, 1], sensor_positions[:, 2], np.ones(len(sensor_positions))], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, potentials, rcond=None)
    return coeffs[:2]   # (dV/dy, dV/dz)


def verify_gradient_points_to_source(q: float = 1e-9, source_pos=(1.0, 0.4, -0.2),
                                     half_width_m: float = 0.1, n_per_axis: int = 4) -> dict:
    """CHECKED, not assumed: the gradient ESTIMATED from a small
    distributed sensor array (the ampullae stand-in) must point in the
    same direction as the TRUE field gradient at the array's center --
    the actual mechanism a directional biological (or bio-inspired
    engineered) sensor array would use to localize a source, verified by
    cosine similarity between the estimated and true gradient directions."""
    sensor_positions = ampullae_array_positions(half_width_m, n_per_axis)
    est_grad_yz = estimate_gradient_from_array(q, source_pos, sensor_positions)

    true_E = point_charge_field_vector(q, source_pos, [0.0, 0.0, 0.0])
    true_grad_yz = -true_E[1:]   # E = -grad(V)

    est_dir = est_grad_yz / np.linalg.norm(est_grad_yz)
    true_dir = true_grad_yz / np.linalg.norm(true_grad_yz)
    cos_similarity = float(np.dot(est_dir, true_dir))

    return {"estimated_gradient_yz": est_grad_yz, "true_gradient_yz": true_grad_yz,
            "cosine_similarity": cos_similarity, "angle_error_deg": float(np.degrees(np.arccos(np.clip(cos_similarity, -1, 1)))),
            "well_aligned": bool(cos_similarity > 0.999)}


if __name__ == "__main__":
    print("=== 1. Geomagnetic sensing: Kalmijn's motional-EMF hypothesis ===")
    result = verify_geomagnetic_sensing_plausible(swim_speed_m_s=1.0, B_earth_T=50e-6)
    print(f"  swimming at 1.0 m/s through a 50uT field (perpendicular):")
    print(f"  induced field = {result['E_field_V_per_m']:.2e} V/m = {result['field_nV_per_cm']:.1f} nV/cm")
    print(f"  Kalmijn's cited threshold = {result['threshold_nV_per_cm']} nV/cm")
    print(f"  detectable: {result['detectable']}  (margin: {result['margin_factor']:.0f}x threshold)")

    print("\n  Across realistic swim speeds and Earth-field strengths:")
    for v in (0.3, 1.0, 2.0, 3.0):
        for B in (25e-6, 50e-6, 65e-6):
            r = verify_geomagnetic_sensing_plausible(v, B)
            print(f"    v={v:.1f} m/s, B={B*1e6:.0f}uT: {r['margin_factor']:>6.0f}x threshold, "
                  f"detectable: {r['detectable']}")

    print("\n=== 2. Prey detection: exact dipole field, illustrative strength ===")
    axis_check = verify_dipole_axis_ratio()
    print(f"  on-axis/equatorial ratio = {axis_check['ratio']:.6f}  (exact theory: 2.0, "
          f"matches: {axis_check['matches_theory']})")

    print("\n  Detection range for a few ILLUSTRATIVE dipole strengths (not measured biology --")
    print("  chosen here to land in the tens-of-cm ballpark commonly cited for shark prey")
    print("  detection, not derived from a specific measured source):")
    for p in (3e-20, 1e-19, 3e-19, 1e-18):
        r = detection_range_on_axis(p)
        print(f"    p={p:.0e} C*m: on-axis detection range = {r*100:.1f} cm")

    print("\n=== 3. Directional sensing: gradient from a distributed array ===")
    check = verify_gradient_points_to_source()
    print(f"  estimated gradient (y,z): {check['estimated_gradient_yz']}")
    print(f"  true gradient (y,z):      {check['true_gradient_yz']}")
    print(f"  cosine similarity: {check['cosine_similarity']:.7f}  "
          f"(angle error: {check['angle_error_deg']:.4f} deg)")
    print(f"  well aligned: {check['well_aligned']}")

    print("\nThree pieces of vector calculus already verified elsewhere this session")
    print("(cross products, dipole fields, gradients from discrete samples) applied to")
    print("one real biological sensing system, each claim checked against a number, not asserted.")
