"""A focal-length picker for hobbyist CO2 laser cutters: the product
identified while analyzing a real amateur-photonics Shopify store
(american-photonics.myshopify.com) -- every replacement lens they sell
(1.5"/2"/2.5"/4" focal lengths, 10.6um CO2) makes a buyer choose between
a finer spot (shorter focal length) and more tolerance for material
height / cut-through depth (longer focal length, more depth of focus).
No one selling the lens tells a buyer which one to pick for THEIR
material and use case -- this module answers that, on top of the
already-tested dgs.paraxial_optics_abcd Gaussian-beam ABCD machinery
(reused directly, not reimplemented).

THE PHYSICS (nothing new -- straight Gaussian beam optics):
  A CO2 tube's raw output has some beam radius w0_raw. Propagated
  through a thin focusing lens of focal length f, it comes to a NEW,
  much smaller waist w0_focus, with depth of focus (2x Rayleigh range)
  around that focus. Both SHRINK together as f shrinks -- there is no
  free lunch: you cannot get a fine spot AND a deep, forgiving focus
  from a single fixed-focus lens.

THE DECISION RULE this module encodes (a real, defensible engineering
choice, stated explicitly rather than hidden in code):
  ENGRAVING only ever cuts the surface -- always recommend the
    SHORTEST available focal length (finest resolution), since there is
    no material depth to stay in focus through.
  CUTTING must stay reasonably focused through the ENTIRE material
    thickness -- recommend the SHORTEST focal length whose depth of
    focus still covers the material thickness; if none of the standard
    sizes do, recommend the LONGEST available and say so honestly
    (the material is simply thick enough that no standard hobbyist lens
    keeps a tight focus all the way through).
"""
import numpy as np

from dgs.paraxial_optics_abcd import (
    gaussian_beam_q, transform_q, beam_waist_and_curvature,
    thin_lens_matrix, free_space_matrix,
)

WAVELENGTH_CO2_M = 10.6e-6   # meters -- every hobbyist CO2 tube runs at this line

STANDARD_FOCAL_LENGTHS_INCH = (1.5, 2.0, 2.5, 4.0)   # the sizes actually sold by hobbyist optics shops

# Typical raw beam radius at the focusing lens, by common hobbyist tube power class.
# These are ROUGH, COMMONLY-CITED defaults for K40-class (40W) vs mid-power (60-80W)
# vs higher-power (100-150W) hobbyist tubes -- NOT a precisely measured universal law,
# and overridable via the raw_beam_radius_mm argument below. Stated as an assumption,
# not fabricated precision.
DEFAULT_RAW_BEAM_RADIUS_MM_BY_POWER = {
    40: 3.0,
    60: 3.5,
    80: 4.0,
    100: 4.5,
    130: 5.0,
    150: 5.0,
}


def default_raw_beam_radius_mm(power_W):
    """Nearest-power lookup into the rough defaults table above."""
    if power_W <= 0:
        raise ValueError("power_W must be positive")
    powers = sorted(DEFAULT_RAW_BEAM_RADIUS_MM_BY_POWER)
    nearest = min(powers, key=lambda p: abs(p - power_W))
    return DEFAULT_RAW_BEAM_RADIUS_MM_BY_POWER[nearest]


def focused_spot_and_depth_of_focus(focal_length_inch, raw_beam_radius_mm,
                                     wavelength_m=WAVELENGTH_CO2_M, path_to_lens_m=1.0):
    """Propagate a Gaussian beam of raw radius raw_beam_radius_mm through
    path_to_lens_m of free space, then a thin lens of the given focal
    length (in inches, matching how these lenses are actually sold),
    and return (focused_spot_diameter_mm, depth_of_focus_mm) at the new
    waist -- built entirely from dgs.paraxial_optics_abcd's tested
    ABCD/q-parameter functions."""
    if focal_length_inch <= 0:
        raise ValueError("focal_length_inch must be positive")
    if raw_beam_radius_mm <= 0:
        raise ValueError("raw_beam_radius_mm must be positive")

    f_m = focal_length_inch * 0.0254
    w0_raw_m = raw_beam_radius_mm * 1e-3

    q0 = gaussian_beam_q(0, w0_raw_m, wavelength_m)
    q_before_lens = transform_q(free_space_matrix(path_to_lens_m), q0)
    q_after_lens = transform_q(thin_lens_matrix(f_m), q_before_lens)
    q_at_focus = transform_q(free_space_matrix(f_m), q_after_lens)
    w_focus_m, _ = beam_waist_and_curvature(q_at_focus, wavelength_m)

    z_R_m = np.pi * w_focus_m ** 2 / wavelength_m
    return 2 * w_focus_m * 1e3, 2 * z_R_m * 1e3   # (spot diameter mm, depth of focus mm)


def recommend_focal_length(use_case, material_thickness_mm=None, power_W=60.0,
                            raw_beam_radius_mm=None,
                            standard_focal_lengths_inch=STANDARD_FOCAL_LENGTHS_INCH):
    """The actual recommendation. use_case: 'engrave' or 'cut'.
    For 'cut', material_thickness_mm is required. Returns a dict with the
    recommended focal length and its predicted spot size / depth of
    focus, plus the full comparison table so a buyer can see the
    tradeoff, not just the answer."""
    if use_case not in ("engrave", "cut"):
        raise ValueError("use_case must be 'engrave' or 'cut'")
    if use_case == "cut" and (material_thickness_mm is None or material_thickness_mm <= 0):
        raise ValueError("material_thickness_mm (>0) is required for use_case='cut'")

    if raw_beam_radius_mm is None:
        raw_beam_radius_mm = default_raw_beam_radius_mm(power_W)

    table = []
    for f_inch in sorted(standard_focal_lengths_inch):
        spot_mm, dof_mm = focused_spot_and_depth_of_focus(f_inch, raw_beam_radius_mm)
        table.append({"focal_length_inch": f_inch, "spot_diameter_mm": spot_mm,
                      "depth_of_focus_mm": dof_mm, "estimated_kerf_mm": spot_mm})

    if use_case == "engrave":
        chosen = table[0]   # shortest available -- finest resolution, no depth requirement
        reason = "engraving only cuts the surface -- always pick the shortest focal length for the finest detail"
    else:
        fitting = [row for row in table if row["depth_of_focus_mm"] >= material_thickness_mm]
        if fitting:
            chosen = fitting[0]   # shortest focal length that still covers the material thickness
            reason = (f"shortest focal length whose depth of focus ({chosen['depth_of_focus_mm']:.2f}mm) "
                       f"still covers the {material_thickness_mm}mm material thickness")
        else:
            chosen = table[-1]   # nothing covers it -- recommend the longest available, say so honestly
            reason = (f"NO standard lens has enough depth of focus for {material_thickness_mm}mm material "
                       f"(best available: {chosen['depth_of_focus_mm']:.2f}mm at f={chosen['focal_length_inch']}\") "
                       f"-- expect a visibly tapered/beveled cut edge, or consider multiple passes")

    return {
        "use_case": use_case, "material_thickness_mm": material_thickness_mm,
        "power_W": power_W, "raw_beam_radius_mm": raw_beam_radius_mm,
        "recommended_focal_length_inch": chosen["focal_length_inch"],
        "predicted_spot_diameter_mm": chosen["spot_diameter_mm"],
        "predicted_depth_of_focus_mm": chosen["depth_of_focus_mm"],
        "predicted_kerf_mm": chosen["estimated_kerf_mm"],
        "reason": reason,
        "full_comparison_table": table,
    }


if __name__ == "__main__":
    print("=== CO2 LASER FOCAL-LENGTH PICKER ===\n")

    print("--- Engraving (always shortest focal length) ---")
    r = recommend_focal_length(use_case="engrave", power_W=60.0)
    print(f"recommended: {r['recommended_focal_length_inch']}\" lens")
    print(f"  spot diameter: {r['predicted_spot_diameter_mm']:.3f} mm")
    print(f"  reason: {r['reason']}")

    print("\n--- Cutting 3mm plywood, 60W ---")
    r = recommend_focal_length(use_case="cut", material_thickness_mm=3.0, power_W=60.0)
    print(f"recommended: {r['recommended_focal_length_inch']}\" lens")
    print(f"  spot diameter: {r['predicted_spot_diameter_mm']:.3f} mm, "
          f"depth of focus: {r['predicted_depth_of_focus_mm']:.2f} mm")
    print(f"  reason: {r['reason']}")

    print("\n--- Cutting 12mm acrylic, 100W (thick material) ---")
    r = recommend_focal_length(use_case="cut", material_thickness_mm=12.0, power_W=100.0)
    print(f"recommended: {r['recommended_focal_length_inch']}\" lens")
    print(f"  spot diameter: {r['predicted_spot_diameter_mm']:.3f} mm, "
          f"depth of focus: {r['predicted_depth_of_focus_mm']:.2f} mm")
    print(f"  reason: {r['reason']}")

    print("\n--- Full comparison table (100W, for the acrylic case above) ---")
    for row in r["full_comparison_table"]:
        print(f"  f={row['focal_length_inch']:.1f}\": spot={row['spot_diameter_mm']:.3f}mm, "
              f"DOF={row['depth_of_focus_mm']:.2f}mm")

    print("\n=== DONE ===")
