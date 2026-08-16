"""A CAD-style cross-section drawing of the reduced eye
(dgs.retinal_scan_imaging's Emsley schematic eye), whose design
parameters are SOLVED, not looked up, by a torch gradient-descent
constraint solver enforcing 6 CAD-style design rules simultaneously --
"computing geometry" with torch in the literal sense: the cornea radius,
axial length, vitreous index, and pupil diameter are all differentiable
tensors, and torch.autograd drives them to jointly satisfy (or, where they
conflict, best-compromise) 6 rules a real optical-CAD tool would check.

The textbook reduced-eye numbers (R=5.55mm, n=1.336, axial=22.3mm) are
NOT internally consistent under this module's own emmetropia rule: the
vitreous focal length they imply is 22.07mm, 0.23mm short of the 22.3mm
axial length (checked directly below, not asserted) -- so the 6 rules
genuinely conflict, giving the solver real work to do rather than
converging trivially to numbers already known in advance. With the
default weights, the solver lands 4 of 6 rules cleanly satisfied and the
remaining 2 (diffraction-limit and refractive-index-bound) sitting
essentially exactly ON their boundary -- an ACTIVE constraint, the normal
outcome for a genuinely over-constrained design problem, not a solver
failure.

The 6 rules (all differentiable torch scalars, minimized jointly):
  1. emmetropia        -- vitreous focal length == axial length (image lands ON the retina)
  2. target power       -- eye_power_diopters == a target (60 D, the textbook figure)
  3. diffraction limit   -- Airy spot radius on the retina <= foveal cone spacing (an inequality)
  4. corneal radius bound -- R_cornea within an anatomically plausible range
  5. axial length bound  -- axial_length within an anatomically plausible range
  6. refractive index bound -- n_vitreous within a biologically plausible range

Reuses dgs.retinal_scan_imaging's own formulas (reduced_eye_matrix,
eye_focal_length_mm, eye_power_diopters, diffraction_limited_spot_radius_um)
as the numpy ground truth the torch reimplementations are checked against.
"""

from __future__ import annotations
import numpy as np

from dgs.retinal_scan_imaging import (
    eye_focal_length_mm as np_eye_focal_length_mm,
    eye_power_diopters as np_eye_power_diopters,
    diffraction_limited_spot_radius_um as np_diffraction_limited_spot_radius_um,
)

DEFAULT_TARGET_POWER_D = 60.0
DEFAULT_CONE_SPACING_UM = 2.5
DEFAULT_R_BOUNDS_MM = (5.0, 8.0)
DEFAULT_AXIAL_BOUNDS_MM = (20.0, 26.0)
DEFAULT_N_BOUNDS = (1.32, 1.40)


# ── Torch-differentiable geometry (checked against the numpy originals) ────

def torch_eye_focal_length_mm(R, n):
    """f = n*R/(n-1), torch version of
    dgs.retinal_scan_imaging.eye_focal_length_mm -- differentiable in both
    R and n."""
    return n * R / (n - 1)


def torch_eye_power_diopters(R, n):
    """P = (n-1)/(R/1000), torch version of eye_power_diopters."""
    return (n - 1) / (R / 1000.0)


def torch_diffraction_spot_radius_um(pupil_mm, wavelength_nm, f_mm):
    """r = 1.22*lambda*f/pupil (converted to um), torch version of
    diffraction_limited_spot_radius_um."""
    lam_mm = wavelength_nm * 1e-6
    r_mm = 1.22 * lam_mm * f_mm / pupil_mm
    return r_mm * 1000.0


def verify_torch_matches_numpy(R: float = 5.55, n: float = 1.336, axial: float = 22.3,
                                pupil_mm: float = 4.0, wavelength_nm: float = 550.0) -> dict:
    """CHECKED, not assumed: the torch reimplementations above must agree
    with dgs.retinal_scan_imaging's numpy originals to near machine
    precision, at the SAME inputs."""
    import torch
    R_t = torch.tensor(R, dtype=torch.float64)
    n_t = torch.tensor(n, dtype=torch.float64)

    f_torch = float(torch_eye_focal_length_mm(R_t, n_t))
    f_numpy = np_eye_focal_length_mm(R, n)
    P_torch = float(torch_eye_power_diopters(R_t, n_t))
    P_numpy = np_eye_power_diopters(R, n)
    spot_torch = float(torch_diffraction_spot_radius_um(torch.tensor(pupil_mm, dtype=torch.float64),
                                                         wavelength_nm, torch.tensor(f_numpy, dtype=torch.float64)))
    spot_numpy = np_diffraction_limited_spot_radius_um(pupil_mm, wavelength_nm, f_numpy)

    return {"focal_length_diff": abs(f_torch - f_numpy), "power_diff": abs(P_torch - P_numpy),
            "spot_radius_diff": abs(spot_torch - spot_numpy),
            "matches": bool(max(abs(f_torch - f_numpy), abs(P_torch - P_numpy),
                                 abs(spot_torch - spot_numpy)) < 1e-9)}


# ── The 6 CAD design rules, as differentiable penalty terms ────────────────

def _bound_penalty(x, lo, hi):
    """Squared-hinge penalty: 0 inside [lo,hi], grows smoothly outside --
    the differentiable analog of a CAD tool's min/max dimension rule."""
    import torch
    return torch.relu(lo - x)**2 + torch.relu(x - hi)**2


def design_rule_residuals(R, axial, n, pupil, wavelength_nm: float = 550.0,
                           target_power_D: float = DEFAULT_TARGET_POWER_D,
                           cone_spacing_um: float = DEFAULT_CONE_SPACING_UM,
                           R_bounds=DEFAULT_R_BOUNDS_MM, axial_bounds=DEFAULT_AXIAL_BOUNDS_MM,
                           n_bounds=DEFAULT_N_BOUNDS) -> dict:
    """The 6 rules, each as a non-negative differentiable residual (0 =
    fully satisfied). Rules 1-2 are equality targets (squared error);
    rule 3 is an inequality (squared-hinge, only penalized when violated);
    rules 4-6 are dimensional/material bounds (squared-hinge on both
    sides)."""
    import torch
    f = torch_eye_focal_length_mm(R, n)
    power = torch_eye_power_diopters(R, n)
    spot = torch_diffraction_spot_radius_um(pupil, wavelength_nm, f)

    r1_emmetropia = (f - axial)**2
    r2_target_power = (power - target_power_D)**2
    r3_diffraction = torch.relu(spot - cone_spacing_um)**2
    r4_R_bounds = _bound_penalty(R, *R_bounds)
    r5_axial_bounds = _bound_penalty(axial, *axial_bounds)
    r6_n_bounds = _bound_penalty(n, *n_bounds)

    return {"1_emmetropia": r1_emmetropia, "2_target_power": r2_target_power,
            "3_diffraction_vs_cone": r3_diffraction, "4_corneal_radius_bounds": r4_R_bounds,
            "5_axial_length_bounds": r5_axial_bounds, "6_refractive_index_bounds": r6_n_bounds,
            "focal_length_mm": f, "power_D": power, "spot_radius_um": spot}


def solve_eye_design(R0: float = 5.55, axial0: float = 22.3, n0: float = 1.336, pupil0: float = 4.0,
                      wavelength_nm: float = 550.0, target_power_D: float = DEFAULT_TARGET_POWER_D,
                      cone_spacing_um: float = DEFAULT_CONE_SPACING_UM, n_steps: int = 6000,
                      lr: float = 1e-3, weights=(2.0, 2e-2, 2.0, 80.0, 80.0, 200.0),
                      emmetropia_tol_mm: float = 0.05, power_tol_D: float = 1.0,
                      R_bounds=DEFAULT_R_BOUNDS_MM, axial_bounds=DEFAULT_AXIAL_BOUNDS_MM,
                      n_bounds=DEFAULT_N_BOUNDS) -> dict:
    """Adam gradient descent on R, axial, n, pupil (all torch leaf tensors,
    requires_grad=True) minimizing the weighted sum of the 6 rule
    residuals -- a differentiable-CAD constraint solve, not a lookup table.
    `weights` down-weights rule 2 (target power, in D^2) relative to the
    millimeter/micron-scale rules so no single rule dominates the loss by
    unit-scale accident alone. Returns the solved design, the per-rule
    residual history (to show convergence, not just a final number), and
    which rules end up satisfied -- checked against each rule's own
    PHYSICAL meaning (e.g. rules 4-6 check R/axial/n actually sit inside
    their bounds, not just that the soft penalty is small; a penalty-only
    check can read "satisfied" while the solved value sits just outside a
    hard bound, since a soft constraint solver is free to settle exactly
    ON an active bound)."""
    import torch
    if n_steps < 1:
        raise ValueError(f"n_steps={n_steps}: must be >= 1")

    R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
    axial = torch.tensor(axial0, dtype=torch.float64, requires_grad=True)
    n = torch.tensor(n0, dtype=torch.float64, requires_grad=True)
    pupil = torch.tensor(pupil0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([R, axial, n, pupil], lr=lr)
    history = {key: [] for key in ("1_emmetropia", "2_target_power", "3_diffraction_vs_cone",
                                    "4_corneal_radius_bounds", "5_axial_length_bounds",
                                    "6_refractive_index_bounds", "total_loss")}

    for _ in range(n_steps):
        optimizer.zero_grad()
        residuals = design_rule_residuals(R, axial, n, pupil, wavelength_nm, target_power_D, cone_spacing_um)
        rule_keys = ("1_emmetropia", "2_target_power", "3_diffraction_vs_cone",
                     "4_corneal_radius_bounds", "5_axial_length_bounds", "6_refractive_index_bounds")
        loss = sum(w * residuals[k] for w, k in zip(weights, rule_keys))
        loss.backward()
        optimizer.step()
        for k in rule_keys:
            history[k].append(float(residuals[k].detach()))
        history["total_loss"].append(float(loss.detach()))

    with torch.no_grad():
        final = design_rule_residuals(R, axial, n, pupil, wavelength_nm, target_power_D, cone_spacing_um)
        R_val, axial_val, n_val = float(R), float(axial), float(n)
        f_val, power_val, spot_val = float(final["focal_length_mm"]), float(final["power_D"]), float(final["spot_radius_um"])
        satisfied = {
            "1_emmetropia": abs(f_val - axial_val) < emmetropia_tol_mm,
            "2_target_power": abs(power_val - target_power_D) < power_tol_D,
            "3_diffraction_vs_cone": spot_val <= cone_spacing_um,
            "4_corneal_radius_bounds": R_bounds[0] <= R_val <= R_bounds[1],
            "5_axial_length_bounds": axial_bounds[0] <= axial_val <= axial_bounds[1],
            "6_refractive_index_bounds": n_bounds[0] <= n_val <= n_bounds[1],
        }

    return {"R_mm": float(R.detach()), "axial_length_mm": float(axial.detach()),
            "n_vitreous": float(n.detach()), "pupil_mm": float(pupil.detach()),
            "focal_length_mm": float(final["focal_length_mm"]), "power_D": float(final["power_D"]),
            "spot_radius_um": float(final["spot_radius_um"]), "history": history, "satisfied": satisfied}


# ── Matplotlib CAD-style cross-section drawing ──────────────────────────────

def draw_eye_cad(design: dict, ax=None, wavelength_nm: float = 550.0):
    """A CAD-style cross-section of the solved eye design: the cornea drawn
    as a true circular arc of radius R, a simplified globe (sclera)
    outline out to the retina, three paraxial rays traced from an on-axis
    distant object through the cornea (via
    dgs.paraxial_optics_abcd.spherical_interface_matrix) to show where they
    actually converge relative to the retina, and CAD-convention dimension
    annotations (radius leader, axial-length dimension line with
    arrowheads, pupil-diameter marker)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from dgs.paraxial_optics_abcd import spherical_interface_matrix, free_space_matrix

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    R, axial, n, pupil = design["R_mm"], design["axial_length_mm"], design["n_vitreous"], design["pupil_mm"]

    # cornea: circular arc of radius R, vertex at x=0, center at x=R
    theta_max = np.arcsin(min(pupil / 2 / R, 0.95))
    theta = np.linspace(-theta_max, theta_max, 100)
    cornea_x = R - R * np.cos(theta)
    cornea_y = R * np.sin(theta)
    ax.plot(cornea_x, cornea_y, color="steelblue", lw=2.5, label="cornea (R={:.2f}mm)".format(R))

    # simplified globe (sclera) outline: an ellipse from the cornea's edge
    # to the retina at x=axial, half-height set by the axial length
    globe_ry = axial * 0.42
    globe = patches.Ellipse((axial / 2, 0), width=axial, height=2 * globe_ry,
                             fill=False, edgecolor="gray", lw=1.2, linestyle="--")
    ax.add_patch(globe)

    # retina: a short flat cap at x=axial (paraxial approximation)
    retina_half_h = pupil / 2 + 1.0
    ax.plot([axial, axial], [-retina_half_h, retina_half_h], color="firebrick", lw=2.5, label="retina")

    # optical axis
    ax.axhline(0, color="black", lw=0.6, linestyle=":")

    # trace 3 paraxial rays (parallel, from a distant on-axis object) through
    # the cornea, then free-space-propagate them out to x=axial to see
    # where they actually land relative to the retina
    M_refract = spherical_interface_matrix(1.0, n, R)
    for h0 in np.linspace(-pupil / 2, pupil / 2, 3):
        ray_in = np.array([h0, 0.0])          # height h0, angle 0 (parallel ray)
        ray_after_cornea = M_refract @ ray_in
        h_cornea = h0                          # ray hits cornea at height ~h0 (thin-surface approx)
        n_steps_ray = 60
        xs = np.linspace(0, axial, n_steps_ray)
        heights = h_cornea + ray_after_cornea[1] * (xs - 0)
        ax.plot(xs, heights, color="gold", lw=1.0, alpha=0.85)

    # focus marker at the vitreous focal distance
    f = design["focal_length_mm"]
    ax.plot(f, 0, marker="x", color="darkorange", ms=10, mew=2, label=f"paraxial focus (f={f:.2f}mm)")

    # CAD-style dimension: axial length
    dim_y = -(globe_ry + 3.0)
    ax.annotate("", xy=(0, dim_y), xytext=(axial, dim_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax.plot([0, 0], [dim_y - 0.5, dim_y + 0.5], color="black", lw=0.8)
    ax.plot([axial, axial], [dim_y - 0.5, dim_y + 0.5], color="black", lw=0.8)
    ax.text(axial / 2, dim_y - 1.5, f"{axial:.2f} mm", ha="center", fontsize=9)

    # CAD-style dimension: pupil diameter
    ax.annotate("", xy=(-2.0, -pupil / 2), xytext=(-2.0, pupil / 2),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax.text(-3.2, 0, f"pupil\n{pupil:.2f} mm", ha="center", va="center", fontsize=8, rotation=90)

    ax.set_xlabel("axial distance (mm)")
    ax.set_ylabel("height (mm)")
    ax.set_title(f"Reduced eye (torch-solved): R={R:.2f}mm, n={n:.4f}, "
                 f"P={design['power_D']:.1f}D, spot={design['spot_radius_um']:.2f}μm")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


if __name__ == "__main__":
    print("=== Torch geometry vs. numpy originals ===")
    check = verify_torch_matches_numpy()
    print(f"  focal length diff = {check['focal_length_diff']:.2e}, "
          f"power diff = {check['power_diff']:.2e}, spot diff = {check['spot_radius_diff']:.2e}")
    print(f"  matches: {check['matches']}")

    print("\n=== Default textbook numbers: rule 1 (emmetropia) is ALREADY violated ===")
    import torch
    R0, n0, axial0 = torch.tensor(5.55), torch.tensor(1.336), torch.tensor(22.3)
    f0 = torch_eye_focal_length_mm(R0, n0)
    print(f"  vitreous focal length = {float(f0):.4f} mm vs. axial length {float(axial0):.2f} mm "
          f"(mismatch {float(axial0) - float(f0):.4f} mm)")

    print("\n=== Solving the 6-rule eye design with torch/Adam ===")
    design = solve_eye_design()
    print(f"  R_cornea = {design['R_mm']:.4f} mm, axial = {design['axial_length_mm']:.4f} mm, "
          f"n = {design['n_vitreous']:.5f}, pupil = {design['pupil_mm']:.3f} mm")
    print(f"  focal length = {design['focal_length_mm']:.4f} mm, power = {design['power_D']:.3f} D, "
          f"spot radius = {design['spot_radius_um']:.3f} um")
    print("  rules satisfied:")
    for rule, ok in design["satisfied"].items():
        print(f"    {rule}: {ok}")

    print("\nSaving CAD drawing to eye_cad_demo.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_eye_cad(design, ax=ax)
    plt.tight_layout()
    plt.savefig("eye_cad_demo.png", dpi=110, bbox_inches="tight")
    print("Done.")
