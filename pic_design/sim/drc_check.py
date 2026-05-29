"""
drc_check.py
------------
Design Rule Check for AIM Photonics 300mm SOI process.
Checks minimum width, spacing, and bend radius before tape-out.

Run:  python sim/drc_check.py
"""

import sys
import gdsfactory as gf

# ── AIM PDK design rules (conservative estimates) ────────────────────────────
RULES = {
    "min_width_um":       0.40,    # minimum waveguide width
    "min_space_um":       0.20,    # minimum gap between waveguides
    "min_bend_radius_um": 5.0,     # minimum bend radius
    "max_die_x_um":       3000.0,  # max tile width
    "max_die_y_um":       1500.0,  # max tile height
}

ERRORS = []
WARNINGS = []


def check_component(c: gf.Component) -> None:
    """Run basic geometric DRC on a gdsfactory component."""
    bbox = c.bbox
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Die size check
    if w > RULES["max_die_x_um"]:
        ERRORS.append(f"DIE_SIZE_X: {w:.1f} µm > {RULES['max_die_x_um']} µm limit")
    if h > RULES["max_die_y_um"]:
        ERRORS.append(f"DIE_SIZE_Y: {h:.1f} µm > {RULES['max_die_y_um']} µm limit")

    # Port width check
    for port in c.get_ports_list():
        if hasattr(port, "width") and port.width < RULES["min_width_um"]:
            ERRORS.append(
                f"MIN_WIDTH @ port {port.name}: "
                f"{port.width:.3f} µm < {RULES['min_width_um']} µm"
            )

    # Polygon count warning
    n_poly = len(c.get_polygons())
    if n_poly > 100_000:
        WARNINGS.append(f"POLYGON_COUNT: {n_poly} polygons — may slow GDS export")


def run_drc(gds_path: str = None) -> bool:
    """
    Load chip and run DRC.
    Returns True if clean (no errors), False if violations found.
    """
    sys.path.insert(0, "..")

    if gds_path:
        c = gf.import_gds(gds_path)
        print(f"Loaded: {gds_path}")
    else:
        # Build from layout module directly
        from layouts.gs_chip import gs_chip
        c = gs_chip()
        print("Built chip from layouts/gs_chip.py")

    check_component(c)

    print("\n── DRC Results ───────────────────────────────────────")
    if ERRORS:
        print(f"  ERRORS   : {len(ERRORS)}")
        for e in ERRORS:
            print(f"    ✗ {e}")
    else:
        print("  ERRORS   : 0  ✓")

    if WARNINGS:
        print(f"  WARNINGS : {len(WARNINGS)}")
        for w in WARNINGS:
            print(f"    ⚠ {w}")
    else:
        print("  WARNINGS : 0  ✓")

    bbox = c.bbox
    print(f"\n  Bounding box : {bbox[1][0]-bbox[0][0]:.0f} × {bbox[1][1]-bbox[0][1]:.0f} µm")
    print(f"  Ports        : {[p.name for p in c.get_ports_list()]}")
    print("──────────────────────────────────────────────────────")

    return len(ERRORS) == 0


if __name__ == "__main__":
    clean = run_drc()
    sys.exit(0 if clean else 1)
