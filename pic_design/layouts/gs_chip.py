"""
gs_chip.py — gdsfactory 9.x layout for two-arm dispersive GS receiver
Exports GDS to pic_design/gds_output/gs_receiver.gds

Chip architecture
-----------------
  Fiber → Grating coupler → MMI 1×2 splitter
             ├── Arm 1: delay snake  L1 = 1600 µm → PD1
             └── Arm 2: delay snake  L2 = 1840 µm → PD2

  L2/L1 = 1.15  →  same diversity ratio as Solli 2009 (D2/D1 = −800/−695)
  ΔL = 240 µm   →  Δ(group delay) = β₂ · ΔL ≈ 0.24 ps²  (chip scale)

Si strip 220 nm × 450 nm at 1550 nm: β₂ ≈ −1.0 ps²/mm  (anomalous)
Compare to fiber: Solli used DCF with D ≈ −700 ps/nm → need ~1 m of DCF.
On-chip version is ×4000 shorter; compensate with higher-dispersion waveguides
(photonic crystal, slot, or chirped Bragg) in a real tapeout.
"""

import pathlib
import gdsfactory as gf
from gdsfactory.gpdk import get_generic_pdk

get_generic_pdk().activate()

# ── Arm lengths (µm).  Ratio 1.15 matches paper's D2/D1. ─────────────────────
L1_UM = 1_600.0
L2_UM = 1_840.0    # = L1 × 1.15


@gf.cell
def arm(length: float, n_bends: int = 4) -> gf.Component:
    """Delay-snake dispersive arm of specified total path length."""
    return gf.components.delay_snake2(length=length, n=n_bends)


@gf.cell
def gs_receiver() -> gf.Component:
    """
    Two-arm dispersive GS receiver PIC.

    Ports
    -----
    o_gc   : fiber input (grating coupler)
    o_arm1 : Arm 1 output → photodetector 1
    o_arm2 : Arm 2 output → photodetector 2
    """
    c = gf.Component()

    gc  = c << gf.components.grating_coupler_elliptical_te()
    mmi = c << gf.components.mmi1x2()
    a1  = c << arm(L1_UM)
    a2  = c << arm(L2_UM)

    # ── Place: GC → MMI ───────────────────────────────────────────────────────
    mmi.xmin = gc.xmax + 20
    mmi.y    = gc.y

    # ── Place: arms side by side to the right of MMI ──────────────────────────
    a1.xmin = mmi.xmax + 30
    a1.y    = mmi.ports["o2"].y

    a2.xmin = mmi.xmax + 30
    a2.y    = mmi.ports["o3"].y - a2.ysize - 20

    # ── Expose top-level ports ────────────────────────────────────────────────
    c.add_port("o_gc",   port=gc.ports["o1"])
    c.add_port("o_arm1", port=a1.ports["o2"])
    c.add_port("o_arm2", port=a2.ports["o2"])

    return c


if __name__ == "__main__":
    out_dir = pathlib.Path(__file__).parent.parent / "gds_output"
    out_dir.mkdir(exist_ok=True)

    chip = gs_receiver()
    gds_path = out_dir / "gs_receiver.gds"
    chip.write_gds(str(gds_path))

    print(f"GDS written : {gds_path}")
    print(f"Die size    : {chip.xsize/1000:.3f} mm × {chip.ysize/1000:.3f} mm")
    print(f"Arm 1       : {L1_UM/1000:.3f} mm")
    print(f"Arm 2       : {L2_UM/1000:.3f} mm  (ratio = {L2_UM/L1_UM:.2f})")
    print(f"Ports       : {[p.name for p in chip.ports]}")
