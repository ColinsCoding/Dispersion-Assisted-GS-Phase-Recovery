"""
gs_chip.py
----------
Top-level chip layout: Dispersion-Assisted GS Phase Recovery PIC.

Architecture on chip (AIM Photonics 300mm silicon-on-insulator):

  fiber_in ──► GC_in ──► Y-splitter ──┬──► spiral_D1 ──► GC_out1 ──► fiber → PD1 → I1(t)
                                       │
                                       └──► spiral_D2 ──► GC_out2 ──► fiber → PD2 → I2(t)

  I1(t), I2(t)  →  TD-GS (optical_dashboard/dsp.py)  →  φ(t)  →  OPA

Die size target:  3 mm × 1.5 mm  (fits AIM MPW tile)
"""

import gdsfactory as gf
from components.dispersive_delay import dispersive_spiral, D1_PS2, D2_PS2
from components.input_splitter   import y_splitter_1x2
from components.grating_coupler_io import gc_array


@gf.cell
def gs_chip() -> gf.Component:
    """
    Full TD-GS chip: input GC → splitter → 2× dispersive spiral → output GCs.
    """
    c = gf.Component("gs_phase_retrieval_chip")

    # ── Grating coupler array (1 in, 2 out) ──────────────────────────────────
    gc = c << gc_array(n=3, pitch_um=127.0)
    gc.move((0, 0))

    # ── Y-splitter ────────────────────────────────────────────────────────────
    spl = c << y_splitter_1x2()
    spl.move((200, 127))

    # ── Dispersive arms ───────────────────────────────────────────────────────
    arm1 = c << dispersive_spiral(D_ps2=D1_PS2)
    arm2 = c << dispersive_spiral(D_ps2=D2_PS2)

    arm1.move((400, 200))
    arm2.move((400, 50))

    # ── Route input GC → splitter ─────────────────────────────────────────────
    gf.routing.route_single(
        c,
        gc.ports["wg_0"],
        spl.ports["o1"],
        cross_section="xs_sc",
    )

    # ── Route splitter → arms ─────────────────────────────────────────────────
    gf.routing.route_single(
        c,
        spl.ports["o2"],
        arm1.ports["o1"],
        cross_section="xs_sc",
    )
    gf.routing.route_single(
        c,
        spl.ports["o3"],
        arm2.ports["o1"],
        cross_section="xs_sc",
    )

    # ── Route arms → output GCs ───────────────────────────────────────────────
    gf.routing.route_single(
        c,
        arm1.ports["o2"],
        gc.ports["wg_1"],
        cross_section="xs_sc",
    )
    gf.routing.route_single(
        c,
        arm2.ports["o2"],
        gc.ports["wg_2"],
        cross_section="xs_sc",
    )

    # ── Chip boundary ─────────────────────────────────────────────────────────
    c.add_ref(
        gf.components.rectangle(size=(3000, 1500), layer=(99, 0))
    ).move((-100, -200))

    # ── Labels ───────────────────────────────────────────────────────────────
    c.add_label("IN",           position=(0,   127),  layer=(66, 0))
    c.add_label("OUT_D1_600",   position=(0,   254),  layer=(66, 0))
    c.add_label("OUT_D2_900",   position=(0,   381),  layer=(66, 0))
    c.add_label("TD-GS PIC v1", position=(500, 700),  layer=(66, 0))

    return c


if __name__ == "__main__":
    chip = gs_chip()
    chip.show()                              # opens KLayout

    # Export GDS for foundry submission
    chip.write_gds("../gds_output/gs_chip_v1.gds")
    print("Wrote gds_output/gs_chip_v1.gds")
    print(f"Bounding box: {chip.bbox}")
    print(f"Ports: {[p.name for p in chip.get_ports_list()]}")
