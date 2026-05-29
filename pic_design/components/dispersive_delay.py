"""
dispersive_delay.py
-------------------
Chip-scale dispersive delay lines for TD-GS phase retrieval.

D1 = -600 ps²,  D2 = -900 ps²  (same as fiber arms in software pipeline)

Implementation options on silicon photonics (AIM / SiEPIC PDK):
  A) Chirped Bragg grating  — highest dispersion/mm, ~1 ps²/mm achievable
  B) Spiral waveguide       — low dispersion, needs ~300 mm total length
  C) Ring resonator bank    — engineered GVD via resonance proximity

This file implements option A (chirped Bragg grating) and option B (spiral).
"""

import gdsfactory as gf
import numpy as np
from gdsfactory.typings import CrossSectionSpec


# ── Physical constants ────────────────────────────────────────────────────────
LAMBDA_NM   = 1550.0          # center wavelength (nm)
N_EFF       = 2.44            # effective index, 220nm Si wire @ 1550nm
N_G         = 4.18            # group index (from dispersion)
BETA2_PS2MM = -1.0            # GVD of Si wire waveguide (ps²/mm) — anomalous


# ── Target dispersion values (from software pipeline) ────────────────────────
D1_PS2 = -600.0               # ps²  arm 1
D2_PS2 = -900.0               # ps²  arm 2


def spiral_length_mm(D_ps2: float, beta2: float = BETA2_PS2MM) -> float:
    """Waveguide length needed to achieve D ps² of dispersion."""
    return abs(D_ps2 / beta2)


@gf.cell
def dispersive_spiral(
    D_ps2: float = D1_PS2,
    width: float = 0.45,
    spacing: float = 3.0,
    cross_section: CrossSectionSpec = "xs_sc",
) -> gf.Component:
    """
    Archimedean spiral waveguide giving ~D_ps2 of group-velocity dispersion.

    Parameters
    ----------
    D_ps2     : target dispersion in ps²  (negative = anomalous)
    width     : waveguide core width in µm
    spacing   : spiral arm spacing in µm
    """
    L_mm   = spiral_length_mm(D_ps2)
    L_um   = L_mm * 1e3

    c = gf.Component(f"spiral_D{abs(int(D_ps2))}ps2")

    spiral = c << gf.components.spiral_double(
        min_bend_radius=10.0,
        separation=spacing,
        number_of_loops=int(L_um / (2 * np.pi * 10 * spacing)) + 1,
        npoints=1000,
        cross_section=cross_section,
    )

    c.add_ports(spiral.get_ports_list())
    c.info["D_ps2"]      = D_ps2
    c.info["length_mm"]  = L_mm
    c.info["beta2"]      = BETA2_PS2MM
    return c


@gf.cell
def chirped_bragg_arm(
    D_ps2: float = D1_PS2,
    width: float  = 0.45,
    pitch_start_nm: float = 318.0,
    pitch_end_nm:   float = 322.0,
    n_periods: int  = 5000,
) -> gf.Component:
    """
    Chirped Bragg grating delay line.

    Dispersion from linear pitch chirp:
        D ≈ n_g² · L² / (c · Δλ_stop)

    pitch_start/end sets the stopband sweep → controls D.
    """
    c   = gf.Component(f"cbg_D{abs(int(D_ps2))}ps2")
    pitches = np.linspace(pitch_start_nm * 1e-3, pitch_end_nm * 1e-3, n_periods)

    x = 0.0
    for i, p in enumerate(pitches):
        # alternating wide / narrow = grating teeth
        w_wide   = width + 0.05
        w_narrow = width - 0.05
        seg = c << gf.components.straight(
            length=p / 2,
            cross_section=gf.cross_section.cross_section(
                width=w_wide if i % 2 == 0 else w_narrow
            ),
        )
        seg.movex(x)
        x += p / 2

    c.add_port("o1", center=(0, 0),      width=width, orientation=180, layer=(1, 0))
    c.add_port("o2", center=(x, 0),      width=width, orientation=0,   layer=(1, 0))
    c.info["D_ps2"]     = D_ps2
    c.info["n_periods"] = n_periods
    return c


if __name__ == "__main__":
    # Preview both arms
    arm1 = dispersive_spiral(D_ps2=D1_PS2)
    arm2 = dispersive_spiral(D_ps2=D2_PS2)

    print(f"Arm 1  D={D1_PS2} ps²  →  spiral length = {spiral_length_mm(D1_PS2):.1f} mm")
    print(f"Arm 2  D={D2_PS2} ps²  →  spiral length = {spiral_length_mm(D2_PS2):.1f} mm")

    arm1.show()
