"""
grating_coupler_io.py
---------------------
Fiber-to-chip I/O: grating couplers for input (laser) and
two outputs (PD1, PD2 → I1(t), I2(t) → GS pipeline).

Coupling loss: ~3 dB per coupler (standard AIM PDK value).
Bandwidth:     ~40 nm  @ 1550 nm center
Angle:         8° (standard for SiO2 cladding, avoids back-reflection)
"""

import gdsfactory as gf


@gf.cell
def gc_array(
    n: int          = 3,          # 1 input + 2 outputs
    pitch_um: float = 127.0,      # standard fiber array pitch
    polarization: str = "te",
) -> gf.Component:
    """
    Linear array of n grating couplers at 127µm pitch.
    Matches standard 8° fiber array (Oz Optics, OZ-1550-8).

    Port naming:
        fiber_0  →  laser input
        fiber_1  →  PD1 output  (arm D1 = -600 ps²)
        fiber_2  →  PD2 output  (arm D2 = -900 ps²)
    """
    c   = gf.Component(f"gc_array_{n}ch")
    gcs = []

    for i in range(n):
        gc = c << gf.components.grating_coupler_elliptical_trenches(
            polarization=polarization,
            wavelength=1.55,
        )
        gc.movey(i * pitch_um)
        gc.rotate(180 if i == 0 else 0)
        gcs.append(gc)
        c.add_port(
            name=f"fiber_{i}",
            port=gc.ports["o1"],
        )
        c.add_port(
            name=f"wg_{i}",
            port=gc.ports["o2"],
        )

    c.info["n_channels"]    = n
    c.info["pitch_um"]      = pitch_um
    c.info["coupling_db"]   = -3.0
    c.info["bandwidth_nm"]  = 40
    return c


if __name__ == "__main__":
    c = gc_array(n=3)
    c.show()
    print(c.get_ports_list())
