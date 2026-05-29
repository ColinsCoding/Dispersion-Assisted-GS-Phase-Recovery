"""
input_splitter.py
-----------------
1x2 Y-junction and directional coupler for splitting input light
into the two dispersive arms (D1, D2).

                    ┌── arm 1 (D1 = -600 ps²) ──► PD1
input ──► splitter ─┤
                    └── arm 2 (D2 = -900 ps²) ──► PD2
"""

import gdsfactory as gf


@gf.cell
def y_splitter_1x2(
    width: float = 0.45,
    length: float = 10.0,
) -> gf.Component:
    """Standard 1x2 Y-junction. ~50/50 split, broadband."""
    c = gf.Component("y_splitter_1x2")
    y = c << gf.components.splitter_1x2(
        gap=0.2,
        length_taper=length,
        cross_section=gf.cross_section.strip(width=width),
    )
    c.add_ports(y.get_ports_list())
    return c


@gf.cell
def dc_splitter(
    width: float = 0.45,
    gap: float   = 0.2,
    length: float = 15.0,
) -> gf.Component:
    """
    Directional coupler — tunable split ratio via length.
    length=15µm ≈ 3dB (50/50) at 1550nm for 220nm Si.
    """
    c = gf.Component("dc_splitter")
    dc = c << gf.components.coupler(
        gap=gap,
        length=length,
        dx=10.0,
        dy=4.0,
        cross_section=gf.cross_section.strip(width=width),
    )
    c.add_ports(dc.get_ports_list())
    return c


if __name__ == "__main__":
    c = y_splitter_1x2()
    c.show()
