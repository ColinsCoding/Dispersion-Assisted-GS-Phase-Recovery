"""Generate notebooks/relativistic_laser_sail_biotech.ipynb -- a concrete
special-relativity demonstration: a laser-driven solar sail (Breakthrough
Starshot-style) carrying a biological sample, analyzed from the sail's own
fast-moving inertial frame. Uses dgs.special_relativity's EXISTING, already-
tested functions (time_dilation, length_contraction, relativistic_doppler,
lorentz_factor) -- not reimplemented, just applied to a concrete, motivating
scenario. Ties back to dgs.michelson_morley (the null result that MOTIVATED
relativity) and the Coppinger/Jalali gas-cell spectroscopy theme (the
Doppler-shifted spectral line is a gas-cell absorption line). NOTE: no
triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "relativistic_laser_sail_biotech.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# A Relativistic Laser Sail Carrying a Biological Sample

A Breakthrough-Starshot-style laser-driven sail, boosted to a relativistic
speed, carries a biological sample and a reference gas cell (the same
absorption-cell theme as the Coppinger/Jalali time-stretch work). Three
consequences of special relativity, all analyzed from Earth's frame,
using `dgs.special_relativity`'s existing, already-tested functions -- not
new formulas, just applied to a concrete scenario.

Motivation reminder: `dgs.michelson_morley` already demonstrated the null
result that motivated special relativity in the first place (no aether
wind detected). This notebook is the other side -- what relativity
actually PREDICTS, once you accept it.""")

code(r"""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
from dgs import special_relativity as sr

C = sr.C_SI
v = 0.8 * C   # a dramatic but clean fraction of c for this thought experiment
lf = sr.lorentz_factor(v)
print(f"sail speed: v = {v:.3e} m/s = {v/C:.2f}c")
print(f"Lorentz factor gamma = {lf['gamma']:.4f}")""")

md(r"""## 1. Time dilation: the biological sample's own clock

A cell-division cycle that takes `tau0 = 24 hours` in the sail's own
(proper) frame -- how long does Earth observe it taking?""")

code(r"""tau0_hours = 24.0
tau0_seconds = tau0_hours * 3600.0
result = sr.time_dilation(tau0_seconds, v)
t_lab_hours = result['t_lab'] / 3600.0
print(f"proper time (sail's own clock): {tau0_hours:.1f} hours")
print(f"Earth-observed time for the SAME cell-division cycle: {t_lab_hours:.2f} hours")
print(f"time dilation ratio (gamma): {result['time_ratio']:.4f}")
print(f"\n--> from Earth, the biological sample appears to age "
      f"{result['time_ratio']:.2f}x SLOWER than its own proper-time clock says.")""")

md(r"""## 2. Length contraction: the sail's own dimensions, as seen from Earth

A sail built 4 meters across (in its own rest frame) -- how wide does
Earth measure it while it's moving?""")

code(r"""L0_sail = 4.0  # meters, rest-frame width
result_L = sr.length_contraction(L0_sail, v)
print(f"rest-frame sail width: {L0_sail:.2f} m")
print(f"Earth-measured width (contracted): {result_L['L_lab']:.2f} m")
print(f"contraction factor 1/gamma: {1/lf['gamma']:.4f}")""")

md(r"""## 3. Relativistic Doppler shift: the onboard gas-cell reference line

The sail carries a reference gas cell (same role as the Coppinger/Jalali
absorption-spectroscopy setup) with a known spectral line -- sodium D,
589 nm (already used elsewhere in this repo's food-science/photonics work
as a real reference wavelength). As the sail recedes from Earth at 0.8c,
what wavelength does Earth actually observe?""")

code(r"""wavelength0_nm = 589.0  # sodium D line, rest frame
f0 = C / (wavelength0_nm * 1e-9)

result_doppler = sr.relativistic_doppler(f0, v, approaching=False)  # receding
f_obs = result_doppler['f_obs']
wavelength_obs_nm = C / f_obs * 1e9

print(f"rest-frame sodium D line: {wavelength0_nm:.1f} nm")
print(f"Earth-observed wavelength (sail receding at 0.8c): {wavelength_obs_nm:.1f} nm")
print(f"redshift z = {result_doppler['redshift_z']:.4f}")
print(f"\n--> the reference line has redshifted from {wavelength0_nm:.0f} nm (yellow-orange)")
print(f"    to {wavelength_obs_nm:.0f} nm -- shifted well into the infrared, NOT visible")
print(f"    to the naked eye anymore. This is the same physics used to measure")
print(f"    galaxy recession speeds via redshifted spectral lines.")""")

md(r"""## Bringing it back to understanding

All three effects -- time dilation, length contraction, the Doppler shift
-- come from the exact same Lorentz transformation
(`dgs.special_relativity.lorentz_transform`), just applied to different
physical quantities (a duration, a length, a frequency). None of them are
independent postulates; they're all consequences of ONE transformation
plus the postulate that the speed of light is the same in every inertial
frame -- the postulate `dgs.michelson_morley`'s null result forced physics
to take seriously in the first place.""")

code(r"""print(f"Single number underlying all three effects: gamma = {lf['gamma']:.4f}")
print(f"  time dilation ratio:        {sr.time_dilation(1.0, v)['time_ratio']:.4f}  (= gamma)")
print(f"  length contraction ratio:   {1/sr.length_contraction(1.0, v)['L_lab']:.4f}  (= gamma)")
print(f"  (Doppler shift depends on gamma AND beta together, via sqrt((1+beta)/(1-beta)))")
assert abs(sr.time_dilation(1.0, v)['time_ratio'] - lf['gamma']) < 1e-12
assert abs(1/sr.length_contraction(1.0, v)['L_lab'] - lf['gamma']) < 1e-12
print("\nVerified: both ratios equal gamma exactly, confirming they're the same")
print("underlying transformation, not two separate coincidental facts.")""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
