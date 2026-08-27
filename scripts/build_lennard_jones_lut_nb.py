"""Build notebooks/lennard_jones_lut.ipynb -- walks through
dgs/lennard_jones_lut.py's lookup-table acceleration of the Lennard-Jones
potential: why real MD codes index by r^2 (not r), how fast linear
interpolation converges to the exact evaluation, what fixed-point
hardware-storage bit width actually costs in accuracy, and a drop-in
comparison against dgs.lennard_jones's exact pair_forces on a real
hexagonal cluster.

Build with `py -3.13 scripts/build_lennard_jones_lut_nb.py`, execute with
`py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
notebooks/lennard_jones_lut.ipynb`.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md("""# Lennard-Jones, but as a hardware engineer would build it

`dgs.lennard_jones` evaluates $V(r) = 4\\varepsilon[(\\sigma/r)^{12} - (\\sigma/r)^6]$
directly -- two real `pow()` calls per pair, every timestep, for every pair
of atoms. Real molecular-dynamics codes (GROMACS, LAMMPS) and
special-purpose MD hardware (D.E. Shaw's Anton machines) don't do that at
scale: they precompute the function on a table and interpolate, exactly
the way a digital-logic lookup table works.

This notebook builds that table, checks that it actually converges to the
exact answer (not just "runs"), and asks the real hardware-design
question: how many bits of fixed-point table storage does a given
accuracy cost?"""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt

from dgs.lennard_jones import lj_potential, lj_force_magnitude, hex_cluster, pair_forces
from dgs.lennard_jones_lut import (
    build_lj_lut, lut_lookup, verify_lut_converges, quantize_table,
    lut_quantization_error, pair_forces_lut,
)
print("loaded dgs.lennard_jones, dgs.lennard_jones_lut")"""))

cells.append(md("""## Part 1 -- the actual trick: index by $r^2$, not $r$

A pairwise distance in any MD loop comes from
$r^2 = dx^2+dy^2+dz^2$ -- you already have $r^2$ for free. Building the
table on a uniform $r^2$ grid means the whole force evaluation never
needs a `sqrt` at all, and the force itself is tabulated as $F(r)/r$ so
applying it is just `(F(r)/r) * (x_i - x_j)` -- no division by $r$
either."""))

cells.append(co("""r2_grid, V_table, FoverR_table = build_lj_lut(r2_min=0.85, r2_max=9.0, n_entries=256)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(np.sqrt(r2_grid), V_table, '.', ms=3, color='#2a6fb0')
axes[0].set_xlabel('r (sigma)'); axes[0].set_ylabel('V(r) (eps)')
axes[0].set_title('Tabulated potential (256 entries, uniform in r^2)')
axes[1].plot(np.sqrt(r2_grid), FoverR_table, '.', ms=3, color='#c0472c')
axes[1].set_xlabel('r (sigma)'); axes[1].set_ylabel('F(r)/r')
axes[1].set_title('Tabulated force/r (multiply by displacement, no sqrt needed)')
fig.tight_layout()
plt.show()"""))

cells.append(md("""## Part 2 -- does the LUT actually converge?

The whole justification for a LUT is that it approximates the exact
evaluation well enough, cheaply. `verify_lut_converges` checks that
directly: RMS error between LUT-interpolated and exact `lj_potential` at
random test points, as a function of table size."""))

cells.append(co("""conv = verify_lut_converges(n_entries_list=(16, 64, 256, 1024, 4096))
for n, err in zip(conv['n_entries'], conv['rms_error']):
    print(f\"n_entries={n:5d}   RMS error vs. exact = {err:.3e}\")
print(f\"monotonically improving: {conv['monotonically_improving']}\")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.loglog(conv['n_entries'], conv['rms_error'], 'o-', color='#2a6fb0')
ax.set_xlabel('LUT entries'); ax.set_ylabel('RMS error vs. exact (eps)')
ax.set_title('Linear-interpolation LUT error vs. table size (log-log)')
fig.tight_layout()
plt.show()"""))

cells.append(md("""Roughly a straight line on log-log axes: linear interpolation converges
like $O(\\text{step}^2)$, so doubling the table size should cut the error
by about 4x -- check the ratios above against that."""))

cells.append(md("""## Part 3 -- the real hardware question: how many bits?

A LUT living in FPGA/ASIC block RAM is finite-width fixed-point memory,
not `float64`. `quantize_table` simulates storing the potential table at
`n_bits` of precision; `lut_quantization_error` sweeps bit width and
measures the RMS error quantization ALONE introduces (table size held
fixed, at 256 entries)."""))

cells.append(co("""quant = lut_quantization_error(n_entries=256, n_bits_list=(4, 6, 8, 10, 12, 16, 24))
print(f\"table dynamic range: {quant['table_range']:.4f} eps\\n\")
for nb_, err in zip(quant['n_bits'], quant['rms_error']):
    print(f\"{nb_:2d}-bit fixed-point:  RMS quantization error = {err:.3e}\")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.semilogy(quant['n_bits'], quant['rms_error'], 'o-', color='#c0472c')
ax.set_xlabel('fixed-point bit width'); ax.set_ylabel('RMS quantization error (eps)')
ax.set_title('Each extra bit roughly halves the table\\'s quantization step')
fig.tight_layout()
plt.show()"""))

cells.append(md("""## Part 4 -- drop-in replacement: does it actually work in an MD loop?

`pair_forces_lut` has the exact same contract as
`dgs.lennard_jones.pair_forces` -- same inputs, same
`(forces, potential_energy)` return -- but every pairwise evaluation is a
table lookup instead of a `pow()` call. Run it on the same hexagonal
cluster used elsewhere in this repo's Lennard-Jones work and compare
directly against the exact evaluation."""))

cells.append(co("""pos = hex_cluster(n_rings=1)
F_exact, U_exact = pair_forces(pos)

r2_grid_fine, V_table_fine, FoverR_fine = build_lj_lut(r2_min=0.7, r2_max=9.0, n_entries=512)
F_lut, U_lut = pair_forces_lut(pos, r2_grid_fine, FoverR_fine, V_table_fine)

print(f\"exact  U = {U_exact:.6f} eps\")
print(f\"LUT    U = {U_lut:.6f} eps   (diff {abs(U_exact-U_lut):.2e})\")
print(f\"max force-component error: {np.max(np.abs(F_exact - F_lut)):.2e}  (512-entry LUT)\")"""))

cells.append(md("""## Summary

| | Exact evaluation | LUT (this notebook) |
|---|---|---|
| Per-pair cost | 2x `pow()` (12th and 6th power) | 1x table lookup + linear blend |
| Needs `sqrt`? | Yes (for the force direction) | No -- everything indexed by $r^2$ |
| Storage | None (recomputed every time) | One table in memory (or block RAM) |
| Accuracy knob | N/A (always exact) | Table size (interpolation) AND bit width (quantization) |

Two independent knobs control accuracy -- table resolution and storage
bit width -- and both were checked here to actually converge/degrade the
way the underlying math predicts, not just assumed. That's the real
systems-engineering tradeoff behind every production MD code's inner
loop, and behind purpose-built MD hardware like Anton: spend memory and
lose a controlled, quantifiable amount of precision to avoid millions of
`pow()` calls per timestep."""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "lennard_jones_lut.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}  ({len(cells)} cells)")
