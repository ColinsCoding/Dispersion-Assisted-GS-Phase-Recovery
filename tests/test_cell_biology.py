"""Test cell_biology: Fick's law sign, Punnett ratios, chi-square, Beer-Lambert,
action spectrum shape, and the yeast-vs-plant turgor comparison."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import cell_biology as cb

# 1. Fick's law: flux is down the gradient (negative dC/dx -> positive flux)
assert cb.fick_flux(D=1e-9, dC_dx=-5.0, A=1.0) > 0
assert cb.fick_flux(D=1e-9, dC_dx=5.0, A=1.0) < 0

# 2. diffusion time scale grows with L^2, not L
t_short = cb.diffusion_time_scale(L=1e-6, D=1e-9)
t_long = cb.diffusion_time_scale(L=2e-6, D=1e-9)
assert abs(t_long / t_short - 4.0) < 1e-9

# 3. monohybrid Punnett cross gives the textbook 1:2:1 genotype ratio
cross = cb.punnett_cross("Aa", "Aa")
assert cross == {"AA": 1, "Aa": 2, "aa": 1}
pheno = cb.phenotype_ratio(cross, "A")
assert pheno == {"dominant": 3, "recessive": 1}

# 4. chi-square is zero for a perfect match, positive for a skewed one
assert cb.chi_square_heredity_test([15, 5], [15, 5]) == 0.0
assert cb.chi_square_heredity_test([12, 8], [15, 5]) > 0

# 5. Beer-Lambert: absorption increases (transmitted intensity decreases) with
#    concentration and path length
I_short = cb.beer_lambert_absorption(1.0, epsilon=0.5, concentration=1.0, path_length=1.0)
I_long = cb.beer_lambert_absorption(1.0, epsilon=0.5, concentration=1.0, path_length=3.0)
assert I_long < I_short

# 6. chlorophyll action spectrum: low absorption in the green "gap" near 550nm,
#    high absorption at the red peak near 660nm (why leaves look green)
spectrum = cb.chlorophyll_action_spectrum([550.0, 660.0])
assert spectrum[1] > spectrum[0]

# 7. quantum yield is a simple ratio, capped sensibly below 1 for realistic inputs
assert abs(cb.quantum_yield(85, 100) - 0.85) < 1e-9

# 8. the level-up: an unwalled yeast cell and a plant cell see the same osmotic
#    pressures, but the plant's turgor pressure suppresses net water flux
yeast = cb.yeast_cell_water_balance(C_in=300, C_out=100)
plant_no_wall = cb.plant_cell_water_balance(C_in=300, C_out=100, wall_strain=0.0)
plant_turgid = cb.plant_cell_water_balance(C_in=300, C_out=100, wall_strain=0.0495)
assert abs(plant_no_wall["water_flux"] - yeast["water_flux"]) < 1e-25  # same physics, no wall yet
assert plant_turgid["water_flux"] < yeast["water_flux"] / 100          # wall meaningfully caps influx

print("test_cell_biology: all checks passed")
