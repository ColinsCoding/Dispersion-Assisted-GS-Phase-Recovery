"""Test dgs.atp_molecular_energetics: cellular ATP concentrations must make
hydrolysis MORE favorable than standard state (a real, checkable
biochemistry fact), the 3-state rate matrix reused from
dgs.photosynthesis_energy_transfer must reach a genuine steady state with
IDENTICAL flux at all three transitions, and the trajectory computed via
solve_population_dynamics must actually converge to that steady state."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.atp_molecular_energetics import (
    atp_hydrolysis_free_energy, atp_synthase_rate_matrix, steady_state_distribution,
    turnover_flux, atp_production_power, ATP_DG_STANDARD_J_PER_MOL,
)
from dgs.photosynthesis_energy_transfer import solve_population_dynamics

# 1. standard-state (1 M each) reproduces the textbook dG almost exactly
standard = atp_hydrolysis_free_energy(1.0, 1.0, 1.0)
assert abs(standard["dG_J_per_mol"] - ATP_DG_STANDARD_J_PER_MOL) < 1e-6

# 2. realistic cellular concentrations (high ATP, low ADP/Pi) make hydrolysis
# MORE favorable (more negative dG) than the standard-state value -- the
# actual biochemistry point, not just "some number changed"
cellular = atp_hydrolysis_free_energy(ATP_M=3e-3, ADP_M=1e-4, Pi_M=3e-3)
assert cellular["dG_J_per_mol"] < standard["dG_J_per_mol"]
assert -70000 < cellular["dG_J_per_mol"] < -40000, \
    f"cellular ATP dG should be in the well-known ~-50 to -60 kJ/mol range, got {cellular['dG_kJ_per_mol']}"

# 3. rejects non-positive concentrations
try:
    atp_hydrolysis_free_energy(0.0, 1e-4, 1e-3)
    assert False, "should reject zero ATP concentration"
except ValueError:
    pass

# 4. the rate matrix has zero column sums (population-conserving cycle)
k_OL, k_LT, k_TO = 300.0, 500.0, 150.0
K = atp_synthase_rate_matrix(k_OL, k_LT, k_TO)
assert K.shape == (3, 3)
assert np.allclose(K.sum(axis=0), 0.0, atol=1e-10), "a closed cycle's rate matrix must conserve total population"

# 5. steady-state populations are non-negative and sum to 1
p_ss = steady_state_distribution(K)
assert abs(p_ss.sum() - 1.0) < 1e-8
assert np.all(p_ss >= -1e-9)
assert np.all(K @ p_ss < 1e-6), "steady state must actually satisfy K p_ss = 0"

# 6. flux is IDENTICAL at all three transitions (the central physical claim
# for an unbranched cycle) -- turnover_flux() already asserts this internally,
# but re-check independently here too
flux = turnover_flux(p_ss, k_OL, k_LT, k_TO)
assert abs(flux["J_OL"] - flux["J_LT"]) < 1e-6
assert abs(flux["J_LT"] - flux["J_TO"]) < 1e-6
assert flux["flux_per_s"] > 0

# 7. faster rates give a faster (not slower) turnover -- a monotonicity sanity check
K_fast = atp_synthase_rate_matrix(k_OL * 2, k_LT * 2, k_TO * 2)
p_ss_fast = steady_state_distribution(K_fast)
flux_fast = turnover_flux(p_ss_fast, k_OL * 2, k_LT * 2, k_TO * 2)
assert flux_fast["flux_per_s"] > flux["flux_per_s"]

# 8. metabolic power: negative (energy released from the ATP pool), and its
# magnitude scales linearly with flux
P = atp_production_power(flux["flux_per_s"], cellular["dG_J_per_mol"])
assert P < 0
P_double = atp_production_power(2 * flux["flux_per_s"], cellular["dG_J_per_mol"])
assert abs(P_double - 2 * P) < 1e-20

# 9. the trajectory from solve_population_dynamics (reused directly from
# dgs.photosynthesis_energy_transfer) actually converges to steady_state_distribution's answer
p0 = np.array([1.0, 0.0, 0.0])
t_long = np.array([0.0, 0.2])
p_t = solve_population_dynamics(K, p0, t_long)
assert np.allclose(p_t[-1], p_ss, atol=1e-3), "long-time trajectory should converge to the steady state"

# 10. input validation
try:
    atp_synthase_rate_matrix(-1.0, 500.0, 150.0)
    assert False, "should reject a negative rate constant"
except ValueError:
    pass
try:
    atp_production_power(-1.0, -30500.0)
    assert False, "should reject negative flux"
except ValueError:
    pass

print(f"all dgs.atp_molecular_energetics tests passed  "
      f"(cellular dG={cellular['dG_kJ_per_mol']:.1f} kJ/mol, flux={flux['flux_per_s']:.1f}/s)")
