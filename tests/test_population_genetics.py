"""Test population_genetics: Hardy-Weinberg recovers p^2:2pq:q^2 exactly,
flags a skewed (inbred) population, and genetic drift fixes faster (with
higher variance) in small populations than large ones."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import population_genetics as pg

# 1. allele frequency from genotype counts
assert abs(pg.allele_frequency({"AA": 25, "Aa": 50, "aa": 25}) - 0.5) < 1e-12

# 2. a population already at p^2:2pq:q^2 reports zero deviation from HW
hw_counts = {"AA": 25, "Aa": 50, "aa": 25}
check = pg.hardy_weinberg_holds(hw_counts)
assert check["holds"]
assert check["max_deviation"] < 1e-9

# 3. a skewed (e.g. inbred) population is correctly flagged as NOT at HW
skewed = {"AA": 40, "Aa": 10, "aa": 50}
check_skewed = pg.hardy_weinberg_holds(skewed)
assert not check_skewed["holds"]
assert check_skewed["max_deviation"] > 0.1

# 4. genetic drift: small populations fix/lose alleles faster and with more
#    variance than large populations (Wright-Fisher)
small_trajs = [pg.simulate_genetic_drift(0.5, N=10, n_generations=300, seed=s) for s in range(30)]
large_trajs = [pg.simulate_genetic_drift(0.5, N=5000, n_generations=300, seed=s) for s in range(30)]
small_final_var = np.var([t[-1] for t in small_trajs])
large_final_var = np.var([t[-1] for t in large_trajs])
assert small_final_var > large_final_var * 5

# 5. expected heterozygosity decays monotonically under pure drift
H = pg.expected_heterozygosity_decay(H0=0.5, N=50, n_generations=100)
assert np.all(np.diff(H) <= 0)
assert H[0] == 0.5

print("test_population_genetics: all checks passed")
