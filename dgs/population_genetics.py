"""The gene pool, made quantitative: allele frequencies, Hardy-Weinberg
equilibrium, and genetic drift -- the population-level extension of the
single-cross Punnett square already in dgs.cell_biology.

A "gene pool" is just a probability distribution over alleles in a
population. `dgs.cell_biology.punnett_cross` answers "what genotypes can two
specific parents produce"; this module answers "what genotype frequencies
does the WHOLE population settle into, generation after generation" -- the
same binomial-sampling idea, scaled from one cross to N individuals over T
generations, including the finite-population noise (genetic drift) that an
infinite-population Hardy-Weinberg calculation can't see.
"""

import numpy as np


def allele_frequency(genotype_counts):
    """p (frequency of the dominant/first allele) from genotype counts
    {'AA': n1, 'Aa': n2, 'aa': n3} -- each Aa contributes one copy of each
    allele, AA contributes two A's, aa contributes two a's."""
    n_AA = genotype_counts.get("AA", 0)
    n_Aa = genotype_counts.get("Aa", 0)
    n_aa = genotype_counts.get("aa", 0)
    total_alleles = 2 * (n_AA + n_Aa + n_aa)
    if total_alleles == 0:
        raise ValueError("genotype_counts has zero individuals")
    n_A = 2 * n_AA + n_Aa
    return n_A / total_alleles


def hardy_weinberg_genotype_frequencies(p):
    """Hardy-Weinberg equilibrium genotype frequencies from allele frequency p
    (q = 1-p): f(AA)=p^2, f(Aa)=2pq, f(aa)=q^2 -- the gene pool's equilibrium
    distribution under random mating, no selection, no drift, no mutation."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    q = 1 - p
    return {"AA": p ** 2, "Aa": 2 * p * q, "aa": q ** 2}


def hardy_weinberg_holds(genotype_counts, tol=0.02):
    """Check whether observed genotype counts match the Hardy-Weinberg
    prediction from their own allele frequency, within `tol` relative
    tolerance on each genotype frequency -- a quick population-genetics
    sanity check (large deviations suggest selection, non-random mating, or
    a small/drifting population)."""
    n = sum(genotype_counts.values())
    p = allele_frequency(genotype_counts)
    expected = hardy_weinberg_genotype_frequencies(p)
    observed = {g: genotype_counts.get(g, 0) / n for g in ("AA", "Aa", "aa")}
    max_rel_dev = max(abs(observed[g] - expected[g]) for g in expected)
    return {"holds": max_rel_dev <= tol, "max_deviation": max_rel_dev,
            "observed": observed, "expected": expected, "p": p}


def simulate_genetic_drift(p0, N, n_generations, seed=0):
    """Wright-Fisher genetic drift: each generation, draw 2N allele copies by
    binomial sampling from the current allele frequency (finite-population
    random mating with no selection) -- the discrete-population noise Hardy-
    Weinberg's infinite-population formula can't capture. Returns the
    allele-frequency trajectory p[0..n_generations]; for small N, p random-
    walks and can fix (hit 0 or 1) -- an allele can vanish from the gene
    pool by chance alone, not just by selection."""
    rng = np.random.default_rng(seed)
    p = np.zeros(n_generations + 1)
    p[0] = p0
    for t in range(n_generations):
        n_A = rng.binomial(2 * N, p[t])
        p[t + 1] = n_A / (2 * N)
        if p[t + 1] in (0.0, 1.0):
            p[t + 1:] = p[t + 1]   # fixed/lost: stays there forever
            break
    return p


def expected_heterozygosity_decay(H0, N, n_generations):
    """Expected heterozygosity decays as H_t = H0 * (1 - 1/(2N))^t under pure
    drift (no mutation, no selection) -- smaller populations lose genetic
    diversity faster, the analytic counterpart to simulate_genetic_drift's
    individual random trajectories."""
    t = np.arange(n_generations + 1)
    return H0 * (1 - 1 / (2 * N)) ** t
