"""Test dgs.selection_rules: the dipole rules (Delta_l=+-1, Delta_m=0,+-1), Laporte parity,
polarization, allowed-decay enumeration, the metastable 2s state, and photon energetics."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dgs import selection_rules as sr

# 1. the basic dipole predicate
assert sr.is_dipole_allowed(1, 0, 0, 0)          # 2p -> 1s : Delta_l=-1, Delta_m=0
assert not sr.is_dipole_allowed(0, 0, 0, 0)      # 2s -> 1s : Delta_l=0  (forbidden)
assert not sr.is_dipole_allowed(2, 0, 0, 0)      # Delta_l=-2 (forbidden)
assert sr.is_dipole_allowed(2, 0, 1, 1)          # d->p, Delta_m=+1
assert not sr.is_dipole_allowed(1, 1, 0, -1)     # Delta_l ok but Delta_m=-2 (forbidden)
assert sr.is_dipole_allowed(1, 0, 2, 1)          # Delta_l=+1 (absorption direction) also 'allowed'

# 2. parity / Laporte: dipole-allowed transitions always flip parity, forbidden Delta_l even don't
assert sr.parity(0) == 1 and sr.parity(1) == -1 and sr.parity(2) == 1
assert sr.parity_flips(1, 0) and sr.parity_flips(2, 1)      # Delta_l odd -> parity flips
assert not sr.parity_flips(0, 0) and not sr.parity_flips(2, 0)   # Delta_l even -> no flip
for li in range(4):
    for lf in range(4):
        if abs(li-lf) == 1:
            assert sr.parity_flips(li, lf)          # allowed => parity change (Laporte)

# 3. polarization from Delta_m
assert sr.polarization(0, 0) == "pi (linear)"
assert sr.polarization(0, 1) is not None and "sigma" in sr.polarization(0, 1)
assert sr.polarization(1, 0) is not None and "sigma" in sr.polarization(1, 0)
assert sr.polarization(0, 2) is None            # |Delta_m|>1 : no dipole photon

# 4. forbidden reasons name the violated rule
assert sr.forbidden_reason(1, 0, 0, 0) == "allowed"
assert "Delta_l" in sr.forbidden_reason(0, 0, 0, 0)
assert "Delta_m" in sr.forbidden_reason(1, 1, 0, -1)

# 5. transition report: allowed 2p->1s carries a photon; forbidden 2s->1s does not conserve into a line
r = sr.transition_report((2,1,0), (1,0,0))
assert r["allowed"] and r["Delta_l"] == -1 and r["parity_flips"]
assert math.isclose(r["photon_eV"], 10.204, abs_tol=1e-2)
assert math.isclose(r["wavelength_nm"], 121.5, abs_tol=0.6)         # Lyman-alpha
assert math.isclose(r["frequency_Hz"], r["photon_eV"]*1.602176634e-19/6.62607015e-34, rel_tol=1e-9)
rf = sr.transition_report((2,0,0), (1,0,0))
assert not rf["allowed"] and "Delta_l" in rf["reason"]

# 6. Delta_n is unrestricted: 2p->1s and 3p->1s both allowed
assert sr.is_dipole_allowed(1, 0, 0, 0)
assert sr.transition_report((3,1,0), (1,0,0))["allowed"]

# 7. allowed decays enumerated correctly
assert sr.allowed_decays(2, 1, 0) == [(1, 0, 0)]                    # 2p -> 1s only
assert sr.allowed_decays(2, 0, 0) == []                            # 2s: nothing allowed
# 3d (l=2) reaches 2p (l=1) but NOT 1s (Delta_l=2) or 2s (Delta_l=2)
d3 = sr.allowed_decays(3, 2, 0)
assert (2, 1, 0) in d3 and all(lf == 1 and nf == 2 for (nf, lf, mf) in d3)
# 3s (l=0) is NOT metastable: it can decay to 2p
assert (2, 1, 0) in sr.allowed_decays(3, 0, 0)

# 8. metastability and lifetime class (the 'timing')
assert sr.is_metastable(2, 0, 0)                  # the famous metastable 2s
assert not sr.is_metastable(2, 1, 0)              # 2p decays fast
assert not sr.is_metastable(3, 0, 0)              # 3s -> 2p is allowed
assert not sr.is_metastable(1, 0, 0)              # ground state is stable, not metastable
assert "metastable" in sr.lifetime_class(2, 0, 0)
assert "ns" in sr.lifetime_class(2, 1, 0)
assert "ground" in sr.lifetime_class(1, 0, 0)

# 9. Balmer-alpha (n=3->2) sub-lines: which l-changes survive the rules
assert sr.is_dipole_allowed(0, 0, 1, 0)          # 3s -> 2p  allowed
assert sr.is_dipole_allowed(1, 0, 0, 0)          # 3p -> 2s  allowed
assert sr.is_dipole_allowed(2, 0, 1, 0)          # 3d -> 2p  allowed
assert not sr.is_dipole_allowed(1, 0, 1, 0)      # 3p -> 2p  forbidden (Delta_l=0)

# 10. bounds
for bad in (lambda: sr.transition_report((2,2,0), (1,0,0)),   # l>=n invalid
            lambda: sr.allowed_decays(2, 2, 0),
            lambda: sr.is_metastable(3, 3, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_selection_rules: all checks passed")
