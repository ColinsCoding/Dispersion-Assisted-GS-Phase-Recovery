"""Test dgs.periodic_table: the Madelung fill order (4s before 3d), subshell
capacities 2(2l+1), ground-state configurations of known elements (H, C, Na, Ar,
K, Fe), noble-gas cores, period/valence/noble-gas classification, and electrons
summing to Z for every element."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import periodic_table as pt

# 1. Madelung order: 1s 2s 2p 3s 3p 4s 3d ... (4s BEFORE 3d)
order = pt.madelung_order()
assert order[:7] == [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2)]
assert order.index((4, 0)) < order.index((3, 2))         # the famous 4s < 3d
# strictly sorted by (n+l, n)
keys = [(n + l, n) for n, l in order]
assert keys == sorted(keys)

# 2. subshell capacities 2(2l+1)
assert [pt.subshell_capacity(l) for l in (0, 1, 2, 3)] == [2, 6, 10, 14]

# 3. known ground-state configurations (written sorted by n, l)
assert pt.configuration_string(1) == "1s1"
assert pt.configuration_string(2) == "1s2"
assert pt.configuration_string(6) == "1s2 2s2 2p2"           # carbon
assert pt.configuration_string(10) == "1s2 2s2 2p6"          # neon
assert pt.configuration_string(11) == "1s2 2s2 2p6 3s1"      # sodium
assert pt.configuration_string(18) == "1s2 2s2 2p6 3s2 3p6"  # argon
assert pt.configuration_string(19) == "1s2 2s2 2p6 3s2 3p6 4s1"   # K: 4s, not 3d
assert pt.configuration_string(26) == "1s2 2s2 2p6 3s2 3p6 3d6 4s2"  # iron

# 4. noble-gas core abbreviations
assert pt.configuration_string(11, noble_core=True) == "[Ne] 3s1"
assert pt.configuration_string(26, noble_core=True) == "[Ar] 3d6 4s2"
assert pt.configuration_string(19, noble_core=True) == "[Ar] 4s1"

# 5. period and valence
assert pt.period(1) == 1 and pt.period(11) == 3 and pt.period(19) == 4
assert pt.valence_electrons(11) == 1        # Na: 3s1
assert pt.valence_electrons(6) == 4         # C: 2s2 2p2
assert pt.valence_electrons(10) == 8        # Ne: 2s2 2p6
assert pt.valence_electrons(26) == 2        # Fe: 4s2 (outer shell)

# 6. noble gases are exactly {2,10,18,36,54,86}
assert all(pt.is_noble_gas(z) for z in (2, 10, 18, 36, 54, 86))
assert not pt.is_noble_gas(11) and not pt.is_noble_gas(6)
# and a noble gas has a full outer shell (valence 8, except He=2)
assert pt.valence_electrons(2) == 2
for z in (10, 18, 36, 54, 86):
    assert pt.valence_electrons(z) == 8

# 7. electrons always sum to Z
for Z in range(1, 119):
    assert pt.total_electrons(Z) == Z

# 8. kwarg bounds
for bad in (lambda: pt.electron_configuration(0),
            lambda: pt.electron_configuration(119),
            lambda: pt.subshell_capacity(4)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_periodic_table: all checks passed")
