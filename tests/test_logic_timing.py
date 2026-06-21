"""Test digital-logic timing: adder logic, critical-path delay, hazards, shifts."""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import logic_timing as lt

# 1. the ripple-carry adder actually adds (exhaustive 4-bit check)
add4 = lt.ripple_carry_adder(4)
for a in range(16):
    for b in range(16):
        vals = {"cin": 0,
                **{f"a{i}": (a >> i) & 1 for i in range(4)},
                **{f"b{i}": (b >> i) & 1 for i in range(4)}}
        out = add4.evaluate(vals)
        s = sum((out[f"sum{i}"] << i) for i in range(4)) + (out["cout3"] << 4)
        assert s == a + b, (a, b, s)

# 2. critical-path delay grows LINEARLY with bit width (the ripple penalty)
d4, _ = lt.ripple_carry_adder(4).critical_path()
d8, _ = lt.ripple_carry_adder(8).critical_path()
d16, _ = lt.ripple_carry_adder(16).critical_path()
assert d4 == lt.ripple_carry_delay(4)          # matches the closed form
assert d8 == lt.ripple_carry_delay(8)
# linear in n: the per-bit slope (delay/bit) is constant (= 2 here)
assert (d8 - d4) // (8 - 4) == (d16 - d8) // (16 - 8) == 2
assert d16 > d8 > d4

# 3. carry-lookahead is asymptotically faster (constant-ish depth) for wide adders
assert lt.carry_lookahead_delay(16) < lt.ripple_carry_delay(16)
assert lt.carry_lookahead_delay(64) < lt.ripple_carry_delay(64)

# 4. a hand circuit with a known critical path: AND then OR
c = lt.Circuit()
for nm in ("x", "y", "z"):
    c.add_input(nm)
c.add_gate("g1", "AND", ["x", "y"])            # delay 1
c.add_gate("g2", "OR", ["g1", "z"])            # delay 1 -> arrival 2
c.mark_output("g2")
d, path = c.critical_path()
assert d == 2 and path[-1] == "g2" and "g1" in path

# 5. fmax = 1 / critical delay
assert abs(c.fmax() - 0.5) < 1e-12

# 6. static hazard window = difference of reconverging path delays
assert lt.detect_static_hazard(1, 1) == 0       # equal paths -> no glitch
assert lt.detect_static_hazard(1, 3) == 2       # 2-unit glitch window

# 7. bit shifts = multiply / divide by powers of two
assert lt.logical_shift(5, 1) == 10             # <<1 = x2
assert lt.logical_shift(20, -2) == 5            # >>2 = /4
assert lt.logical_shift(255, 1, width=8) == 254 # masked to 8 bits (overflow drops)
assert lt.barrel_shifter_levels(32) == 5        # log2(32) mux levels
assert lt.barrel_shifter_levels(8) == 3

print(f"TEST PASS  (4-bit adder exact for all 256 cases; ripple delay linear "
      f"d4={d4} d8={d8} d16={d16}; CLA<ripple; hazard window; shift=mul/div; "
      f"barrel levels=log2 n)")
