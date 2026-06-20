"""Smoke-test the accumulator CPU (ALU -> instruction set) in digital_logic."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import digital_logic as dl

# 1. arithmetic: 3 + 4 = 7
r = dl.run_program([("LOADI", 3), ("STORE", 0), ("LOADI", 4), ("ADD", 0), ("HALT", 0)])
assert r["acc"] == 7, r["acc"]

# 2. subtraction with two's complement: 10 - 6 = 4
r = dl.run_program([("LOADI", 10), ("STORE", 0), ("LOADI", 6),
                    ("STORE", 1), ("LOAD", 0), ("SUB", 1), ("HALT", 0)])
assert r["acc"] == 4, r["acc"]

# 3. bitwise logic: 0b1100 XOR 0b1010 = 0b0110 = 6
r = dl.run_program([("LOADI", 0b1010), ("STORE", 0), ("LOADI", 0b1100),
                    ("XOR", 0), ("HALT", 0)])
assert r["acc"] == 0b0110, r["acc"]

# 4. NOT in 8-bit two's complement: ~0 = 255
r = dl.run_program([("LOADI", 0), ("NOT", 0), ("HALT", 0)])
assert r["acc"] == 255, r["acc"]

# 5. control flow: multiply 3*4 by repeated addition (uses JZ + JMP loop)
mem = [0] * 256
mem[0], mem[1], mem[2], mem[3] = 3, 4, 0, 1     # a, counter=b, result, one
prog = [
    ("LOAD", 1),   # 0: ACC = counter
    ("JZ", 9),     # 1: if counter==0 -> end
    ("LOAD", 2),   # 2: ACC = result
    ("ADD", 0),    # 3: result + a
    ("STORE", 2),  # 4: result = ACC
    ("LOAD", 1),   # 5: ACC = counter
    ("SUB", 3),    # 6: counter - 1
    ("STORE", 1),  # 7: counter = ACC
    ("JMP", 0),    # 8: loop
    ("LOAD", 2),   # 9: ACC = result
    ("HALT", 0),   # 10
]
r = dl.run_program(prog, mem=mem, trace=True)
assert r["acc"] == 12, r["acc"]
assert r["mem"][2] == 12
assert r["flags"]["zero"] == 0
assert any(step[2] == "JZ" for step in r["trace"])     # the branch really fired

# 6. zero flag + JZ taken: 5 - 5 = 0 sets zero, branch jumps over the LOADI 99
r = dl.run_program([("LOADI", 5), ("STORE", 0), ("LOAD", 0), ("SUB", 0),
                    ("JZ", 6), ("LOADI", 99), ("HALT", 0)])
assert r["acc"] == 0 and r["flags"]["zero"] == 1, r

# 7. guards: unknown instruction + runaway loop
try:
    dl.run_program([("FOO", 0)])
except ValueError:
    pass
else:
    raise AssertionError("should reject unknown instruction")
try:
    dl.run_program([("JMP", 0)], max_cycles=50)   # infinite self-loop
except RuntimeError:
    pass
else:
    raise AssertionError("should trip max_cycles guard")

print(f"SMOKE PASS  (3*4={dl.run_program(prog, mem=mem)['acc']} via repeated-add loop, "
      f"{r['cycles']} cycles in test 6)")
