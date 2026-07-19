"""Test parity_bit/check_parity: correct even/odd parity computation, clean
data passes, single-bit flips are detected, double-bit flips are NOT (the
real, well-known blind spot of simple parity checking)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import digital_logic as dl

data = [1, 0, 1, 1, 0, 0, 1, 0]   # four 1s

# 1. even parity: total 1-count (data + parity) is even
p_even = dl.parity_bit(data, even=True)
assert (sum(data) + p_even) % 2 == 0

# 2. odd parity: total 1-count is odd
p_odd = dl.parity_bit(data, even=False)
assert (sum(data) + p_odd) % 2 == 1

# 3. uncorrupted data passes the check
assert dl.check_parity(data, p_even, even=True)

# 4. a single bit flip IS detected
corrupted = data.copy()
corrupted[3] ^= 1
assert not dl.check_parity(corrupted, p_even, even=True)

# 5. a double bit flip is NOT detected -- the known limitation of parity
corrupted2 = data.copy()
corrupted2[3] ^= 1
corrupted2[5] ^= 1
assert dl.check_parity(corrupted2, p_even, even=True)

print("test_parity: all checks passed")
