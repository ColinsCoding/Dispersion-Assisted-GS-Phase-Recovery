"""Test the gate-level Boolean address decoder: exhaustive one-hot
verification, cross-check against the arithmetic flat-index formula
already trusted in test_cuda_pointer_arithmetic.py, and confirm reading
the wrong address really does read a DIFFERENT physical DRAM cell's
voltage, not a labeled error."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import memory_address_decoder as mad
from dgs import memory_circuits as mc

# 1. the decoder is one-hot for EVERY address, not just a few samples --
#    exhaustive over 1..6 bit address widths
for n_bits in range(1, 7):
    assert mad.decoder_is_onehot(n_bits), n_bits

# 2. minterm() really is an AND of literals: it must reject every address
#    except the one exact bit pattern it targets
assert mad.minterm((0, 1, 1), 3) == 1        # 011 = 3
assert mad.minterm((0, 1, 0), 3) == 0
assert mad.minterm((1, 1, 1), 3) == 0

# 3. the gate-level decoder and the arithmetic flat_address formula agree
#    on EVERY (row, col) pair, for several row/col bit-width combinations
for n_row_bits, n_col_bits in [(1, 1), (2, 2), (2, 3), (3, 2)]:
    assert mad.decoder_matches_flat_index(n_row_bits, n_col_bits), (n_row_bits, n_col_bits)

# 4. flat_address matches the obvious row-major arithmetic directly
assert mad.flat_address(0, 0, n_cols=4) == 0
assert mad.flat_address(1, 2, n_cols=4) == 6
assert mad.flat_address(3, 3, n_cols=4) == 15

# 5. read_cell_voltage retrieves the requested cell's actual value, and
#    a different (row, col) genuinely reads a different physical voltage
#    (this is real data, not a mock -- built from dgs.memory_circuits'
#    own DRAM decay law)
n_row_bits, n_col_bits = 2, 2
n_rows, n_cols = 4, 4
rng = np.random.default_rng(1)
access_time_s = rng.uniform(1e-3, 8e-3, size=(n_rows, n_cols))
V0, R_leak, C_cell = 3.3, 3e12, 30e-15
cell_voltages = mc.dram_cell_decay(V0, access_time_s, R_leak, C_cell)

for row in range(n_rows):
    for col in range(n_cols):
        v = mad.read_cell_voltage(cell_voltages, row, col, n_row_bits, n_col_bits)
        assert v == cell_voltages[row, col]

# two distinct addresses must (generically) read two distinct voltages --
# proves decode selects an actual different cell, not just an alias
v_a = mad.read_cell_voltage(cell_voltages, 2, 1, n_row_bits, n_col_bits)
v_b = mad.read_cell_voltage(cell_voltages, 0, 3, n_row_bits, n_col_bits)
assert v_a != v_b

# 6. read_cell_voltage must reject a decoder that would select the wrong
#    cell -- simulate this by asking for an out-of-range address, which
#    address_to_bits must catch
try:
    mad.read_cell_voltage(cell_voltages, 8, 0, n_row_bits, n_col_bits)
except ValueError:
    pass
else:
    raise AssertionError("should reject an address that doesn't fit n_row_bits")

# 7. GHz clock budget: more clock cycles fit in a fixed refresh interval
#    at a higher clock frequency (a monotonic, physically obvious fact,
#    but worth checking the formula actually respects it)
budget_low = mad.accesses_per_refresh_interval(1.6e9, V0, 1.5, R_leak, C_cell)
budget_high = mad.accesses_per_refresh_interval(6.4e9, V0, 1.5, R_leak, C_cell)
assert budget_high > budget_low
assert abs(budget_high / budget_low - 4.0) < 1e-9   # exactly proportional to clock_hz

print("all dgs.memory_address_decoder tests passed")
