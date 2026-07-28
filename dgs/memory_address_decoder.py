"""How a machine finds ONE cell in a 2D grid of physical memory.

An address decoder is not an abstraction bolted on top of Boolean algebra --
it IS Boolean algebra, in gate form. A real DRAM/SRAM chip splits an address
into a ROW half and a COLUMN half; each half drives an n-to-2^n decoder, a
bank of AND gates where output line j is literally the Boolean minterm for
binary pattern j (exactly the minterm construction dgs.boolean_algebra's
TruthTable enumerates). The cell that actually gets read is the AND of
(row line r is HIGH) and (column line c is HIGH) -- one wordline crossing
one bitline.

That gate-level picture and the arithmetic flat_index_from_multi_index
formula in dgs.torch.cuda_pointer_arithmetic (row*n_cols + col, row-major
strides) are two descriptions of the SAME thing -- this module builds the
gate-level decoder from scratch and cross-checks it against the arithmetic
one, then wires the result into dgs.memory_circuits' actual physical DRAM
cell model: decoding the wrong address reads a different cell's real,
independently-decaying voltage, not an abstract error code.
"""

import numpy as np

from dgs.torch.cuda_pointer_arithmetic import flat_index_from_multi_index
from dgs import memory_circuits as mc


def address_to_bits(value, n_bits):
    """value, as an n_bits-wide tuple of 0/1, MSB first."""
    if not (0 <= value < 2 ** n_bits):
        raise ValueError(f"value {value} does not fit in {n_bits} bits")
    return tuple((value >> (n_bits - 1 - k)) & 1 for k in range(n_bits))


def minterm(bits, j):
    """The literal AND-of-literals for minterm j of len(bits) Boolean
    variables: HIGH iff `bits` exactly equals j's binary pattern. This is
    the gate-level implementation of one output line of an n-to-2^n
    decoder -- not a lookup, an actual Boolean AND of true/complemented
    address lines."""
    n = len(bits)
    j_bits = address_to_bits(j, n)
    return int(all(b == jb for b, jb in zip(bits, j_bits)))


def decoder_outputs(address_bits):
    """All 2^n output lines of an n-to-2^n decoder. Exactly one is HIGH by
    construction (the minterms of n variables partition the space)."""
    n = len(address_bits)
    return [minterm(address_bits, j) for j in range(2 ** n)]


def decoder_is_onehot(n_bits):
    """Exhaustive check (every possible address, not a sample): the
    decoder built from minterm() selects exactly one output line for
    every address in an n_bits-wide space."""
    for value in range(2 ** n_bits):
        outputs = decoder_outputs(address_to_bits(value, n_bits))
        if sum(outputs) != 1 or outputs[value] != 1:
            return False
    return True


def decode_cell(row, col, n_row_bits, n_col_bits):
    """The physical cell-select grid: AND(row decoder's row-th output,
    column decoder's col-th output) for every (r, c) pair -- exactly one
    entry is 1. Returns the full 2D select grid."""
    row_lines = decoder_outputs(address_to_bits(row, n_row_bits))
    col_lines = decoder_outputs(address_to_bits(col, n_col_bits))
    n_rows, n_cols = 2 ** n_row_bits, 2 ** n_col_bits
    return [[row_lines[r] & col_lines[c] for c in range(n_cols)] for r in range(n_rows)]


def flat_address(row, col, n_cols):
    """The arithmetic-side address for the SAME (row, col): reuses
    flat_index_from_multi_index verbatim, treating (row, col) as a 2D
    index with row-major strides (n_cols, 1) -- the same convention as a
    C/NumPy 2D array, and the same formula CUDA kernels use for a flat
    thread/element index."""
    return flat_index_from_multi_index((row, col), (n_cols, 1))


def decoder_matches_flat_index(n_row_bits, n_col_bits):
    """Exhaustive cross-check, EVERY (row, col) pair: the single cell the
    gate-level AND-decoder selects has the same linear position, in
    row-major scan order, as flat_address computes arithmetically."""
    n_rows, n_cols = 2 ** n_row_bits, 2 ** n_col_bits
    for row in range(n_rows):
        for col in range(n_cols):
            grid = decode_cell(row, col, n_row_bits, n_col_bits)
            flat_from_grid = [
                r * n_cols + c
                for r in range(n_rows) for c in range(n_cols)
                if grid[r][c] == 1
            ]
            if flat_from_grid != [flat_address(row, col, n_cols)]:
                return False
    return True


def read_cell_voltage(cell_voltages, row, col, n_row_bits, n_col_bits):
    """Decode (row, col) with the gate-level AND-decoder above, confirm
    exactly one physical cell is selected, and return THAT cell's actual
    voltage out of cell_voltages (an (n_rows, n_cols) array of
    independently-decaying DRAM cells, e.g. from mc.dram_cell_decay).
    Decoding a different address reads a different cell's real physical
    state -- not a labeled error, an actually wrong voltage."""
    grid = decode_cell(row, col, n_row_bits, n_col_bits)
    n_rows, n_cols = len(grid), len(grid[0])
    selected = [(r, c) for r in range(n_rows) for c in range(n_cols) if grid[r][c] == 1]
    if selected != [(row, col)]:
        raise AssertionError(f"decoder should select exactly ({row},{col}), got {selected}")
    return cell_voltages[row, col]


def accesses_per_refresh_interval(clock_hz, V0, V_threshold, R_leak, C_cell):
    """How many address-decode/access cycles fit inside one DRAM refresh
    interval, at a real clock frequency (GHz-scale) -- ties the Boolean
    decoder's per-cycle operation to the physical retention-time budget
    from dgs.memory_circuits.dram_refresh_interval."""
    if clock_hz <= 0:
        raise ValueError("clock_hz must be positive")
    refresh_s = mc.dram_refresh_interval(V0, V_threshold, R_leak, C_cell)
    return refresh_s * clock_hz


if __name__ == "__main__":
    n_row_bits, n_col_bits = 2, 2   # a 4x4 toy memory array
    n_rows, n_cols = 2 ** n_row_bits, 2 ** n_col_bits

    print("=== Decoder is one-hot for every address (exhaustive) ===")
    print(f"  {n_row_bits}-bit row decoder one-hot: {decoder_is_onehot(n_row_bits)}")
    print(f"  {n_col_bits}-bit column decoder one-hot: {decoder_is_onehot(n_col_bits)}")

    print("\n=== Gate-level decoder matches the arithmetic flat-index formula (exhaustive) ===")
    print(f"  decode_cell(row,col) selects the same cell flat_address(row,col) computes: "
          f"{decoder_matches_flat_index(n_row_bits, n_col_bits)}")

    print("\n=== Reading the RIGHT cell vs. a WRONG cell out of real, decaying DRAM voltages ===")
    rng = np.random.default_rng(0)
    V0 = 3.3
    access_time_s = rng.uniform(1e-3, 8e-3, size=(n_rows, n_cols))  # each cell read at a different moment
    R_leak, C_cell = 3e12, 30e-15
    cell_voltages = mc.dram_cell_decay(V0, access_time_s, R_leak, C_cell)

    target_row, target_col = 2, 1
    correct = read_cell_voltage(cell_voltages, target_row, target_col, n_row_bits, n_col_bits)
    wrong_row, wrong_col = 0, 3
    actually_read_if_miswired = cell_voltages[wrong_row, wrong_col]
    print(f"  requested (row={target_row}, col={target_col}): voltage = {correct:.4f} V")
    print(f"  a decoder bug landing on (row={wrong_row}, col={wrong_col}) instead would read "
          f"{actually_read_if_miswired:.4f} V -- a different physical cell's real charge, not a labeled error")

    print("\n=== GHz clock budget: how many accesses fit before a refresh is required ===")
    for clock_ghz in (1.6, 3.2, 6.4):
        n_accesses = accesses_per_refresh_interval(clock_ghz * 1e9, V0, 1.5, R_leak, C_cell)
        print(f"  clock={clock_ghz} GHz: {n_accesses:,.0f} address-decode cycles fit in one refresh interval")
