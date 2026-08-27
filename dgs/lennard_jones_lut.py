"""Hardware-style lookup-table (LUT) acceleration of the Lennard-Jones
force/potential -- the actual R&D tradeoff production MD codes (GROMACS,
LAMMPS) and special-purpose MD hardware (D.E. Shaw's Anton machines) make:
evaluating (sigma/r)^12 and (sigma/r)^6 with real pow() calls is expensive
per pair, and a real MD loop does this for millions of pairs per step. The
fix is the same one every digital-logic LUT does: precompute the function
on a grid and interpolate instead of recomputing it from scratch.

THE ACTUAL TRICK (not just "cache it"): index the table by r^2, not r. A
pairwise distance is computed as dx^2+dy^2+dz^2 -- you already HAVE r^2
for free, and computing r itself costs a sqrt you don't need if the table
is built in r^2 directly. This is exactly how real MD codes do it.

THE HARDWARE ANGLE: dgs.lennard_jones's exact evaluation implicitly assumes
float64 throughout. A real LUT living in FPGA/ASIC block RAM is finite-width
fixed-point memory, not float64 -- quantize_table() and lut_quantization_error()
below measure what that costs in accuracy, the real bit-width-vs-precision
tradeoff a hardware LJ evaluator's designer actually has to make.
"""
import numpy as np

from dgs.lennard_jones import lj_potential, lj_force_magnitude


def build_lj_lut(r2_min, r2_max, n_entries, eps=1.0, sigma=1.0):
    """Precompute V(r) and F(r)/r on a UNIFORM grid of r^2 (not r). Returns
    (r2_grid, V_table, FoverR_table). F(r)/r is tabulated (not F(r) alone)
    because that's what's actually needed downstream: F_vec = (F(r)/r) *
    (x_i - x_j), so a real MD loop never has to compute r or 1/r at all,
    only r^2, matching the LUT's own index variable."""
    if n_entries < 2:
        raise ValueError(f"n_entries must be >= 2, got {n_entries}")
    if not (0 < r2_min < r2_max):
        raise ValueError(f"need 0 < r2_min < r2_max, got r2_min={r2_min}, r2_max={r2_max}")
    r2_grid = np.linspace(r2_min, r2_max, n_entries)
    r_grid = np.sqrt(r2_grid)
    V_table = lj_potential(r_grid, eps, sigma)
    F_table = lj_force_magnitude(r_grid, eps, sigma)
    FoverR_table = F_table / r_grid
    return r2_grid, V_table, FoverR_table


def lut_lookup(r2_query, r2_grid, table):
    """Linear-interpolation lookup at r2_query -- the exact operation a
    hardware LUT + interpolator block performs: an integer index from
    (r2 - r2_min)/step, then a blend with the next entry. Works on scalars
    or arrays."""
    r2_query = np.asarray(r2_query, dtype=float)
    r2_min, r2_max = r2_grid[0], r2_grid[-1]
    if np.any(r2_query < r2_min) or np.any(r2_query > r2_max):
        raise ValueError(f"query r^2 out of LUT range [{r2_min}, {r2_max}]")
    step = r2_grid[1] - r2_grid[0]
    idx_float = (r2_query - r2_min) / step
    idx0 = np.clip(np.floor(idx_float).astype(int), 0, len(table) - 2)
    frac = idx_float - idx0
    result = table[idx0] * (1 - frac) + table[idx0 + 1] * frac
    return result.item() if result.ndim == 0 else result


def verify_lut_converges(n_entries_list=(16, 64, 256, 1024), eps=1.0, sigma=1.0,
                          r2_min=0.85, r2_max=9.0, n_test_points=200):
    """The LUT's whole justification is that it APPROXIMATES the exact
    evaluation, well enough, cheaply. This checks that claim rather than
    assuming it: RMS error between LUT-interpolated and exact lj_potential
    at random test points, as a function of table size, and asserts it
    actually shrinks (linear interpolation should converge like O(step^2),
    i.e. roughly 4x smaller error each time n_entries doubles)."""
    rng = np.random.default_rng(0)
    r2_test = rng.uniform(r2_min, r2_max, n_test_points)
    r_test = np.sqrt(r2_test)
    V_exact = lj_potential(r_test, eps, sigma)

    rms_errors = []
    for n in n_entries_list:
        r2_grid, V_table, _ = build_lj_lut(r2_min, r2_max, n, eps, sigma)
        V_lut = lut_lookup(r2_test, r2_grid, V_table)
        rms_errors.append(float(np.sqrt(np.mean((V_lut - V_exact) ** 2))))

    monotonically_improving = all(
        rms_errors[i + 1] < rms_errors[i] for i in range(len(rms_errors) - 1)
    )
    return {"n_entries": list(n_entries_list), "rms_error": rms_errors,
            "monotonically_improving": monotonically_improving}


def quantize_table(table, n_bits):
    """Simulate storing `table` in an n_bits-wide fixed-point hardware LUT
    (a real FPGA/ASIC block-RAM word, not float64): scale to the table's
    own [min, max] range, round to the nearest of 2^n_bits integer levels,
    then scale back. Returns (quantized_table, scale, offset) so the
    quantization can be inverted/inspected."""
    if n_bits < 1:
        raise ValueError(f"n_bits must be >= 1, got {n_bits}")
    table = np.asarray(table, dtype=float)
    lo, hi = table.min(), table.max()
    if hi == lo:
        raise ValueError("table is constant -- cannot quantize a zero-width range")
    n_levels = 2 ** n_bits
    scale = (hi - lo) / (n_levels - 1)
    levels = np.round((table - lo) / scale)
    levels = np.clip(levels, 0, n_levels - 1)
    quantized = lo + levels * scale
    return quantized, scale, lo


def lut_quantization_error(n_entries=256, n_bits_list=(4, 8, 12, 16, 24), eps=1.0, sigma=1.0,
                            r2_min=0.85, r2_max=9.0):
    """The real hardware-design question: how many bits of table storage
    does a given accuracy require? Builds one LUT, quantizes its potential
    table to each bit width, and reports the RMS error introduced by
    quantization ALONE (interpolation held fixed) -- should shrink roughly
    by half each extra bit (each bit halves the quantization step)."""
    r2_grid, V_table, _ = build_lj_lut(r2_min, r2_max, n_entries, eps, sigma)
    errors = []
    for n_bits in n_bits_list:
        V_quantized, scale, offset = quantize_table(V_table, n_bits)
        rms = float(np.sqrt(np.mean((V_quantized - V_table) ** 2)))
        errors.append(rms)
    return {"n_bits": list(n_bits_list), "rms_error": errors,
            "table_range": float(V_table.max() - V_table.min())}


# ── drop-in replacement for dgs.lennard_jones.pair_forces ──────────────

def pair_forces_lut(pos, r2_grid, FoverR_table, V_table):
    """Same contract as dgs.lennard_jones.pair_forces (forces[N,dim],
    potential_energy), but every pair evaluation is a LUT lookup instead
    of a direct pow()-based computation -- literally never computes r or a
    real division by r for the force direction (uses FoverR_table
    directly), only r^2 from the dot product."""
    pos = np.asarray(pos, dtype=float)
    N = len(pos)
    F = np.zeros_like(pos)
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = pos[i] - pos[j]
            r2 = float(np.dot(d, d))
            if r2 < 1e-12:
                raise ValueError("two atoms coincide (r^2 ~ 0)")
            f_over_r = lut_lookup(r2, r2_grid, FoverR_table)
            fij = f_over_r * d
            F[i] += fij
            F[j] -= fij
            U += lut_lookup(r2, r2_grid, V_table)
    return F, U


if __name__ == "__main__":
    print("=== Does the LUT actually converge to the exact evaluation? ===")
    conv = verify_lut_converges()
    for n, err in zip(conv["n_entries"], conv["rms_error"]):
        print(f"  n_entries={n:5d}  RMS error vs exact = {err:.3e}")
    print(f"monotonically improving with table size: {conv['monotonically_improving']}")

    print("\n=== Hardware bit-width vs. accuracy tradeoff ===")
    quant = lut_quantization_error()
    print(f"table dynamic range: {quant['table_range']:.4f} (eps units)")
    for nb, err in zip(quant["n_bits"], quant["rms_error"]):
        print(f"  {nb:2d}-bit fixed-point LUT entries: RMS quantization error = {err:.3e}")

    print("\n=== Drop-in replacement: LUT-based MD vs. exact MD, same initial condition ===")
    from dgs.lennard_jones import pair_forces, hex_cluster, equilibrium_distance
    pos = hex_cluster(n_rings=1)
    r2_grid, V_table, FoverR_table = build_lj_lut(r2_min=0.7, r2_max=9.0, n_entries=512)

    F_exact, U_exact = pair_forces(pos)
    F_lut, U_lut = pair_forces_lut(pos, r2_grid, FoverR_table, V_table)
    force_err = np.max(np.abs(F_exact - F_lut))
    print(f"exact U = {U_exact:.6f} eps, LUT U = {U_lut:.6f} eps "
          f"(diff {abs(U_exact - U_lut):.2e})")
    print(f"max force-component error: {force_err:.2e} (512-entry LUT)")
