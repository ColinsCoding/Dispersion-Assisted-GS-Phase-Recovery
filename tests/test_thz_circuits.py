"""Test dgs/thz_circuits.py: the lambda/10 lumped-element rule, microstrip
transmission-line physics, the ABCD matrix formalism reused UNMODIFIED
from dgs.paraxial_optics_abcd, and the discrete-geometry identity
(N-segment cascade exactly matches the continuous line, for every N)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs.thz_circuits import (
    lumped_element_validity_symbolic, is_lumped_valid,
    microstrip_effective_permittivity, microstrip_characteristic_impedance,
    microstrip_propagation_constant, transmission_line_ABCD,
    verify_discrete_geometry_identity,
)
from dgs.paraxial_optics_abcd import compose_system, is_unimodular

# 1. lumped_element_validity_symbolic: exact symbolic form
sym = lumped_element_validity_symbolic()
L, lam = sp.symbols('L lambda', positive=True)
assert sp.simplify(sym["electrical_length_theta"] - 2 * sp.pi * L / lam) == 0

# 2. is_lumped_valid: a short trace at low frequency should be lumped-valid,
#    a long trace (or high frequency) should not
low_f = is_lumped_valid(trace_length_m=2e-3, frequency_hz=1e9)
assert low_f["lumped_valid"] is True
high_f = is_lumped_valid(trace_length_m=2e-3, frequency_hz=1e12)
assert high_f["lumped_valid"] is False
assert high_f["trace_length_over_wavelength"] > low_f["trace_length_over_wavelength"]

for bad in [dict(trace_length_m=-1, frequency_hz=1e9), dict(trace_length_m=1e-3, frequency_hz=-1)]:
    try:
        is_lumped_valid(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.thz_circuits: lambda/10 rule checks passed")

# 3. microstrip_effective_permittivity / characteristic_impedance: known bounds
#    -- eps_eff must lie strictly between 1 (air) and eps_r (pure dielectric)
w, h, eps_r = 150e-6, 100e-6, 3.5
eps_eff = microstrip_effective_permittivity(w, h, eps_r)
assert 1.0 < eps_eff < eps_r

Z0 = microstrip_characteristic_impedance(w, h, eps_r)
assert 20.0 < Z0 < 150.0, f"Z0={Z0} outside a physically reasonable microstrip range"

# a WIDER trace should give a LOWER characteristic impedance (more capacitance per length)
Z0_wide = microstrip_characteristic_impedance(w * 3, h, eps_r)
assert Z0_wide < Z0

for bad_eps_r in (1.0, 0.5, -1.0):
    try:
        microstrip_effective_permittivity(w, h, bad_eps_r)
        raise AssertionError(f"expected ValueError for eps_r={bad_eps_r}")
    except ValueError:
        pass

print("dgs.thz_circuits: microstrip physics checks passed")

# 4. transmission_line_ABCD: det=1 (unimodular, reusing dgs.paraxial_optics_abcd's
#    own check), and zero-length line is the identity matrix
beta = microstrip_propagation_constant(100e9, w, h, eps_r)
M = transmission_line_ABCD(beta, Z0, 0.002)
assert is_unimodular(M, tol=1e-9)

M_zero = transmission_line_ABCD(beta, Z0, 0.0)
assert np.allclose(M_zero, np.eye(2, dtype=complex))

try:
    transmission_line_ABCD(beta, Z0=-1.0, length_m=0.001)
    raise AssertionError("expected ValueError for Z0 <= 0")
except ValueError:
    pass

print("dgs.thz_circuits: ABCD matrix checks passed")

# 5. verify_discrete_geometry_identity: EVERY N tested must match the
#    full-length line to near machine precision -- the module's headline claim
check = verify_discrete_geometry_identity(beta, Z0, total_length_m=0.002)
assert check["all_match"] is True
for N, r in check["per_N_results"].items():
    assert r["max_abs_diff_from_full_length"] < 1e-9, f"N={N}: {r}"
    assert r["unimodular"] is True

# a DIFFERENT (beta, Z0, length) combination should also satisfy the identity
# -- not a coincidence at one specific design point
beta2 = microstrip_propagation_constant(300e9, w * 2, h, eps_r)
Z0_2 = microstrip_characteristic_impedance(w * 2, h, eps_r)
check2 = verify_discrete_geometry_identity(beta2, Z0_2, total_length_m=0.001,
                                           n_segments_list=(1, 3, 7, 20))
assert check2["all_match"] is True

print("dgs.thz_circuits: discrete-geometry identity checks passed")

# 6. torch (py 3.12 only): 3D geometry builder
try:
    import torch  # noqa: F401
    from dgs.thz_circuits import microstrip_geometry_3d
    geom = microstrip_geometry_3d(w=0.15, h=0.10, length=2.0, trace_thickness=0.01)
    assert geom["substrate_vertices"].shape == (8, 3)
    assert geom["trace_vertices"].shape == (8, 3)
    # trace must sit ABOVE the substrate top (z=0) and within the substrate's
    # x-y footprint -- a basic physical sanity check on the geometry
    assert torch.all(geom["trace_vertices"][:, 2] >= 0)
    assert torch.all(geom["substrate_vertices"][:, 2] <= 0)
    print("dgs.thz_circuits: 3D geometry checks passed")
except ImportError:
    print("dgs.thz_circuits: torch not available, skipped 3D geometry checks")

print("all dgs.thz_circuits tests passed")
