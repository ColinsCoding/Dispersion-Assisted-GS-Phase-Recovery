"""THz circuit design: when a "circuit" stops being lumped elements and
becomes a distributed transmission line, and the SAME 2x2 ABCD
ray-transfer-matrix formalism dgs.paraxial_optics_abcd already uses for
ray optics, reused unmodified for a microstrip transmission line's
2-port network -- ray optics and RF network theory are the same linear
algebra, made literal by calling the exact same functions
(compose_system, is_unimodular) on both.

THE LUMPED/DISTRIBUTED BOUNDARY: a circuit element behaves as a lumped
(instantaneous, Kirchhoff's-laws) component only while its physical size
is much smaller than the signal wavelength -- the standard engineering
rule of thumb is electrical length < lambda/10. At THz frequencies
(100 GHz-10 THz), a wavelength in a typical dielectric is only
hundreds of microns to a few mm, so ORDINARY circuit-board trace lengths
(often already mm-scale at GHz) routinely violate lambda/10 and MUST be
treated as distributed transmission lines, not lumped RLC -- checked
directly below for representative THz frequencies and trace lengths, not
assumed.

MICROSTRIP TRANSMISSION LINE (quasi-TEM, Hammerstad-Jensen
approximation): characteristic impedance Z0 and effective permittivity
eps_eff from trace width w, substrate height h, and substrate relative
permittivity eps_r -- the standard planar-circuit building block used in
real THz integrated circuits.

THE DISCRETE-GEOMETRY IDENTITY (checked, not assumed): physically slicing
a transmission line of length L into N discrete segments and cascading
each segment's ABCD matrix gives EXACTLY (to floating-point precision,
for every N, not just as N gets large) the same ABCD matrix as the single
continuous line -- because phase accumulates additively along the line
(cos(N*theta) built from N copies of a rotation-like matrix is exactly
cos of the total angle), not because discretization happens to converge.
"""

from __future__ import annotations
import numpy as np
import sympy as sp

from dgs.paraxial_optics_abcd import compose_system, is_unimodular

C_LIGHT = 299792458.0   # m/s


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Lumped vs. distributed: the lambda/10 rule, symbolic and numeric ────

def lumped_element_validity_symbolic():
    """Symbolic statement of the lambda/10 rule: electrical length
    theta = 2*pi*L/lambda; "lumped" requires theta << 2*pi/10, i.e.
    L << lambda/10. Returns the SymPy expressions with init_printing
    already enabled, so a caller in a notebook sees them typeset."""
    sp.init_printing()
    L, lam = sp.symbols('L lambda', positive=True)
    theta = 2 * sp.pi * L / lam
    lumped_condition = sp.Lt(L, lam / 10)
    return {"electrical_length_theta": theta, "lumped_condition": lumped_condition}


def is_lumped_valid(trace_length_m: float, frequency_hz: float, eps_eff: float = 1.0,
                    c: float = C_LIGHT) -> dict:
    """CHECKED: given a physical trace length, a signal frequency, and the
    medium's effective permittivity (1.0 = vacuum/air), computes the
    wavelength, the electrical length in wavelengths, and whether the
    lambda/10 lumped-element rule of thumb is satisfied."""
    _validate_positive(trace_length_m=trace_length_m, frequency_hz=frequency_hz, eps_eff=eps_eff)
    wavelength = c / (frequency_hz * np.sqrt(eps_eff))
    length_over_wavelength = trace_length_m / wavelength
    return {"wavelength_m": wavelength, "trace_length_over_wavelength": length_over_wavelength,
            "lumped_valid": bool(length_over_wavelength < 0.1)}


# ── 2. Microstrip transmission line (quasi-TEM, Hammerstad-Jensen) ─────────

def microstrip_effective_permittivity(w: float, h: float, eps_r: float) -> float:
    """eps_eff = (eps_r+1)/2 + (eps_r-1)/2 * 1/sqrt(1+12h/w) -- the
    standard quasi-TEM approximation for w/h >= 1 (wide trace relative to
    substrate height, the usual THz planar-circuit regime)."""
    _validate_positive(w=w, h=h, eps_r=eps_r)
    if eps_r <= 1.0:
        raise ValueError(f"eps_r must be > 1 (a real dielectric), got {eps_r}")
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 / np.sqrt(1 + 12 * h / w)


def microstrip_characteristic_impedance(w: float, h: float, eps_r: float) -> float:
    """Z0 = 120*pi / (sqrt(eps_eff) * (w/h + 1.393 + 0.667*ln(w/h + 1.444)))
    -- Hammerstad-Jensen, valid for w/h >= 1."""
    _validate_positive(w=w, h=h, eps_r=eps_r)
    eps_eff = microstrip_effective_permittivity(w, h, eps_r)
    wh = w / h
    return 120 * np.pi / (np.sqrt(eps_eff) * (wh + 1.393 + 0.667 * np.log(wh + 1.444)))


def microstrip_propagation_constant(frequency_hz: float, w: float, h: float, eps_r: float,
                                    c: float = C_LIGHT) -> float:
    """beta = omega*sqrt(eps_eff)/c."""
    _validate_positive(frequency_hz=frequency_hz, w=w, h=h, eps_r=eps_r)
    eps_eff = microstrip_effective_permittivity(w, h, eps_r)
    return 2 * np.pi * frequency_hz * np.sqrt(eps_eff) / c


# ── 3. The transmission-line ABCD matrix, reusing dgs.paraxial_optics_abcd ─

def transmission_line_ABCD(beta: float, Z0: float, length_m: float) -> np.ndarray:
    """The standard lossless transmission-line 2-port ABCD matrix:
        [[cos(beta*d),      i*Z0*sin(beta*d)],
         [i*sin(beta*d)/Z0, cos(beta*d)     ]]
    -- COMPLEX-valued (unlike dgs.paraxial_optics_abcd's real ray-optics
    matrices), since it represents a phasor/impedance relationship, not a
    real-valued ray height and angle. Passed straight into
    dgs.paraxial_optics_abcd.compose_system and is_unimodular below --
    those functions don't know or care that this matrix came from RF
    network theory instead of ray optics."""
    if Z0 <= 0 or length_m < 0:
        raise ValueError(f"Z0 must be > 0 and length_m >= 0, got Z0={Z0}, length_m={length_m}")
    bd = beta * length_m
    return np.array([[np.cos(bd), 1j * Z0 * np.sin(bd)],
                      [1j * np.sin(bd) / Z0, np.cos(bd)]], dtype=complex)


def verify_discrete_geometry_identity(beta: float, Z0: float, total_length_m: float,
                                      n_segments_list=(1, 2, 5, 10, 50)) -> dict:
    """CHECKED, not assumed: physically slicing the line into N segments
    and cascading their ABCD matrices via
    dgs.paraxial_optics_abcd.compose_system gives EXACTLY the same matrix
    as the single full-length line, for EVERY N tested -- not a limit
    that's approached, an algebraic identity that holds from N=1."""
    M_full = transmission_line_ABCD(beta, Z0, total_length_m)
    results = {}
    for N in n_segments_list:
        segment = transmission_line_ABCD(beta, Z0, total_length_m / N)
        M_cascade = compose_system(*([segment] * N))
        max_diff = float(np.max(np.abs(M_cascade - M_full)))
        results[N] = {"max_abs_diff_from_full_length": max_diff,
                      "unimodular": bool(is_unimodular(M_cascade, tol=1e-6))}
    return {"M_full": M_full, "per_N_results": results,
            "all_match": all(r["max_abs_diff_from_full_length"] < 1e-9 for r in results.values())}


# ── 4. 3D microstrip geometry, built with torch tensors ────────────────────

def microstrip_geometry_3d(w: float, h: float, length: float, trace_thickness: float,
                           n_points: int = 2):
    """Builds the 3-D box geometry (8 corner vertices each) of a
    microstrip's conducting trace and dielectric substrate, as torch
    tensors -- the physical "discrete geometry" a THz circuit designer
    actually lays out, not just an abstract ABCD matrix."""
    import torch
    _validate_positive(w=w, h=h, length=length, trace_thickness=trace_thickness)

    def box_vertices(x0, x1, y0, y1, z0, z1):
        xs = torch.tensor([x0, x1])
        ys = torch.tensor([y0, y1])
        zs = torch.tensor([z0, z1])
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    substrate = box_vertices(0, length, -w * 3, w * 3, -h, 0)
    trace = box_vertices(0, length, -w / 2, w / 2, 0, trace_thickness)
    return {"substrate_vertices": substrate, "trace_vertices": trace}


if __name__ == "__main__":
    print("=== 1. Lumped vs. distributed: the lambda/10 rule ===")
    sym = lumped_element_validity_symbolic()
    print(f"  electrical length theta = {sym['electrical_length_theta']}")
    print(f"  lumped-valid condition:  {sym['lumped_condition']}")

    print("\n  A 2mm trace at increasing frequency (vacuum wavelength):")
    for f in (1e9, 100e9, 300e9, 1e12, 3e12):
        check = is_lumped_valid(trace_length_m=2e-3, frequency_hz=f)
        print(f"    f={f:>8.2e} Hz: lambda={check['wavelength_m']*1e3:.4f} mm, "
              f"L/lambda={check['trace_length_over_wavelength']:.4f}, "
              f"lumped valid: {check['lumped_valid']}")

    print("\n=== 2. Microstrip transmission line (100 GHz THz-adjacent design) ===")
    w, h, eps_r = 150e-6, 100e-6, 3.5   # 150um trace, 100um substrate, eps_r=3.5 (e.g. quartz-like)
    eps_eff = microstrip_effective_permittivity(w, h, eps_r)
    Z0 = microstrip_characteristic_impedance(w, h, eps_r)
    f_design = 100e9
    beta = microstrip_propagation_constant(f_design, w, h, eps_r)
    print(f"  w={w*1e6:.0f}um, h={h*1e6:.0f}um, eps_r={eps_r}")
    print(f"  eps_eff = {eps_eff:.4f}, Z0 = {Z0:.2f} ohm, beta @ {f_design/1e9:.0f}GHz = {beta:.2f} rad/m")

    print("\n=== 3. ABCD matrix: reusing dgs.paraxial_optics_abcd unmodified ===")
    L_line = 0.002   # 2mm line
    M = transmission_line_ABCD(beta, Z0, L_line)
    print(f"  M = \n{M}")
    print(f"  is_unimodular (det=1, from ray-optics code, no changes): {is_unimodular(M, tol=1e-6)}")

    print("\n=== 4. Discrete-geometry identity: N segments == 1 continuous line ===")
    check = verify_discrete_geometry_identity(beta, Z0, L_line)
    for N, r in check["per_N_results"].items():
        print(f"  N={N:>3}: max diff from full-length ABCD = {r['max_abs_diff_from_full_length']:.3e}, "
              f"unimodular: {r['unimodular']}")
    print(f"  all N match exactly: {check['all_match']}")

    print("\n=== 5. 3D geometry (torch tensors) ===")
    try:
        import torch
        geom = microstrip_geometry_3d(w=w * 1e3, h=h * 1e3, length=L_line * 1e3, trace_thickness=0.005)
        print(f"  substrate vertices shape: {geom['substrate_vertices'].shape}")
        print(f"  trace vertices shape:     {geom['trace_vertices'].shape}")
    except ImportError:
        print("  torch not available in this interpreter -- run under py -3.12")

    print("\nSame 2x2 ABCD matrix machinery (compose_system, is_unimodular) verifies")
    print("a THz transmission line exactly as it verifies a telescope -- the physics")
    print("differs, the linear algebra doesn't, and discretizing the line's physical")
    print("geometry into any number of segments reproduces the continuous answer exactly.")
