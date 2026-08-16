"""Griffiths Problem 1.50 (all three fields, including the textbook's own
F3 that's BOTH irrotational and solenoidal), the two generic identities
Problems 1.51(d)/1.52(d) cite (curl of a gradient is always zero, Eq 1.44;
divergence of a curl is always zero, Eq 1.46), and a direct numerical
instantiation of Theorem 1's and Theorem 2's (a)=>(c) AND (c)=>(b) chains
-- SymPy for the exact symbolic algebra, PyTorch (3D autograd,
torch.func.jacrev+vmap) for an independent numerical cross-check.

Problem 1.50's three fields span all three regions of the irrotational/
solenoidal Venn diagram:

    F1 = x^2*zhat                     div=0,  curl=-2x*yhat != 0   (PURELY solenoidal)
    F2 = x*xhat+y*yhat+z*zhat (= r)    div=3,  curl=0               (PURELY irrotational)
    F3 = yz*xhat+xz*yhat+xy*zhat       div=0,  curl=0               (BOTH)

F1 has a vector potential (A1=(x^3/3)*yhat) but NO scalar potential (its
curl isn't zero). F2 has a scalar potential (V2=-(x^2+y^2+z^2)/2, using
Theorem 1(d)'s F=-grad(V) sign convention -- Griffiths' own worked
solution uses U2=+(1/2)(x^2+y^2+z^2) with F2=+grad(U2) instead; V2=-U2,
same physics) but NO vector potential (div != 0, and div(curl A)=0 always
would forbid it). F3, satisfying BOTH theorems, has BOTH: U3=xyz
(F3=grad(U3)) and a vector potential A3 (derived below by Griffiths'
own systematic method: set Ax=0, solve two of the three curl equations,
then fix up the third).

Then, for the field that actually satisfies each theorem, both remaining
equivalence-chain links are checked directly, not just asserted:

    Theorem 1(c): oint F2.dl = 0            for a closed loop
    Theorem 1(b): int_a^b F2.dl is the SAME  for two different paths a->b
    Theorem 2(c): oint F1.da = 0            for a closed surface
    Theorem 2(b): int F1.da is the SAME      for two different open
                  surfaces sharing one boundary loop, with matching
                  orientation (the sign-convention subtlety Griffiths'
                  own Problem 1.52 solution flags: "sign change because
                  for surface I da is outward, whereas for surface II it
                  is inward" -- avoided here by using a single consistent
                  "upward" orientation convention for the open-surface
                  comparison, rather than opposite outward normals).
"""

from __future__ import annotations
import os
import re
import subprocess
import time

import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D, curl as sp_curl, divergence as sp_div, gradient as sp_grad

MATLAB_DEFAULT = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"


# ── 1. Problem 1.50: div, curl, and potentials for all three fields ────────

def field_F1_properties() -> dict:
    """F1 = x^2*zhat: div(F1)=0 (solenoidal), curl(F1)=-2x*yhat != 0 (NOT
    irrotational). Vector potential A1=(x^3/3)*yhat verified exactly."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    F1 = x**2 * N.k

    div_F1 = sp.simplify(sp_div(F1))
    curl_F1 = sp_curl(F1)
    if div_F1 != 0:
        raise AssertionError(f"expected div(F1)=0, got {div_F1}")

    A1 = (x**3 / 3) * N.j
    curl_A1 = sp_curl(A1)
    matches = _vector_is_zero(curl_A1 - F1, N)
    if not matches:
        raise AssertionError(f"curl(A1) != F1: curl(A1)={curl_A1}")

    return {"div_F1": div_F1, "curl_F1": curl_F1, "vector_potential_A1": A1,
            "curl_A1_matches_F1": matches}


def field_F2_properties() -> dict:
    """F2 = r = x*xhat+y*yhat+z*zhat: div(F2)=3 (NOT solenoidal),
    curl(F2)=0 (irrotational). Scalar potential
    V2=-(x^2+y^2+z^2)/2 verified exactly (Theorem 1(d)'s F=-grad(V)
    convention; equals -1 times Griffiths' own U2=+(x^2+y^2+z^2)/2)."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    F2 = x * N.i + y * N.j + z * N.k

    div_F2 = sp.simplify(sp_div(F2))
    curl_F2 = sp_curl(F2)
    curl_is_zero = _vector_is_zero(curl_F2, N)
    if not curl_is_zero:
        raise AssertionError(f"expected curl(F2)=0, got {curl_F2}")

    V2 = -(x**2 + y**2 + z**2) / 2
    grad_V2 = sp_grad(V2)
    matches = _vector_is_zero(-grad_V2 - F2, N)
    if not matches:
        raise AssertionError(f"-grad(V2) != F2: grad(V2)={grad_V2}")

    return {"div_F2": div_F2, "curl_F2_is_zero": curl_is_zero, "scalar_potential_V2": V2,
            "minus_grad_V2_matches_F2": matches}


def field_F3_properties() -> dict:
    """F3 = yz*xhat+xz*yhat+xy*zhat: div(F3)=0 AND curl(F3)=0 -- BOTH
    theorems apply, so F3 has BOTH a scalar potential (U3=xyz) and a
    vector potential. The vector potential A3 here is derived by
    Griffiths' own systematic method (set Ax=0, solve two curl equations,
    fix the third) rather than transcribed from the printed page -- vector
    potentials are famously non-unique (Griffiths' own solution says so
    explicitly), so THIS module's A3 need only satisfy curl(A3)=F3
    exactly, which it does."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    F3 = y * z * N.i + x * z * N.j + x * y * N.k

    div_F3 = sp.simplify(sp_div(F3))
    curl_F3 = sp_curl(F3)
    div_is_zero = div_F3 == 0
    curl_is_zero = _vector_is_zero(curl_F3, N)
    if not (div_is_zero and curl_is_zero):
        raise AssertionError(f"expected div(F3)=0 and curl(F3)=0, got div={div_F3}, curl={curl_F3}")

    U3 = x * y * z
    grad_U3 = sp_grad(U3)
    scalar_matches = _vector_is_zero(grad_U3 - F3, N)
    if not scalar_matches:
        raise AssertionError(f"grad(U3) != F3: grad(U3)={grad_U3}")

    # A3, derived by setting Ax=0 and solving the resulting curl equations
    A3 = (x**2 * y / 2) * N.j + (z * (y**2 - x**2) / 2) * N.k
    curl_A3 = sp_curl(A3)
    vector_matches = _vector_is_zero(curl_A3 - F3, N)
    if not vector_matches:
        raise AssertionError(f"curl(A3) != F3: curl(A3)={curl_A3}")

    return {"div_F3": div_F3, "curl_F3_is_zero": curl_is_zero,
            "scalar_potential_U3": U3, "grad_U3_matches_F3": scalar_matches,
            "vector_potential_A3": A3, "curl_A3_matches_F3": vector_matches}


def _vector_is_zero(vec, coord_sys) -> bool:
    return sp.simplify(vec.dot(coord_sys.i)) == 0 and \
           sp.simplify(vec.dot(coord_sys.j)) == 0 and \
           sp.simplify(vec.dot(coord_sys.k)) == 0


# ── 2. The two generic identities Problems 1.51(d)/1.52(d) cite ────────────

def curl_of_gradient_is_zero_generic() -> bool:
    """Eq 1.44: curl(grad f) = 0 for a GENERIC (undefined) scalar function
    f(x,y,z) -- proven symbolically for ANY f, not a specific example,
    the identity Problem 1.51's (d)=>(a) step invokes."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    f = sp.Function('f')(x, y, z)
    result = sp_curl(sp_grad(f))
    is_zero = _vector_is_zero(result, N)
    if not is_zero:
        raise AssertionError(f"curl(grad f) != 0 for generic f: {result}")
    return is_zero


def divergence_of_curl_is_zero_generic() -> bool:
    """Eq 1.46: div(curl A) = 0 for a GENERIC (undefined) vector field A
    -- proven symbolically for ANY A, the identity Problem 1.52's
    (d)=>(a) step invokes."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    Ax, Ay, Az = (sp.Function(name)(x, y, z) for name in ("Ax", "Ay", "Az"))
    A = Ax * N.i + Ay * N.j + Az * N.k
    result = sp.simplify(sp_div(sp_curl(A)))
    is_zero = (result == 0)
    if not is_zero:
        raise AssertionError(f"div(curl A) != 0 for generic A: {result}")
    return is_zero


# ── 3. Theorem 1 (b) and (c), instantiated on F2 (irrotational) ────────────

def _line_integral_F2(path_points, n_per_segment: int = 2000) -> float:
    total = 0.0
    for p0, p1 in zip(path_points[:-1], path_points[1:]):
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        t = np.linspace(0, 1, n_per_segment)
        pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        dl = (p1 - p0) / n_per_segment
        total += float(np.sum(pts @ dl))   # F2(x,y,z) = (x,y,z)
    return total


def closed_loop_line_integral_F2(loop_points, n_per_segment: int = 2000) -> float:
    """Theorem 1(c): oint F2.dl around a closed, genuinely non-planar
    polygon (first point == last point). Predicted EXACTLY zero since F2
    is irrotational."""
    if len(loop_points) < 3 or not np.allclose(loop_points[0], loop_points[-1]):
        raise ValueError("loop_points must have >= 3 points and be closed (first == last)")
    return _line_integral_F2(loop_points, n_per_segment)


def two_path_independence_F2(point_a, point_b, path_I_via, path_II_via,
                             n_per_segment: int = 2000) -> dict:
    """Theorem 1(b): int_a^b F2.dl along TWO DIFFERENT paths from
    point_a to point_b (each routed through its own intermediate
    waypoint) must give the SAME value, since F2 is irrotational."""
    path_I = [point_a, path_I_via, point_b]
    path_II = [point_a, path_II_via, point_b]
    integral_I = _line_integral_F2(path_I, n_per_segment)
    integral_II = _line_integral_F2(path_II, n_per_segment)
    return {"integral_path_I": integral_I, "integral_path_II": integral_II,
            "abs_diff": abs(integral_I - integral_II)}


# ── 4. Theorem 2 (b) and (c), instantiated on F1 (solenoidal) ──────────────

def _F1_field(pts: np.ndarray) -> np.ndarray:
    x = pts[..., 0]
    return np.stack([np.zeros_like(x), np.zeros_like(x), x**2], axis=-1)


def closed_cube_surface_flux_F1(n_per_axis: int = 400) -> float:
    """Theorem 2(c): oint F1.da over the closed surface of the unit cube
    [0,1]^3 (all six faces, consistent OUTWARD normals). Predicted
    EXACTLY zero since F1 is solenoidal."""
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis={n_per_axis} must be >= 2")
    grid = (np.arange(n_per_axis) + 0.5) / n_per_axis
    U, V = np.meshgrid(grid, grid, indexing="ij")
    dA = 1.0 / n_per_axis**2
    faces = [
        (lambda u, v: np.stack([u, v, np.zeros_like(u)], -1), np.array([0.0, 0.0, -1.0])),
        (lambda u, v: np.stack([u, v, np.ones_like(u)], -1), np.array([0.0, 0.0, 1.0])),
        (lambda u, v: np.stack([np.zeros_like(u), u, v], -1), np.array([-1.0, 0.0, 0.0])),
        (lambda u, v: np.stack([np.ones_like(u), u, v], -1), np.array([1.0, 0.0, 0.0])),
        (lambda u, v: np.stack([u, np.zeros_like(u), v], -1), np.array([0.0, -1.0, 0.0])),
        (lambda u, v: np.stack([u, np.ones_like(u), v], -1), np.array([0.0, 1.0, 0.0])),
    ]
    total = 0.0
    for param, normal in faces:
        pts = param(U, V).reshape(-1, 3)
        total += float(np.sum(_F1_field(pts) @ normal) * dA)
    return total


def two_surface_independence_F1(bump_height: float = 0.7, n_per_axis: int = 400) -> dict:
    """Theorem 2(b): int F1.da over TWO DIFFERENT open surfaces sharing
    the SAME boundary loop (the unit-square perimeter at z=0) -- a flat
    patch and a sinusoidal "bump" that returns to z=0 exactly on that
    perimeter -- must give the SAME value, since F1 is solenoidal. Both
    surfaces use the SAME "upward" orientation convention (not opposite
    outward/inward normals, the convention that would flip the sign per
    Griffiths' own Problem 1.52 solution note)."""
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis={n_per_axis} must be >= 2")
    grid = (np.arange(n_per_axis) + 0.5) / n_per_axis
    U, V = np.meshgrid(grid, grid, indexing="ij")
    dA = 1.0 / n_per_axis**2

    def flux_upward(z_func, dzdu_func, dzdv_func):
        pts = np.stack([U, V, z_func(U, V)], axis=-1).reshape(-1, 3)
        normal = np.stack([-dzdu_func(U, V), -dzdv_func(U, V), np.ones_like(U)], axis=-1).reshape(-1, 3)
        return float(np.sum(np.einsum('ij,ij->i', _F1_field(pts), normal)) * dA)

    flat_flux = flux_upward(lambda u, v: np.zeros_like(u), lambda u, v: np.zeros_like(u),
                             lambda u, v: np.zeros_like(u))
    h = bump_height
    bump_flux = flux_upward(
        lambda u, v: h * np.sin(np.pi * u) * np.sin(np.pi * v),
        lambda u, v: h * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v),
        lambda u, v: h * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v))
    return {"flat_surface_flux": flat_flux, "bump_surface_flux": bump_flux,
            "abs_diff": abs(flat_flux - bump_flux)}


# ── 5. PyTorch: 3D autograd cross-check of the div/curl claims ─────────────

def torch_div_curl(field: str, points: np.ndarray) -> dict:
    """Autograd-EXACT divergence and curl (torch.func.jacrev+vmap) of F1,
    F2, or F3 at each row of `points` -- an independent numerical
    cross-check of the SymPy results in Sections 1."""
    import torch
    from torch.func import jacrev, vmap

    def F1(p):
        x = p[0]
        return torch.stack([torch.zeros_like(x), torch.zeros_like(x), x**2])

    def F2(p):
        return p

    def F3(p):
        x, y, z = p[0], p[1], p[2]
        return torch.stack([y * z, x * z, x * y])

    field_fn = {"F1": F1, "F2": F2, "F3": F3}[field]
    pts = torch.as_tensor(np.asarray(points, dtype=np.float64))
    jac = vmap(jacrev(field_fn))(pts)
    div = torch.einsum('nii->n', jac)
    curl = torch.stack([jac[:, 2, 1] - jac[:, 1, 2], jac[:, 0, 2] - jac[:, 2, 0],
                         jac[:, 1, 0] - jac[:, 0, 1]], dim=1)
    return {"divergence": div.detach().numpy(), "curl": curl.detach().numpy()}


# ── 6. MATLAB: independent finite-difference div/curl of F1, F2, F3 ────────
# No Symbolic Math Toolbox on this machine (checked directly in
# dgs.griffiths_1p47_1p48_polyglot -- `syms` raises), so MATLAB's role
# here is the same as there: an independently-coded NUMERICAL check
# (central finite differences), not a second symbolic derivation.

MATLAB_SOURCE_1P50 = r"""
h = 1e-6;
pt = [1.3, -0.7, 2.1];

function v = F1(p)
    v = [0, 0, p(1)^2];
end
function v = F2(p)
    v = p;
end
function v = F3(p)
    v = [p(2)*p(3), p(1)*p(3), p(1)*p(2)];
end

fields = {@F1, @F2, @F3};
for k = 1:3
    f = fields{k};
    fxp = f(pt + [h,0,0]); fxm = f(pt - [h,0,0]);
    fyp = f(pt + [0,h,0]); fym = f(pt - [0,h,0]);
    fzp = f(pt + [0,0,h]); fzm = f(pt - [0,0,h]);

    dFx_dx = (fxp(1)-fxm(1))/(2*h);
    dFy_dy = (fyp(2)-fym(2))/(2*h);
    dFz_dz = (fzp(3)-fzm(3))/(2*h);
    div_val = dFx_dx + dFy_dy + dFz_dz;

    curl_x = (fyp(3)-fym(3))/(2*h) - (fzp(2)-fzm(2))/(2*h);
    curl_y = (fzp(1)-fzm(1))/(2*h) - (fxp(3)-fxm(3))/(2*h);
    curl_z = (fxp(2)-fxm(2))/(2*h) - (fyp(1)-fym(1))/(2*h);

    fprintf('%.10e %.10e %.10e %.10e\n', div_val, curl_x, curl_y, curl_z);
end
"""


def run_matlab_1p50_check(out_dir: str, matlab_path: str = MATLAB_DEFAULT) -> dict:
    """Writes and runs MATLAB_SOURCE_1P50 headless via `matlab -batch`:
    independent finite-difference divergence/curl of F1, F2, F3 at one
    test point, cross-checked against the SymPy-exact and torch-autograd
    results elsewhere in this module."""
    m_path = os.path.join(out_dir, "griffiths_1p50.m")
    with open(m_path, "w") as f:
        f.write(MATLAB_SOURCE_1P50)
    m_path_fwd = m_path.replace("\\", "/")

    result = subprocess.run(
        [matlab_path, "-batch", f"run('{m_path_fwd}')"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"matlab failed (code {result.returncode}): {result.stderr}\n{result.stdout}")

    pattern = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
    rows = [m.groups() for line in result.stdout.splitlines() if (m := pattern.match(line))]
    if len(rows) != 3:
        raise RuntimeError(f"expected 3 output lines from MATLAB, got {len(rows)}: {result.stdout!r}")
    names = ("F1", "F2", "F3")
    out = {}
    for name, row in zip(names, rows):
        div_val, cx, cy, cz = (float(v) for v in row)
        out[name] = {"divergence": div_val, "curl": np.array([cx, cy, cz])}
    return out


# ── 7. Timing benchmark: SymPy vs. torch vs. MATLAB "time of solving" ──────

def benchmark_solve_times(n_torch_points: int = 200, matlab_path: str = MATLAB_DEFAULT,
                          run_matlab: bool = True) -> dict:
    """Wall-clock time for each language/tool to "solve" Problem 1.50:
    SymPy does the FULL exact symbolic job (div, curl, AND verifying every
    potential for all three fields, plus the two generic identities);
    torch does the NUMERICAL autograd-exact div/curl at n_torch_points
    sample points for all three fields; MATLAB does the NUMERICAL
    finite-difference div/curl at one point for all three fields, via a
    subprocess launch of `matlab -batch` (which carries real, substantial
    process-startup overhead -- reported honestly here, not excluded, since
    "time to get an answer from a fresh MATLAB invocation" is a genuine
    real-world cost, not an artifact to hide)."""
    times = {}

    t0 = time.perf_counter()
    field_F1_properties()
    field_F2_properties()
    field_F3_properties()
    curl_of_gradient_is_zero_generic()
    divergence_of_curl_is_zero_generic()
    times["sympy_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import torch  # noqa: F401  -- import timing intentionally excluded (measures the SOLVE, not module load)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(n_torch_points, 3))
    for name in ("F1", "F2", "F3"):
        torch_div_curl(name, pts)
    times["torch_seconds"] = time.perf_counter() - t0

    if run_matlab:
        import tempfile
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            run_matlab_1p50_check(tmp, matlab_path=matlab_path)
        times["matlab_seconds"] = time.perf_counter() - t0

    return times


if __name__ == "__main__":
    print("=== Problem 1.50: all three fields, div/curl/potentials ===")
    p1 = field_F1_properties()
    print(f"  F1 = x^2*zhat: div={p1['div_F1']}, curl={p1['curl_F1']}  (purely solenoidal)")
    print(f"    vector potential A1 = {p1['vector_potential_A1']}, curl(A1)==F1: {p1['curl_A1_matches_F1']}")

    p2 = field_F2_properties()
    print(f"  F2 = r: div={p2['div_F2']}, curl==0: {p2['curl_F2_is_zero']}  (purely irrotational)")
    print(f"    scalar potential V2 = {p2['scalar_potential_V2']}, -grad(V2)==F2: {p2['minus_grad_V2_matches_F2']}")

    p3 = field_F3_properties()
    print(f"  F3 = yz*xhat+xz*yhat+xy*zhat: div==0: {p3['div_F3']==0}, curl==0: {p3['curl_F3_is_zero']}  (BOTH)")
    print(f"    scalar potential U3 = {p3['scalar_potential_U3']}, grad(U3)==F3: {p3['grad_U3_matches_F3']}")
    print(f"    vector potential A3 = {p3['vector_potential_A3']}, curl(A3)==F3: {p3['curl_A3_matches_F3']}")

    print("\n=== The two generic identities (Problems 1.51(d), 1.52(d)) ===")
    print(f"  curl(grad f)=0 for generic f: {curl_of_gradient_is_zero_generic()}")
    print(f"  div(curl A)=0 for generic A:  {divergence_of_curl_is_zero_generic()}")

    print("\n=== PyTorch cross-check ===")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(50, 3))
    t1, t2, t3 = (torch_div_curl(k, pts) for k in ("F1", "F2", "F3"))
    print(f"  F1: max|div| = {np.max(np.abs(t1['divergence'])):.3e}  (expect ~0)")
    print(f"  F2: max|curl| = {np.max(np.abs(t2['curl'])):.3e}  (expect ~0)")
    print(f"  F3: max|div| = {np.max(np.abs(t3['divergence'])):.3e}, max|curl| = {np.max(np.abs(t3['curl'])):.3e}  (expect both ~0)")

    print("\n=== Theorem 1: (c) closed loop, (b) path independence, for F2 ===")
    loop = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 0)]
    print(f"  oint F2.dl (non-planar loop): {closed_loop_line_integral_F2(loop):.3e}  (expect 0)")
    paths = two_path_independence_F2((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0))
    print(f"  path I: {paths['integral_path_I']:.6f}, path II: {paths['integral_path_II']:.6f}, "
          f"diff: {paths['abs_diff']:.3e}")

    print("\n=== Theorem 2: (c) closed surface, (b) surface independence, for F1 ===")
    print(f"  oint F1.da (unit cube, outward normals): {closed_cube_surface_flux_F1():.3e}  (expect 0)")
    surfaces = two_surface_independence_F1()
    print(f"  flat surface: {surfaces['flat_surface_flux']:.6f}, bump surface: {surfaces['bump_surface_flux']:.6f}, "
          f"diff: {surfaces['abs_diff']:.3e}")

    print("\n=== MATLAB: independent finite-difference div/curl cross-check ===")
    if os.path.exists(MATLAB_DEFAULT):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            matlab_result = run_matlab_1p50_check(tmp)
        for name in ("F1", "F2", "F3"):
            r = matlab_result[name]
            print(f"  {name}: div={r['divergence']:.6f}, curl={r['curl']}")
    else:
        print(f"  MATLAB not found at {MATLAB_DEFAULT} -- skipped")

    print("\n=== Timing benchmark: time to 'solve' Problem 1.50, per tool ===")
    times = benchmark_solve_times(run_matlab=os.path.exists(MATLAB_DEFAULT))
    for tool, seconds in times.items():
        print(f"  {tool:>16}: {seconds*1000:>9.2f} ms")

    print("\nF1, F2, F3 span the three regions of the irrotational/solenoidal Venn diagram;")
    print("each theorem's full (a)-(b)-(c)-(d) chain is checked, not just the closed-form (d),")
    print("on the field that actually satisfies it. The timing numbers above are for THIS")
    print("problem size specifically -- SymPy's symbolic overhead and MATLAB's process-launch")
    print("cost dominate here in a way that would NOT generalize to a much larger problem.")
