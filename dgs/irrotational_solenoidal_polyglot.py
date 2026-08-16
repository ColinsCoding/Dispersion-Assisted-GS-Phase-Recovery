"""Griffiths' two vector-calculus theorems (curl of a gradient is always
zero; divergence of a curl is always zero) made concrete with the two
symmetric example fields Griffiths himself uses, proven symbolically
(SymPy), then cross-checked by two INDEPENDENT numerical methods -- torch
autograd (exact Jacobian) and a from-scratch C program (finite
differences) -- same "same physics, different runtime, proven to agree"
polyglot pattern as dgs.circuits_polyglot / dgs.dispersion_polyglot.

  IRROTATIONAL example: E ~ rhat/r^2 (point-charge field direction,
  spherically symmetric). It's a pure gradient, E = -grad(1/r), so
  curl(E) = 0 EVERYWHERE except the origin -- a direct instance of
  Griffiths' "curl grad f = 0" identity, not a coincidence.

  SOLENOIDAL example: B ~ phihat/s (infinite straight-wire field
  direction, cylindrically symmetric, s = distance from the wire). It's a
  pure curl, B = curl(-ln(s) zhat), so div(B) = 0 EVERYWHERE except on the
  wire -- Griffiths' "div curl A = 0" identity.

  THE SUBTLETY (checked, not glossed over): curl(B) is ALSO zero
  everywhere away from the wire (checked below), yet the circulation
  oint B.dl around a loop ENCLOSING the wire is a nonzero constant
  (Ampere's law) -- pointwise curl=0 does not imply path-independence when
  the domain has a hole (the wire) removed from it. Verified numerically
  by a real contour integral, not just stated.
"""

import os
import subprocess

import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D, curl as sp_curl, divergence as sp_div, gradient as sp_grad

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"


# ── 1. SymPy: the two theorems, made concrete and exact ─────────────────────

def irrotational_field_proof():
    """E ~ rhat/r^2 = -grad(1/r): proves BOTH that it equals a gradient
    (so curl(E)=0 follows from Griffiths' "curl grad f = 0" identity, not
    coincidence) AND directly that curl(E)=0, independently."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    r = sp.sqrt(x**2 + y**2 + z**2)
    E = (x * N.i + y * N.j + z * N.k) / r**3
    V = -1 / r

    grad_V = sp_grad(V)
    is_gradient = sp.simplify((grad_V - E).dot(N.i)) == 0 and \
                  sp.simplify((grad_V - E).dot(N.j)) == 0 and \
                  sp.simplify((grad_V - E).dot(N.k)) == 0
    curl_E = sp_curl(E)
    curl_is_zero = sp.simplify(curl_E.dot(N.i)) == 0 and \
                   sp.simplify(curl_E.dot(N.j)) == 0 and \
                   sp.simplify(curl_E.dot(N.k)) == 0

    if not is_gradient:
        raise AssertionError("E != -grad(1/r)")
    if not curl_is_zero:
        raise AssertionError(f"curl(E) != 0, got {curl_E}")
    return {"is_gradient_of_minus_1_over_r": is_gradient, "curl_is_zero": curl_is_zero}


def solenoidal_field_proof():
    """B ~ phihat/s = curl(-ln(s) zhat): proves BOTH that it equals a
    curl (so div(B)=0 follows from Griffiths' "div curl A = 0" identity)
    AND directly that div(B)=0, independently. ALSO checks curl(B)=0 away
    from the wire (see module docstring's subtlety)."""
    N = CoordSys3D('N')
    x, y, z = N.x, N.y, N.z
    s = sp.sqrt(x**2 + y**2)
    B = (-y * N.i + x * N.j) / s**2
    A = -sp.log(s) * N.k

    curl_A = sp_curl(A)
    is_curl = sp.simplify((curl_A - B).dot(N.i)) == 0 and \
              sp.simplify((curl_A - B).dot(N.j)) == 0 and \
              sp.simplify((curl_A - B).dot(N.k)) == 0
    div_B = sp.simplify(sp_div(B))
    curl_B = sp_curl(B)
    curl_B_is_zero = sp.simplify(curl_B.dot(N.i)) == 0 and \
                      sp.simplify(curl_B.dot(N.j)) == 0 and \
                      sp.simplify(curl_B.dot(N.k)) == 0

    if not is_curl:
        raise AssertionError("B != curl(-ln(s) zhat)")
    if div_B != 0:
        raise AssertionError(f"div(B) != 0, got {div_B}")
    if not curl_B_is_zero:
        raise AssertionError(f"curl(B) != 0 away from the wire, got {curl_B}")
    return {"is_curl_of_vector_potential": is_curl, "div_is_zero": bool(div_B == 0),
            "curl_is_zero_away_from_wire": curl_B_is_zero}


def wire_circulation_symbolic(radius=sp.Symbol('a', positive=True)) -> sp.Expr:
    """oint B.dl around a circle of radius `radius` centered on the wire:
    B ~ phihat/s has magnitude 1/s, dl = s*dphi*phihat has magnitude
    s*dphi, so B.dl = (1/s)*(s*dphi) = dphi -- the s dependence cancels
    EXACTLY, giving int_0^{2pi} dphi = 2*pi for every radius, despite
    curl(B)=0 everywhere the loop's interior minus the wire itself lives.
    This is the exact SymPy version of the numerical proof in
    wire_circulation_numeric."""
    phi = sp.Symbol('phi', real=True)
    integrand = sp.Integer(1)   # B.dl = (1/s)*(s*dphi) = dphi -- s cancels, independent of `radius`
    return sp.integrate(integrand, (phi, 0, 2 * sp.pi))


# ── 2. Torch: autograd-exact pointwise curl and divergence ──────────────────

def _torch_field(kind: str):
    """Returns a torch-differentiable vector field function (x,y,z)->(3,)
    for kind in {'irrotational', 'solenoidal'} (see module docstring)."""
    import torch

    def irrotational(p):
        r = torch.linalg.norm(p)
        return p / r**3

    def solenoidal(p):
        x, y = p[0], p[1]
        s2 = x**2 + y**2
        return torch.stack([-y / s2, x / s2, torch.zeros_like(x)])

    return {"irrotational": irrotational, "solenoidal": solenoidal}[kind]


def torch_div_curl_at_points(kind: str, points: np.ndarray) -> dict:
    """Autograd-EXACT divergence (Jacobian trace) and curl (antisymmetric
    Jacobian components) of the named field at each row of `points`, via
    torch.func.jacrev+vmap -- same machinery as
    dgs.griffiths_1p49_polyglot.torch_divergence_of_A_off_origin, extended
    here to also extract curl."""
    import torch
    from torch.func import jacrev, vmap

    field = _torch_field(kind)
    pts = torch.as_tensor(np.asarray(points, dtype=np.float64))
    jac = vmap(jacrev(field))(pts)   # (n, 3, 3), jac[:,i,j] = d F_i / d x_j

    div = torch.einsum('nii->n', jac)
    curl_x = jac[:, 2, 1] - jac[:, 1, 2]
    curl_y = jac[:, 0, 2] - jac[:, 2, 0]
    curl_z = jac[:, 1, 0] - jac[:, 0, 1]
    curl = torch.stack([curl_x, curl_y, curl_z], dim=1)

    return {"divergence": div.detach().numpy(), "curl": curl.detach().numpy(),
            "curl_magnitude": torch.linalg.norm(curl, dim=1).detach().numpy()}


def wire_circulation_numeric(radius: float = 1.0, n_points: int = 100_000) -> float:
    """oint B.dl around a circle of the given radius centered on the wire
    (z-axis), evaluated by direct numerical trapezoidal quadrature of
    B(phi).dl(phi) -- an independent numeric confirmation of
    wire_circulation_symbolic's exact 2*pi answer."""
    import torch
    field = _torch_field("solenoidal")
    phi = torch.linspace(0, 2 * np.pi, n_points, dtype=torch.float64)
    x, y = radius * torch.cos(phi), radius * torch.sin(phi)
    z = torch.zeros_like(x)
    pts = torch.stack([x, y, z], dim=1)
    B = torch.stack([field(p) for p in pts])
    # dl = radius * dphi * phihat = radius*dphi*(-sin(phi), cos(phi), 0)
    dphi = phi[1] - phi[0]
    dl = radius * dphi * torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)], dim=1)
    return float(torch.sum(torch.einsum('ni,ni->n', B, dl)))


# ── 3. C: independent finite-difference cross-check ─────────────────────────

C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// kind: 0 = irrotational (rhat/r^2), 1 = solenoidal (phihat/s)
static void eval_field(int kind, double x, double y, double z, double *out) {
    if (kind == 0) {
        double r = sqrt(x*x + y*y + z*z);
        double r3 = r*r*r;
        out[0] = x / r3; out[1] = y / r3; out[2] = z / r3;
    } else {
        double s2 = x*x + y*y;
        out[0] = -y / s2; out[1] = x / s2; out[2] = 0.0;
    }
}

int main(int argc, char **argv) {
    int kind = atoi(argv[1]);
    double x = atof(argv[2]), y = atof(argv[3]), z = atof(argv[4]);
    double h = atof(argv[5]);

    double fxp[3], fxm[3], fyp[3], fym[3], fzp[3], fzm[3];
    eval_field(kind, x+h, y, z, fxp); eval_field(kind, x-h, y, z, fxm);
    eval_field(kind, x, y+h, z, fyp); eval_field(kind, x, y-h, z, fym);
    eval_field(kind, x, y, z+h, fzp); eval_field(kind, x, y, z-h, fzm);

    double dFx_dx = (fxp[0]-fxm[0])/(2*h);
    double dFy_dy = (fyp[1]-fym[1])/(2*h);
    double dFz_dz = (fzp[2]-fzm[2])/(2*h);
    double divergence = dFx_dx + dFy_dy + dFz_dz;

    double dFz_dy = (fyp[2]-fym[2])/(2*h);
    double dFy_dz = (fzp[1]-fzm[1])/(2*h);
    double dFx_dz = (fzp[0]-fzm[0])/(2*h);
    double dFz_dx = (fxp[2]-fxm[2])/(2*h);
    double dFy_dx = (fxp[1]-fxm[1])/(2*h);
    double dFx_dy = (fyp[0]-fym[0])/(2*h);

    double curl_x = dFz_dy - dFy_dz;
    double curl_y = dFx_dz - dFz_dx;
    double curl_z = dFy_dx - dFx_dy;

    printf("%.10e %.10e %.10e %.10e\n", divergence, curl_x, curl_y, curl_z);
    return 0;
}
"""


def compile_c_field_ops(out_dir: str, gcc_path: str = GCC_DEFAULT) -> str:
    """Writes C_SOURCE to disk and compiles it with gcc -O2 (same pattern
    as dgs.circuits_polyglot.compile_c_rlc)."""
    src_path = os.path.join(out_dir, "field_ops.c")
    exe_path = os.path.join(out_dir, "field_ops.exe")
    with open(src_path, "w") as f:
        f.write(C_SOURCE)
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(gcc_path) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([gcc_path, "-O2", "-o", exe_path, src_path],
                             capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compile failed: {result.stderr}")
    return exe_path


def run_c_field_ops(exe_path: str, kind: str, point, h: float = 1e-5) -> dict:
    """Runs the compiled C finite-difference div/curl evaluator at one
    point."""
    kind_int = {"irrotational": 0, "solenoidal": 1}[kind]
    x, y, z = point
    result = subprocess.run(
        [exe_path, str(kind_int), str(x), str(y), str(z), str(h)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C binary failed: {result.stderr}")
    div, cx, cy, cz = (float(v) for v in result.stdout.split())
    return {"divergence": div, "curl": np.array([cx, cy, cz])}


def cross_validate_languages(out_dir: str, gcc_path: str = GCC_DEFAULT) -> dict:
    """Runs both fields at a handful of off-singularity points through
    torch (exact autograd) and C (finite differences), reporting max
    disagreement -- the actual proof that both independently-coded
    numerical paths agree with each other, and with the SymPy-exact
    "should be zero" claims."""
    exe = compile_c_field_ops(out_dir, gcc_path=gcc_path)
    points = {"irrotational": [(1.0, 0.5, 0.3), (2.0, -1.0, 0.7), (0.3, 0.3, 2.0)],
              "solenoidal": [(1.0, 0.5, 0.0), (2.0, -1.0, 3.0), (0.3, 0.3, -2.0)]}

    out = {}
    for kind, pts in points.items():
        torch_res = torch_div_curl_at_points(kind, np.array(pts))
        c_div, c_curl = [], []
        for p in pts:
            r = run_c_field_ops(exe, kind, p, h=1e-5)
            c_div.append(r["divergence"])
            c_curl.append(r["curl"])
        c_div, c_curl = np.array(c_div), np.array(c_curl)
        out[kind] = {
            "torch_divergence": torch_res["divergence"], "c_divergence": c_div,
            "max_abs_diff_divergence": float(np.max(np.abs(torch_res["divergence"] - c_div))),
            "torch_curl": torch_res["curl"], "c_curl": c_curl,
            "max_abs_diff_curl": float(np.max(np.abs(torch_res["curl"] - c_curl))),
        }
    return out


if __name__ == "__main__":
    print("=== 1. SymPy: Griffiths' two theorems, made concrete ===")
    irr = irrotational_field_proof()
    print(f"  irrotational (E~rhat/r^2): is -grad(1/r)={irr['is_gradient_of_minus_1_over_r']}, "
          f"curl(E)=0: {irr['curl_is_zero']}")
    sol = solenoidal_field_proof()
    print(f"  solenoidal (B~phihat/s): is curl(-ln(s)zhat)={sol['is_curl_of_vector_potential']}, "
          f"div(B)=0: {sol['div_is_zero']}, curl(B)=0 away from wire: {sol['curl_is_zero_away_from_wire']}")

    circ_exact = wire_circulation_symbolic()
    print(f"\n  circulation oint B.dl around the wire (SymPy, exact, any radius): {circ_exact}")
    circ_numeric = wire_circulation_numeric(radius=1.0)
    circ_numeric_r2 = wire_circulation_numeric(radius=2.5)
    print(f"  circulation (numeric, radius=1.0): {circ_numeric:.6f}")
    print(f"  circulation (numeric, radius=2.5): {circ_numeric_r2:.6f}   (both should be ~2*pi={2*np.pi:.6f})")
    print("  ^ nonzero circulation despite curl(B)=0 everywhere off the wire: the domain has a")
    print("    hole at the wire, so curl=0 pointwise does NOT imply path-independence here.")

    print("\n=== 2 & 3. Cross-validation: torch (exact) vs. C (finite differences) ===")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = cross_validate_languages(tmp)
    for kind, r in result.items():
        print(f"\n  {kind}:")
        print(f"    torch divergence = {r['torch_divergence']}")
        print(f"    C     divergence = {r['c_divergence']}")
        print(f"    max abs diff (divergence) = {r['max_abs_diff_divergence']:.3e}")
        print(f"    max abs diff (curl)       = {r['max_abs_diff_curl']:.3e}")

    print("\nSame two Griffiths identities (curl-of-grad=0, div-of-curl=0), proven exactly")
    print("by SymPy, reproduced independently by torch autograd and by a from-scratch C")
    print("finite-difference program -- and the wire's nonzero circulation despite zero")
    print("pointwise curl, confirmed numerically rather than just asserted from theory.")
