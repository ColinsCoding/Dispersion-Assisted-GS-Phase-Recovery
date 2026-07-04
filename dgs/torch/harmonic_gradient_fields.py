"""Problem 1.20 (construct v with zero divergence AND zero curl
everywhere), formalized as linear algebra and verified with torch
autograd -- and, instead of hunting for more examples by trial and error
("crazy combinations"), a SINGLE constructive theorem that generates
infinitely many correct answers on demand:

  If f is HARMONIC (Laplace's equation: nabla^2 f = 0), then v = grad(f)
  automatically has BOTH properties:
    * curl(v) = curl(grad(f)) = 0   -- ALWAYS true, for any f whatsoever
      (dgs.curl_div_modern_physics.curl_of_gradient_is_zero_symbolic
       already proved this identically, for a GENERIC f)
    * div(v)  = div(grad(f)) = nabla^2(f) = 0  -- true PRECISELY WHEN f
      is harmonic, which is the one extra condition to impose

So "construct a vector field with zero div and zero curl" reduces to
"pick literally any harmonic scalar function" -- and harmonic functions
are not scarce: every real/imaginary part of a complex-analytic function
is harmonic (a genuinely deep fact from complex analysis, the Cauchy-
Riemann equations), giving an inexhaustible supply.

LINEAR ALGEBRA formalization: at a point, a vector field v=(vx,vy,vz) has
a 3x3 JACOBIAN matrix J_ij = d(v_i)/d(x_j). Divergence is TRACE(J) (the
sum of the diagonal); curl is built from J's ANTISYMMETRIC part (the
piece that doesn't survive transposition) -- these are the two
irreducible linear-algebra invariants a rank-2 tensor can offer, which is
exactly why div and curl are the only two independent first derivatives
of a vector field worth naming.

torch.autograd.functional.jacobian computes J directly (computer-
engineering machinery -- automatic differentiation -- doing the same
calculus SymPy did symbolically elsewhere in this repo).
"""

import torch


def jacobian_torch(v_func, point):
    """The full 3x3 Jacobian of v=(vx,vy,vz) at `point`, via torch
    autograd (torch.autograd.functional.jacobian), not finite differences
    and not hand-differentiation."""
    point = torch.as_tensor(point, dtype=torch.float64).requires_grad_(True)
    J = torch.autograd.functional.jacobian(v_func, point)
    return J


def divergence_from_jacobian(J):
    """div(v) = trace(J) -- the sum of the diagonal, one of the only two
    basis-independent (linear-algebra) invariants a matrix has (the other
    being determinant, which doesn't correspond to a first-derivative
    vector-calculus operator)."""
    return torch.trace(J)


def curl_from_jacobian(J):
    """curl(v) built from J's ANTISYMMETRIC part: curl_x=J[2,1]-J[1,2],
    curl_y=J[0,2]-J[2,0], curl_z=J[1,0]-J[0,1] -- each component is twice
    one off-diagonal entry of (J - J^T)/2, the piece of J invariant under
    rotations but odd under transposition."""
    return torch.tensor([
        J[2, 1] - J[1, 2],
        J[0, 2] - J[2, 0],
        J[1, 0] - J[0, 1],
    ], dtype=J.dtype)


def verify_vector_field_torch(v_func, point):
    """div and curl of v_func at `point`, both via torch autograd's
    Jacobian -- one function call replaces the whole by-hand derivative
    bookkeeping."""
    J = jacobian_torch(v_func, point)
    return divergence_from_jacobian(J), curl_from_jacobian(J)


def laplacian_torch(f_func, point):
    """nabla^2(f) at `point`, via torch's SECOND autograd pass (the
    Hessian's trace) -- tests whether a candidate scalar f is harmonic."""
    point = torch.as_tensor(point, dtype=torch.float64).requires_grad_(True)
    H = torch.autograd.functional.hessian(f_func, point)
    return torch.trace(H)


def gradient_field_from_scalar(f_func, point):
    """v = grad(f) at `point`, via torch autograd (the first derivative
    pass) -- the CONSTRUCTIVE half of the theorem: build the candidate
    vector field directly from a scalar, rather than guessing components."""
    point = torch.as_tensor(point, dtype=torch.float64).requires_grad_(True)
    f_val = f_func(point)
    grad_f, = torch.autograd.grad(f_val, point, create_graph=True)
    return grad_f


if __name__ == "__main__":
    print("=== Verifying the textbook Problem 1.20 examples via torch autograd ===")
    examples = {
        "v=(y,x,0)": lambda p: torch.stack([p[1], p[0], torch.zeros_like(p[0])]),
        "v=(yz,xz,xy)": lambda p: torch.stack([p[1]*p[2], p[0]*p[2], p[0]*p[1]]),
        "v=(3x^2z-z^3, 3y, x^3-3xz^2)  [SUSPECT, see below]": lambda p: torch.stack([
            3*p[0]**2*p[2] - p[2]**3, 3*p[1], p[0]**3 - 3*p[0]*p[2]**2]),
        "v=(sin(x)cosh(y), -cos(x)sinh(y), 0)": lambda p: torch.stack([
            torch.sin(p[0])*torch.cosh(p[1]), -torch.cos(p[0])*torch.sinh(p[1]), torch.zeros_like(p[0])]),
    }
    point = [1.3, -0.7, 0.5]
    for name, v_func in examples.items():
        div, curl = verify_vector_field_torch(v_func, point)
        flag = "  <-- NOT ZERO, this example is WRONG as transcribed" if abs(div.item()) > 1e-8 else ""
        print(f"{name}: div={div.item():.6f}, |curl|={torch.norm(curl).item():.2e}{flag}")

    print("\n=== The constructive theorem: v=grad(f) for ANY harmonic f works ===")
    harmonic_candidates = {
        "f=x^2-y^2": lambda p: p[0]**2 - p[1]**2,
        "f=x^3-3xy^2": lambda p: p[0]**3 - 3*p[0]*p[1]**2,   # Re(z^3), z=x+iy
        "f=exp(x)sin(y)": lambda p: torch.exp(p[0])*torch.sin(p[1]),  # Im(exp(z))
        "f=xyz": lambda p: p[0]*p[1]*p[2],
    }
    for name, f_func in harmonic_candidates.items():
        lap = laplacian_torch(f_func, point)
        v_func_from_f = lambda p, f=f_func: gradient_field_from_scalar(f, p)
        div, curl = verify_vector_field_torch(v_func_from_f, point)
        print(f"{name}: laplacian={lap.item():.2e}, "
              f"grad(f) has div={div.item():.2e}, |curl|={torch.norm(curl).item():.2e}")

    print("\n=== The point ===")
    print("Every harmonic f, no matter how it's built, hands you a new valid")
    print("Problem 1.20 answer for free via v=grad(f) -- no combinatorial search needed,")
    print("just an inexhaustible supply from complex analysis (real/imaginary parts of")
    print("ANY analytic function z^n, exp(z), sin(z), log(z), etc. are all harmonic).")
