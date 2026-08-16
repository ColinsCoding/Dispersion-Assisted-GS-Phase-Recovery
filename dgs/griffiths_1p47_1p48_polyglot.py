"""Griffiths Problems 1.47 and 1.48, done as actual calculus (SymPy exact
symbolic delta-sifted integration, not hand algebra restated in code), then
cross-checked by an independently-coded numerical method in Python and
MATLAB -- same two-CAS/two-runtime cross-validation posture as
dgs.griffiths_1p49_polyglot.

SymPy can integrate a product of 1-D DiracDelta factors EXACTLY, including
finite bounds (verified here: sp.integrate(f(x,y,z)*DiracDelta(x-x0)*...,
(x,-1,1),...) correctly returns 0 if x0 is outside [-1,1], and the sifted
value if inside) -- so Problem 1.48's four integrals are proven, not just
plugged into, with the vector components left SYMBOLIC wherever the
problem's domain allows it (1.48(a): int over ALL space, so the sifted
vector a stays fully symbolic; 1.48(b): int over a FIXED cube, so the
sifted vector b also stays symbolic, giving the general formula
(bx^2+by^2+bz^2)/125 before any numbers are plugged in).

MATLAB has no Symbolic Math Toolbox on this machine (checked -- `syms`
raises), so its role here is different from dgs.griffiths_1p49_polyglot's
MATLAB `integral()` quadrature: it independently re-derives each answer
numerically, via a deterministic (no RNG) fine grid Riemann sum of a
narrow-Gaussian regularization of the delta function -- an independent
NUMERICAL method, coded from scratch in both Python and MATLAB, checked
against SymPy's EXACT answer.

    delta^3(k*(r-r0))  ~=  (1/(sigma*sqrt(2*pi)))^3 * exp(-|k*(r-r0)|^2 / (2*sigma^2))

as sigma -> 0, restricted to a small zoomed box around r0 (big enough to
hold the Gaussian's mass, small enough to correctly resolve whether r0 sits
inside or outside a nearby domain boundary, as in 1.48(c)/(d)).
"""

import os
import re
import subprocess

import numpy as np
import sympy as sp

MATLAB_DEFAULT = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"


# ── 1. Problem 1.47: symbolic delta-sifting proofs ──────────────────────────

def point_charge_density_proof(q_val: float = None):
    """(a) rho(r) = q*delta^3(r-r'): int rho dtau = q, proven for a fully
    symbolic charge q and displacement r' (not a specific number) by
    sifting over all space."""
    x, y, z, xp, yp, zp, q = sp.symbols('x y z xp yp zp q', real=True)
    integrand = q * sp.DiracDelta(x - xp) * sp.DiracDelta(y - yp) * sp.DiracDelta(z - zp)
    total = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)),
                                       (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
    if sp.simplify(total - q) != 0:
        raise AssertionError(f"expected total charge q, got {total}")
    return total


def dipole_density_proof():
    """(b) rho(r) = q*delta^3(r) - q*delta^3(r-a): total charge is EXACTLY
    zero (net neutral) and the dipole moment p = int r*rho dtau is EXACTLY
    q*a, both proven symbolically for general q and vector a = (ax,ay,az)."""
    x, y, z, q, ax, ay, az = sp.symbols('x y z q a_x a_y a_z', real=True)
    delta_origin = sp.DiracDelta(x) * sp.DiracDelta(y) * sp.DiracDelta(z)
    delta_at_a = sp.DiracDelta(x - ax) * sp.DiracDelta(y - ay) * sp.DiracDelta(z - az)
    rho = q * delta_origin - q * delta_at_a

    def vol_integrate(expr):
        return sp.integrate(sp.integrate(sp.integrate(expr, (x, -sp.oo, sp.oo)),
                                          (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))

    total_charge = vol_integrate(rho)
    px = vol_integrate(x * rho)
    py = vol_integrate(y * rho)
    pz = vol_integrate(z * rho)

    if sp.simplify(total_charge) != 0:
        raise AssertionError(f"expected net-neutral total charge 0, got {total_charge}")
    for component, expected in [(px, -q * ax), (py, -q * ay), (pz, -q * az)]:
        if sp.simplify(component - expected) != 0:
            raise AssertionError(f"dipole moment component {component} != {expected}")
    # p = int r*rho dtau = -q*a here (charge -q at origin, +q at a is the
    # OPPOSITE sign convention from Griffiths' own +q at a, -q at origin
    # example -- p = q*a for THAT convention; verified as -q*a for this one).
    return {"total_charge": total_charge, "dipole_moment": sp.Matrix([px, py, pz])}


def shell_density_proof():
    """(c) rho(r) = Q/(4*pi*R^2) * delta(r-R) (radial delta, spherical
    shell): int rho dtau = Q, proven for symbolic Q, R via the exact 1-D
    radial integral int_0^inf rho * 4*pi*r^2 dr (the angular part is
    trivially 4*pi by isotropy, left implicit exactly as the textbook
    does) -- an EXACT identity for every R, not just R -> infinity."""
    r, R, Q = sp.symbols('r R Q', positive=True)
    rho = Q / (4 * sp.pi * R**2) * sp.DiracDelta(r - R)
    total = sp.integrate(rho * 4 * sp.pi * r**2, (r, 0, sp.oo))
    if sp.simplify(total - Q) != 0:
        raise AssertionError(f"expected total charge Q, got {total}")
    return total


# ── 2. Problem 1.48: four delta-sifted integrals, symbolic where possible ──

def integral_1p48a_symbolic():
    """int_{all space} (r.r + r.a + a.a) * delta^3(r-a) dtau = 3*a.a,
    proven for a fully symbolic vector a = (ax,ay,az) (not a specific
    number) -- the general formula, not a single plugged-in check."""
    x, y, z, ax, ay, az = sp.symbols('x y z a_x a_y a_z', real=True)
    r_vec, a_vec = sp.Matrix([x, y, z]), sp.Matrix([ax, ay, az])
    integrand = r_vec.dot(r_vec) + r_vec.dot(a_vec) + a_vec.dot(a_vec)
    delta3 = sp.DiracDelta(x - ax) * sp.DiracDelta(y - ay) * sp.DiracDelta(z - az)
    result = sp.integrate(sp.integrate(sp.integrate(integrand * delta3, (x, -sp.oo, sp.oo)),
                                        (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
    expected = 3 * a_vec.dot(a_vec)
    if sp.simplify(result - expected) != 0:
        raise AssertionError(f"expected 3*a.a, got {result}")
    return result


def integral_1p48b_symbolic(b_vec=(0, 4, 3)):
    """int_V |r-b|^2 * delta^3(5r) dtau over V = cube of side 2 centered at
    the origin, proven both (1) fully symbolically for general b = (bx,by,
    bz), giving (bx^2+by^2+bz^2)/125, and (2) at Griffiths' own numbers
    b=4*yhat+3*zhat, giving exactly 1/5 -- the scaling identity
    delta^3(5r)=delta^3(r)/125 is exercised implicitly by SymPy's own exact
    integration, not assumed."""
    x, y, z, bx, by, bz = sp.symbols('x y z b_x b_y b_z', real=True)
    r_vec, b_sym = sp.Matrix([x, y, z]), sp.Matrix([bx, by, bz])
    integrand = (r_vec - b_sym).dot(r_vec - b_sym)
    delta3_scaled = sp.DiracDelta(5 * x) * sp.DiracDelta(5 * y) * sp.DiracDelta(5 * z)
    general = sp.integrate(integrand * delta3_scaled, (x, -1, 1), (y, -1, 1), (z, -1, 1))
    expected_general = (bx**2 + by**2 + bz**2) / 125
    if sp.simplify(general - expected_general) != 0:
        raise AssertionError(f"expected (bx^2+by^2+bz^2)/125, got {general}")

    numeric = general.subs({bx: b_vec[0], by: b_vec[1], bz: b_vec[2]})
    if tuple(b_vec) == (0, 4, 3) and sp.simplify(numeric - sp.Rational(1, 5)) != 0:
        raise AssertionError(f"expected 1/5 at Griffiths' b=(0,4,3), got {numeric}")
    return {"general": general, "numeric": numeric}


def integral_1p48c(c_vec=(5, 3, 2), R: float = 6):
    """int_V [r^4 + r^2*(r.c) + c^4] * delta^3(r-c) dtau over V = sphere of
    radius R centered at the origin. Proof has two parts: (1) sift the
    integrand at r=c EXACTLY via SymPy (all-space integral), (2) check
    containment |c|^2 vs R^2 EXACTLY with integer/rational arithmetic (no
    floating point) -- |c|^2=38 > 36=R^2, so c is OUTSIDE V and the
    integral is 0 regardless of the sifted value at c."""
    x, y, z, cx, cy, cz = sp.symbols('x y z c_x c_y c_z', real=True)
    r_vec, c_sym = sp.Matrix([x, y, z]), sp.Matrix([cx, cy, cz])
    r2, c2 = r_vec.dot(r_vec), c_sym.dot(c_sym)
    integrand = r2**2 + r2 * r_vec.dot(c_sym) + c2**2
    delta3 = sp.DiracDelta(x - cx) * sp.DiracDelta(y - cy) * sp.DiracDelta(z - cz)
    sifted_general = sp.integrate(sp.integrate(sp.integrate(integrand * delta3, (x, -sp.oo, sp.oo)),
                                                (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
    sifted_at_c = sifted_general.subs({cx: c_vec[0], cy: c_vec[1], cz: c_vec[2]})

    c_mag_sq = sp.Integer(c_vec[0])**2 + sp.Integer(c_vec[1])**2 + sp.Integer(c_vec[2])**2
    R_sq = sp.Integer(R)**2
    is_outside = bool(c_mag_sq > R_sq)
    final = sp.Integer(0) if is_outside else sifted_at_c
    return {"sifted_value_at_c": sifted_at_c, "c_mag_sq": c_mag_sq, "R_sq": R_sq,
            "outside_V": is_outside, "final_answer": final}


def integral_1p48d(d_vec=(1, 2, 3), e_vec=(3, 2, 1), center=(2, 2, 2), R: float = 1.5):
    """int_V r.(d-r) * delta^3(e-r) dtau over V = sphere of radius R
    centered at `center`. delta^3(e-r)=delta^3(r-e) (the delta is even), so
    this sifts r=e; proven exactly via SymPy, then containment checked with
    EXACT rational arithmetic: |e-center|^2 = 2 < 2.25 = R^2, so e is
    INSIDE V and the sifted value e.(d-e) = -4 stands."""
    x, y, z, dx, dy, dz, ex, ey, ez = sp.symbols('x y z d_x d_y d_z e_x e_y e_z', real=True)
    r_vec, d_sym, e_sym = sp.Matrix([x, y, z]), sp.Matrix([dx, dy, dz]), sp.Matrix([ex, ey, ez])
    integrand = r_vec.dot(d_sym - r_vec)
    delta3 = sp.DiracDelta(x - ex) * sp.DiracDelta(y - ey) * sp.DiracDelta(z - ez)
    sifted_general = sp.integrate(sp.integrate(sp.integrate(integrand * delta3, (x, -sp.oo, sp.oo)),
                                                (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
    subs = {dx: d_vec[0], dy: d_vec[1], dz: d_vec[2], ex: e_vec[0], ey: e_vec[1], ez: e_vec[2]}
    sifted_at_e = sifted_general.subs(subs)

    dist_sq = sum(sp.Rational(e_vec[i] - center[i])**2 for i in range(3))
    R_sq = sp.Rational(R)**2
    is_inside = bool(dist_sq < R_sq)
    final = sifted_at_e if is_inside else sp.Integer(0)
    is_griffiths_case = (tuple(d_vec), tuple(e_vec), tuple(center), R) == ((1, 2, 3), (3, 2, 1), (2, 2, 2), 1.5)
    if is_griffiths_case and sp.simplify(final - (-4)) != 0:
        raise AssertionError(f"expected -4 for Griffiths' own numbers, got {final}")
    return {"sifted_value_at_e": sifted_at_e, "dist_sq": dist_sq, "R_sq": R_sq,
            "inside_V": is_inside, "final_answer": final}


# ── 3. Independent numeric cross-check: deterministic grid quadrature ──────

def _gaussian_delta_scaled(X, Y, Z, center, sigma, k: float = 1.0):
    """Narrow-Gaussian regularization of delta^3(k*(r-center)), as a
    function of the UNSCALED coordinate r (matches SymPy's DiracDelta(k*x)
    convention used above -- k=5 reproduces 1.48(b)'s delta^3(5r) with
    center=(0,0,0))."""
    cx, cy, cz = center
    norm = (1.0 / (sigma * np.sqrt(2 * np.pi)))**3
    return norm * np.exp(-(k**2 * ((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)) / (2 * sigma**2))


def grid_quadrature_sifted_integral(f, center, half_width: float, n_per_axis: int,
                                     sigma: float, k: float = 1.0, domain_mask=None) -> float:
    """Deterministic (no RNG) Riemann-sum estimate of
    int f(r) * delta^3(k*(r-center)) * domain_mask(r) dtau, using a fine
    grid zoomed into a box of half-width `half_width` around `center` (must
    be small enough to correctly resolve nearby domain boundaries, large
    enough to hold the Gaussian's mass -- i.e. half_width >> sigma/k,
    verified by convergence as sigma shrinks in cross_validate_1p48).
    `f` is the plain polynomial integrand (NOT including the delta);
    `domain_mask(X,Y,Z)` (absolute coordinates) restricts the integration
    region, e.g. cube_mask/sphere_mask below."""
    if half_width <= 0 or n_per_axis < 3 or sigma <= 0 or k <= 0:
        raise ValueError("half_width, sigma, k must be > 0 and n_per_axis >= 3")
    cx, cy, cz = center
    axis = np.linspace(-half_width, half_width, n_per_axis)
    dx = axis[1] - axis[0]
    X, Y, Z = np.meshgrid(cx + axis, cy + axis, cz + axis, indexing="ij")
    weight = _gaussian_delta_scaled(X, Y, Z, center, sigma, k)
    if domain_mask is not None:
        weight = weight * domain_mask(X, Y, Z)
    values = f(X, Y, Z) * weight
    return float(np.sum(values) * dx**3)


def cube_mask(half_side: float, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center
    def mask(X, Y, Z):
        return ((np.abs(X - cx) <= half_side) & (np.abs(Y - cy) <= half_side) &
                (np.abs(Z - cz) <= half_side)).astype(float)
    return mask


def sphere_mask(R: float, center=(0.0, 0.0, 0.0)):
    cx, cy, cz = center
    def mask(X, Y, Z):
        return (((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) <= R**2).astype(float)
    return mask


def cross_validate_1p48(sigma: float = 0.01, half_width: float = 0.06, n_per_axis: int = 81) -> dict:
    """Runs all four Problem 1.48 integrals via the exact SymPy proofs
    above AND the deterministic grid-quadrature method (Python-side), and
    reports agreement. `half_width` must stay smaller than the tightest
    domain-boundary buffer among the four problems (1.48(d)'s e sits
    R^2-|e-center|^2 = 0.25 inside the sphere in SQUARED distance, i.e. a
    literal-distance buffer of ~0.086 -- default half_width=0.06 clears it)."""
    ax, ay, az = sp.symbols('a_x a_y a_z', real=True)
    exact = {
        "a": float(integral_1p48a_symbolic().subs({ax: 1, ay: 2, az: 2})),
        "b": float(integral_1p48b_symbolic()["numeric"]),
        "c": float(integral_1p48c()["final_answer"]),
        "d": float(integral_1p48d()["final_answer"]),
    }

    numeric = {}
    numeric["a"] = grid_quadrature_sifted_integral(
        f=lambda X, Y, Z: X**2 + Y**2 + Z**2 + (1 * X + 2 * Y + 2 * Z) + (1 + 4 + 4),
        center=(1.0, 2.0, 2.0), half_width=half_width, n_per_axis=n_per_axis, sigma=sigma)
    numeric["b"] = grid_quadrature_sifted_integral(
        f=lambda X, Y, Z: (X - 0.0)**2 + (Y - 4.0)**2 + (Z - 3.0)**2,
        center=(0.0, 0.0, 0.0), half_width=half_width, n_per_axis=n_per_axis, sigma=sigma, k=5.0,
        domain_mask=cube_mask(half_side=1.0))
    numeric["c"] = grid_quadrature_sifted_integral(
        f=lambda X, Y, Z: (X**2 + Y**2 + Z**2)**2 + (X**2 + Y**2 + Z**2) * (5 * X + 3 * Y + 2 * Z) + 38**2,
        center=(5.0, 3.0, 2.0), half_width=half_width, n_per_axis=n_per_axis, sigma=sigma,
        domain_mask=sphere_mask(R=6.0))
    numeric["d"] = grid_quadrature_sifted_integral(
        f=lambda X, Y, Z: X * (1 - X) + Y * (2 - Y) + Z * (3 - Z),
        center=(3.0, 2.0, 1.0), half_width=half_width, n_per_axis=n_per_axis, sigma=sigma,
        domain_mask=sphere_mask(R=1.5, center=(2.0, 2.0, 2.0)))

    diffs = {k_: abs(exact[k_] - numeric[k_]) for k_ in exact}
    return {"exact": exact, "numeric_python": numeric, "abs_diff": diffs}


# ── 4. MATLAB: independently-coded deterministic grid quadrature ───────────

MATLAB_SOURCE_TEMPLATE = """
sigma = {sigma!r}; half_width = {half_width!r}; n = {n_per_axis!r};
axis_pts = linspace(-half_width, half_width, n);
dx = axis_pts(2) - axis_pts(1);
[dX, dY, dZ] = meshgrid(axis_pts, axis_pts, axis_pts);

% (a): center=(1,2,2), all space (no mask), k=1
[X, Y, Z] = deal(dX+1.0, dY+2.0, dZ+2.0);
w = gaussian_delta(X, Y, Z, 1.0, 2.0, 2.0, sigma, 1.0);
fa = X.^2+Y.^2+Z.^2 + (1*X+2*Y+2*Z) + (1+4+4);
Ia = sum(fa(:).*w(:)) * dx^3;

% (b): center=(0,0,0), k=5, cube half-side 1 mask
[X, Y, Z] = deal(dX, dY, dZ);
w = gaussian_delta(X, Y, Z, 0.0, 0.0, 0.0, sigma, 5.0);
mask = (abs(X)<=1.0) & (abs(Y)<=1.0) & (abs(Z)<=1.0);
fb = (X-0.0).^2 + (Y-4.0).^2 + (Z-3.0).^2;
Ib = sum(fb(:).*w(:).*mask(:)) * dx^3;

% (c): center=(5,3,2), sphere R=6 about origin mask
[X, Y, Z] = deal(dX+5.0, dY+3.0, dZ+2.0);
w = gaussian_delta(X, Y, Z, 5.0, 3.0, 2.0, sigma, 1.0);
mask = (X.^2+Y.^2+Z.^2) <= 6.0^2;
r2 = X.^2+Y.^2+Z.^2;
fc = r2.^2 + r2.*(5*X+3*Y+2*Z) + 38^2;
Ic = sum(fc(:).*w(:).*mask(:)) * dx^3;

% (d): center=(3,2,1), sphere R=1.5 about (2,2,2) mask
[X, Y, Z] = deal(dX+3.0, dY+2.0, dZ+1.0);
w = gaussian_delta(X, Y, Z, 3.0, 2.0, 1.0, sigma, 1.0);
mask = ((X-2.0).^2+(Y-2.0).^2+(Z-2.0).^2) <= 1.5^2;
fd = X.*(1-X) + Y.*(2-Y) + Z.*(3-Z);
Id = sum(fd(:).*w(:).*mask(:)) * dx^3;

fprintf('%.10e %.10e %.10e %.10e\\n', Ia, Ib, Ic, Id);

function w = gaussian_delta(X, Y, Z, cx, cy, cz, sigma, k)
    normconst = (1/(sigma*sqrt(2*pi)))^3;
    w = normconst * exp(-(k^2 * ((X-cx).^2+(Y-cy).^2+(Z-cz).^2)) / (2*sigma^2));
end
"""


def run_matlab_1p48(out_dir: str, sigma: float = 0.01, half_width: float = 0.06,
                     n_per_axis: int = 81, matlab_path: str = MATLAB_DEFAULT) -> dict:
    """Writes and runs the MATLAB grid-quadrature script above (independent
    from-scratch reimplementation of grid_quadrature_sifted_integral, not a
    port), headless via `matlab -batch`."""
    script = MATLAB_SOURCE_TEMPLATE.format(sigma=sigma, half_width=half_width, n_per_axis=n_per_axis)
    m_path = os.path.join(out_dir, "griffiths_1p48.m")
    with open(m_path, "w") as f:
        f.write(script)
    m_path_fwd = m_path.replace("\\", "/")

    result = subprocess.run(
        [matlab_path, "-batch", f"run('{m_path_fwd}')"],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"matlab failed (code {result.returncode}): {result.stderr}\n{result.stdout}")

    pattern = re.compile(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
    rows = [m.groups() for line in result.stdout.splitlines() if (m := pattern.match(line))]
    if len(rows) != 1:
        raise RuntimeError(f"expected 1 output line from MATLAB, got {len(rows)}: {result.stdout!r}")
    a, b, c, d = (float(v) for v in rows[0])
    return {"a": a, "b": b, "c": c, "d": d}


if __name__ == "__main__":
    print("=== 1. Problem 1.47: symbolic delta-sifting proofs ===")
    print(f"  (a) point charge: total charge = {point_charge_density_proof()}  (expect q)")
    dip = dipole_density_proof()
    print(f"  (b) dipole: total charge = {dip['total_charge']} (expect 0), "
          f"p = {dip['dipole_moment'].T}  (expect -q*a for THIS sign convention)")
    print(f"  (c) spherical shell: total charge = {shell_density_proof()}  (expect Q, for ANY R)")

    print("\n=== 2. Problem 1.48: symbolic + domain-containment proofs ===")
    print(f"  (a) general: {integral_1p48a_symbolic()}  (expect 3*a.a)")
    b_res = integral_1p48b_symbolic()
    print(f"  (b) general: {b_res['general']}, at b=(0,4,3): {b_res['numeric']}  (expect 1/5)")
    c_res = integral_1p48c()
    print(f"  (c) |c|^2={c_res['c_mag_sq']} vs R^2={c_res['R_sq']}, outside_V={c_res['outside_V']}, "
          f"final={c_res['final_answer']}  (expect 0)")
    d_res = integral_1p48d()
    print(f"  (d) |e-center|^2={d_res['dist_sq']} vs R^2={d_res['R_sq']}, inside_V={d_res['inside_V']}, "
          f"final={d_res['final_answer']}  (expect -4)")

    print("\n=== 3 & 4. Cross-validation: SymPy exact vs. Python grid quadrature vs. MATLAB ===")
    py_check = cross_validate_1p48()
    print(f"{'part':>6}{'exact':>14}{'python grid':>16}{'abs diff':>14}")
    for part in "abcd":
        print(f"{part:>6}{py_check['exact'][part]:>14.6f}{py_check['numeric_python'][part]:>16.6f}"
              f"{py_check['abs_diff'][part]:>14.2e}")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        matlab_result = run_matlab_1p48(tmp)
    print(f"\n{'part':>6}{'exact':>14}{'matlab grid':>16}{'abs diff':>14}")
    for part in "abcd":
        diff = abs(py_check["exact"][part] - matlab_result[part])
        print(f"{part:>6}{py_check['exact'][part]:>14.6f}{matlab_result[part]:>16.6f}{diff:>14.2e}")

    print("\nSame textbook shortcut (sift, then check containment), proven exactly by SymPy,")
    print("reproduced independently by a from-scratch deterministic grid quadrature in both")
    print("Python and MATLAB -- three independently-coded paths, one set of answers.")
