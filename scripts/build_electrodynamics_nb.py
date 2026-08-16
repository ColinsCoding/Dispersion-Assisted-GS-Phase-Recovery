"""Build notebooks/electrodynamics_to_dispersion_operator.ipynb -- the full
research-notebook pipeline Calculus -> Vector Analysis -> Electrostatics ->
Potentials -> Fields in Matter -> Maxwell/Waves -> Optics/Photonics ->
Detector Electronics -> Sampling -> Inverse Problems -> AI.

Runs on the py312 kernel (Python 3.12 (torch)) -- Part 13 needs torch, which
is py-3.12-only in this environment; everything else (scipy/sympy/pandas)
works identically on 3.12 or 3.13, so the whole notebook runs under one
kernel rather than splitting. Build with `py -3.12 scripts/build_electrodynamics_nb.py`,
execute with `py -3.12 -m jupyter nbconvert --to notebook --execute --inplace
notebooks/electrodynamics_to_dispersion_operator.ipynb`.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

# ============================================================================
# Title
# ============================================================================
cells.append(md("""# Calculus -> AI: a dispersion-assisted phase-recovery research notebook

$$\\boxed{\\text{Calculus} \\to \\text{Vector Analysis} \\to \\text{Electrostatics} \\to \\text{Potentials}
\\to \\text{Fields in Matter} \\to \\text{Maxwell/Waves} \\to \\text{Optics/Photonics} \\to
\\text{Detector Electronics} \\to \\text{Sampling} \\to \\text{Inverse Problems} \\to \\text{AI}}$$

Every section below follows the same pattern:

$$\\text{physical question} \\to \\text{mathematical model} \\to \\text{symbolic derivation}
\\to \\text{numerical implementation} \\to \\text{visualization} \\to \\text{dimensional analysis}
\\to \\text{physical interpretation}$$

The destination is a single operator this repo's Gerchberg-Saxton receiver inverts,

$$H(f) = e^{\\,i\\pi D f^2},$$

and Part 6/9 below show it is not an ad-hoc choice: it falls straight out of Maxwell's
equations, through the dispersion relation $k(\\omega)$ and its Taylor expansion. Every
numerical/symbolic routine used here already exists as a tested module in `dgs/` or
`griffiths/` (this repo's physics library) -- citations are given inline so you can go
read the source and its own test file."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # repo root (notebook runs in notebooks/)
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
sp.init_printing()

from griffiths import electrodynamics as ed
from dgs import dispersion_gs_prototype as dg
mu, eps, omega, k, z, t = ed.mu, ed.eps, ed.omega, ed.k, ed.z, ed.t
print("loaded griffiths + dgs")"""))

# ============================================================================
# PART 1 -- Vector analysis
# ============================================================================
cells.append(md("""## Part 1 -- Vector analysis for electromagnetism

Every field quantity in electromagnetism is built from four operators on a scalar or
vector field of $\\mathbf r = x\\hat x + y\\hat y + z\\hat z$: gradient $\\nabla V$ (points
uphill), divergence $\\nabla\\cdot\\mathbf E$ (net outflow per volume -- sources/sinks),
curl $\\nabla\\times\\mathbf E$ (local rotation), and the Laplacian $\\nabla^2 V = \\nabla\\cdot
(\\nabla V)$ (curvature -- how far $V$ deviates from its neighbors' average). **Assumptions:**
fields are smooth (differentiable) everywhere they're evaluated; SI units throughout.

**Why symmetry matters.** In Cartesian coordinates the *formulas* for grad/div/curl never
change, but the *difficulty* of solving a real problem does: a field with spherical symmetry
depends on one variable ($r$) instead of three, collapsing a 3D PDE into an ODE. Below, the
same physics (the point-charge potential $V=1/r$, the line-charge potential $V=\\ln\\rho$) is
verified harmonic in the coordinate system where it is natural.

**What would I actually measure in a lab?** None of $\\nabla V, \\nabla\\cdot\\mathbf E$ etc.
are measured directly -- a lab measures $V$ (voltmeter) or $\\mathbf E$ (force on a test
charge, or a field probe) at a set of points, then estimates derivatives by finite
differences between those points. The smaller the probe spacing, the better the derivative
estimate -- exactly the finite-difference idea used numerically in Part 4.

**MATLAB equivalent:** `gradient(V, dx, dy, dz)`; `del2(V)` approximates $\\nabla^2 V/4$ on a
grid (note the factor of 4 -- MATLAB's convention differs from the raw 5-point stencil used
in Part 4).

**Questions to ask:** What is the independent variable in each derivative? Is the field
given symbolically (exact) or only at sample points (must use finite differences, with
truncation error)? Does the coordinate system match the problem's symmetry?"""))

cells.append(co("""from griffiths.vectors import x, y, z, grad, div, curl

# Cartesian example: no special symmetry, all three coordinates appear
V_cart = x**2 * y * z
A_cart = sp.Matrix([x*y, y*z, z*x])
print("Cartesian:  V = x^2 y z ,   A = (xy, yz, zx)")
print("grad V ="); display(grad(V_cart).T)
print("div A  ="); display(div(A_cart))
print("curl A ="); display(curl(A_cart).T)
print("laplacian V = div(grad V) ="); display(sp.simplify(div(grad(V_cart))))"""))

cells.append(co("""# Spherical symmetry: V = 1/r, the point-charge potential (away from the origin)
r_sym = sp.sqrt(x**2 + y**2 + z**2)
V_sph = 1 / r_sym
lap_sph = sp.simplify(div(grad(V_sph)))
print("Spherical symmetry V = 1/r :  nabla^2 V ="); display(lap_sph)
print("-> harmonic everywhere except r=0: Laplace's equation with NO integration needed,")
print("   because a spherical Gaussian surface exploits the symmetry directly (Part 3).")

# Cylindrical symmetry: V = ln(rho), the infinite-line-charge potential
rho_sym = sp.sqrt(x**2 + y**2)
V_cyl = sp.log(rho_sym)
lap_cyl = sp.simplify(div(grad(V_cyl)))
print("\\nCylindrical symmetry V = ln(rho) :  nabla^2 V ="); display(lap_cyl)
print("-> also harmonic away from the axis: same lesson, different natural coordinate.")
print("\\nBoth V's were built directly in Cartesian (x,y,z) and are STILL harmonic --")
print("the coordinate choice doesn't change the physics, only how much algebra it takes.")"""))

# ============================================================================
# PART 2 -- Electrostatics
# ============================================================================
cells.append(md("""## Part 2 -- Electrostatics: $\\rho(\\mathbf r) \\to V(\\mathbf r) \\to \\mathbf E(\\mathbf r)$

$$V(\\mathbf r) = \\frac{1}{4\\pi\\epsilon_0}\\int \\frac{\\rho(\\mathbf r')}{|\\mathbf r-\\mathbf r'|}\\,d\\tau',
\\qquad \\mathbf E = -\\nabla V.$$

For discrete point charges the integral collapses to Coulomb superposition
$\\mathbf E=\\sum_i k q_i(\\mathbf r-\\mathbf r_i)/|\\mathbf r-\\mathbf r_i|^3$ -- exact, no
approximation. **Assumptions:** static charges (no time dependence, no radiation), vacuum
permittivity $\\epsilon_0$ (no dielectric yet -- that's Part 5), point/line/ring idealizations
of real distributed charge. **Units:** $V$ in volts, $E$ in V/m, $\\rho$ in C/m$^3$ (or C/m
for a line, C for a point).

**What would I actually measure in a lab?** A voltmeter reads $V$ (relative to a chosen
ground); an electrometer or a calibrated deflection (e.g. a charged pith ball on a torsion
balance) reads force, from which $E=F/q_{\\rm test}$. Neither instrument sees $\\rho$
directly -- it's always inferred from $V$ or $E$.

**MATLAB equivalent:** `meshgrid(x,y)` for the grid; `quiver(X,Y,Ex,Ey)` for field vectors;
`contour(X,Y,V)` for equipotentials.

**Questions to ask:** Is the reference point for $V$ at infinity, or (as for the infinite
line, where the field falls too slowly) does it have to be finite? What symmetry does each
configuration have, and does that make a shortcut (Part 3) available?"""))

cells.append(co("""from dgs.charge_configurations import coulomb_field, dipole_moment, dipole_field, leading_multipole

Q = 1e-9   # 1 nC point charge at the origin
E_at_1m = coulomb_field([Q], [[0, 0, 0]], [1.0, 0, 0])
print(f"point charge Q=1 nC:  E(1 m, 0, 0) = {E_at_1m} V/m   (|E|={np.linalg.norm(E_at_1m):.3f} V/m)")

# physical dipole: +/-1 nC separated by 2 cm along z
a = 0.02
charges, positions = [Q, -Q], [[0, 0, a/2], [0, 0, -a/2]]
p = dipole_moment(charges, positions)
print(f"\\ndipole p = {np.round(p,5)} C*m,  leading multipole: {leading_multipole(charges, positions)}")
E_axis = coulomb_field(charges, positions, [0, 0, 1.0])
E_equator = coulomb_field(charges, positions, [1.0, 0, 0])
print(f"exact field on axis (1 m):    {E_axis}   vs ideal point-dipole: {dipole_field(p, [0,0,1.0])}")
print(f"exact field on equator (1 m): {E_equator}   (opposite p, as the dipole formula predicts)")"""))

cells.append(co("""from griffiths.electrostatics import ring_field_axis, line_charge_potential, eps0 as eps0_g

lam_s, R_s, Z_s = sp.symbols('lambda R Z', positive=True)
Ez_ring = ring_field_axis(lam_s, R_s, Z_s)
print("charged ring (radius R, line charge lambda), on-axis field E_z(Z):")
display(Ez_ring)

s_s, s_ref_s = sp.symbols('s s_ref', positive=True)
V_line = line_charge_potential(lam_s, s_s, s_ref_s)
print("\\ninfinite line charge potential (needs a FINITE reference s_ref -- infinity diverges):")
display(V_line)

Ez_num = sp.lambdify(Z_s, Ez_ring.subs({lam_s: 1e-9, R_s: 0.05, eps0_g: 8.8541878128e-12}))
Zs = np.linspace(0.001, 0.3, 300)
plt.figure(figsize=(6, 3.2))
plt.plot(Zs*100, Ez_num(Zs))
plt.xlabel('height above ring center Z (cm)'); plt.ylabel('E_z (V/m)')
plt.title('on-axis field of a charged ring (peaks near Z = R/sqrt(2))')
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

cells.append(co("""# a simple two-electrode geometry: +Q / -Q point charges 4 cm apart
Qe = 2e-9
charges2, pos2 = [Qe, -Qe], [np.array([-0.02, 0, 0]), np.array([0.02, 0, 0])]
K = 8.9875517873681764e9
xs2 = np.linspace(-0.06, 0.06, 46); ys2 = np.linspace(-0.05, 0.05, 40)
Xg, Yg = np.meshgrid(xs2, ys2)
Ex2 = np.full_like(Xg, np.nan); Ey2 = np.full_like(Xg, np.nan); Vg = np.full_like(Xg, np.nan)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        pt = np.array([Xg[i, j], Yg[i, j], 0.0])
        if min(np.linalg.norm(pt - p) for p in pos2) < 0.004:
            continue   # skip points right on top of a charge (field diverges)
        Ef = coulomb_field(charges2, pos2, pt)
        Ex2[i, j], Ey2[i, j] = Ef[0], Ef[1]
        Vg[i, j] = sum(K*q/np.linalg.norm(pt - p) for q, p in zip(charges2, pos2))

fig, ax = plt.subplots(figsize=(6, 5))
ax.contour(Xg*100, Yg*100, Vg, levels=25, cmap='RdBu_r')
skip = 3
ax.quiver(Xg[::skip, ::skip]*100, Yg[::skip, ::skip]*100, Ex2[::skip, ::skip], Ey2[::skip, ::skip], width=0.003)
ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
ax.set_title('two-electrode geometry: equipotentials (color) + field vectors')
plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 3 -- Gauss's law
# ============================================================================
cells.append(md("""## Part 3 -- Gauss's law and closed surfaces

$$\\oint_S \\mathbf E\\cdot d\\mathbf A = \\frac{Q_{\\rm enc}}{\\epsilon_0}.$$

The flux through **any** closed surface depends only on the enclosed charge. A sphere is
the natural surface for a point charge because $E$ is constant in magnitude and everywhere
parallel to $d\\mathbf A$ on it -- the integral becomes $E \\cdot 4\\pi r^2$, no calculus
needed. `dgs.gauss_law.numerical_flux` verifies the law with **zero symmetry assumed**: it
numerically integrates the exact Coulomb field over a sampled sphere.

**What would I actually measure in a lab?** You cannot measure "flux" directly with one
instrument; you'd map $\\mathbf E$ (or $V$, and take $-\\nabla V$) at many points on a real
closed surface (e.g. a Faraday cage's outer skin) and numerically sum $\\mathbf E\\cdot
\\hat n\\, \\Delta A$ -- exactly what the convergence study below does.

**MATLAB equivalent:** `sum(dot(E, N, 2) .* dA)` for the discretized flux integral.

**Questions to ask:** Does the numerical estimate converge to the theoretical value as the
mesh refines? Does it go to *exactly* zero for charge placed **outside** the surface (the
whole content of the law, not just "flux falls off with distance")?"""))

cells.append(co("""from dgs.gauss_law import gauss_flux, point_charge_field, line_charge_field, sheet_field, numerical_flux

Q3 = 1e-9
print(f"Gauss's law: flux through ANY closed surface = Q_enc/eps0 = {gauss_flux(Q3):.3f} V*m")
print(f"point charge field at r=1m:            {point_charge_field(Q3, 1.0):.3f} V/m   (~1/r^2)")
print(f"infinite line (lambda=1nC/m) at r=1m:  {line_charge_field(1e-9, 1.0):.3f} V/m   (~1/r)")
print(f"infinite sheet (sigma=1nC/m^2):        {sheet_field(1e-9):.3f} V/m   (uniform, independent of r)")"""))

cells.append(co("""ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
theo = gauss_flux(Q3)
rows = []
for n in ns:
    num = numerical_flux([Q3], [[0, 0, 0]], [0, 0, 0], 1.0, n=n)
    rows.append({"grid size n": n, "numerical flux": num, "theoretical flux": theo,
                 "relative error": abs(num - theo)/abs(theo)})
df_flux = pd.DataFrame(rows)
display(df_flux)

fx = numerical_flux([Q3], [[2.0, 0, 0]], [0, 0, 0], 1.0, n=30000)
print(f"charge OUTSIDE the sphere: flux = {fx:.2e}  (should be ~0 -- what goes in comes out)")

plt.figure(figsize=(5.5, 3.5))
plt.loglog(df_flux["grid size n"], df_flux["relative error"], 'o-')
plt.xlabel('n (quadrature points on the sphere)'); plt.ylabel('relative error')
plt.title("Gauss's law: numerical-flux convergence")
plt.grid(alpha=0.3, which='both'); plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 4 -- Laplace / Poisson
# ============================================================================
cells.append(md("""## Part 4 -- Potentials, Laplace, and Poisson equations

$$\\nabla^2 V = -\\frac{\\rho}{\\epsilon_0}, \\qquad \\text{charge-free: } \\nabla^2 V = 0.$$

The chain from PDE to answer:

$$\\boxed{\\text{PDE} \\to \\text{finite differences} \\to \\text{sparse matrix} \\to \\text{linear solver}}$$

Discretizing on a grid turns the 5-point stencil
$(V_{i+1,j}+V_{i-1,j}+V_{i,j+1}+V_{i,j-1}-4V_{i,j})/h^2 = -\\rho_{i,j}/\\epsilon_0$
into **one linear equation per grid point** -- a big sparse matrix equation $A\\mathbf v =
\\mathbf b$. Every module elsewhere in this repo solves Laplace **analytically** (separation
of variables, multipole series) -- powerful only where the geometry has enough symmetry.
Two electrode strips in a grounded box has none, so `dgs.poisson_2d` (built for this
notebook -- no such general-purpose solver existed in the repo before) assembles the sparse
system with `scipy.sparse` and solves it with `scipy.sparse.linalg.spsolve`, matching the
pattern already used for a different PDE in `dgs/grill_heat_equation.py`.

**What would I actually measure in a lab?** A probe (or a resistive-paper analog board, the
classic undergrad lab for this exact equation) reads $V$ at grid points; $\\mathbf E=-\\nabla
V$ is then a finite difference of the *measured* data, same as Part 1's "what would I
measure" note.

**MATLAB equivalent:** `A = delsq(numgrid('S',n))` builds the same 5-point-stencil sparse
Laplacian; `A\\b` (backslash) is `spsolve`.

**Questions to ask:** Does refining the grid reduce the error, and at what *rate* (order of
convergence)? Are the boundary conditions physically consistent with each other (checked
below by comparing to the exact linear parallel-plate solution)?"""))

cells.append(co("""from griffiths.vectors import grad, div

Vf = sp.Function('V')(x, y, z)
rho_f = sp.Function('rho')(x, y, z)
eps0_s = sp.Symbol('epsilon_0', positive=True)
print("Poisson's equation:"); display(sp.Eq(div(grad(Vf)), -rho_f/eps0_s))
print("\\ncharge-free (Laplace's equation):"); display(sp.Eq(div(grad(Vf)), 0))"""))

cells.append(co("""from dgs.poisson_2d import solve_poisson, field_from_potential, parallel_plate_boundary

nxp, nyp, Lp, V0p = 61, 61, 1.0, 1.0
dxp = dyp = Lp/(nxp - 1)
rho_pp, mask_pp, vals_pp = parallel_plate_boundary(nxp, nyp, V0=V0p)
Vgrid = solve_poisson(rho_pp, nxp, nyp, dxp, dyp, mask_pp, vals_pp)
Exg, Eyg = field_from_potential(Vgrid, dxp, dyp)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
im = axs[0].imshow(Vgrid, origin='lower', extent=[0, Lp, 0, Lp], cmap='RdBu_r')
axs[0].set_title('V(x,y): sparse finite-difference Poisson solve  (A v = b)')
plt.colorbar(im, ax=axs[0], label='V (volts)')
sk = 6
axs[1].quiver(np.linspace(0, Lp, nxp)[::sk], np.linspace(0, Lp, nyp)[::sk], Exg[::sk, ::sk], Eyg[::sk, ::sk])
axs[1].set_title(f'E = -grad(V)   interior |Ex| mean={np.abs(Exg[5:-5,5:-5]).mean():.3f} V/m '
                  f'(theory 2V0/L={2*V0p/Lp:.3f})')
plt.tight_layout(); plt.show()"""))

cells.append(co("""from dgs.poisson_2d import manufactured_solution

resolutions = [11, 21, 41, 81, 161]
rows4 = []
for n in resolutions:
    h = 1.0/(n - 1)
    rho_m, mask_m, vals_m, V_exact = manufactured_solution(n, n, 1.0, 1.0)
    V_num = solve_poisson(rho_m, n, n, h, h, mask_m, vals_m)
    rows4.append({"grid n": n, "h": h, "max error": np.abs(V_num - V_exact).max()})
df_pde = pd.DataFrame(rows4)
df_pde["order (log2 ratio)"] = [np.nan] + list(
    np.log(df_pde["max error"].values[:-1] / df_pde["max error"].values[1:]) / np.log(2))
display(df_pde)

plt.figure(figsize=(5, 3.5))
plt.loglog(df_pde['h'], df_pde['max error'], 'o-', label='measured')
plt.loglog(df_pde['h'], df_pde['max error'].iloc[0]*(df_pde['h']/df_pde['h'].iloc[0])**2, '--', label='O(h^2) reference')
plt.xlabel('grid spacing h'); plt.ylabel('max error'); plt.legend(); plt.grid(alpha=0.3, which='both')
plt.title('manufactured-solution convergence: 2nd-order finite differences')
plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 5 -- Fields in matter
# ============================================================================
cells.append(md("""## Part 5 -- Fields in matter

$$\\mathbf P = \\epsilon_0\\chi_e\\mathbf E, \\qquad \\mathbf D = \\epsilon_0\\mathbf E + \\mathbf P.$$

A dielectric polarizes in response to $\\mathbf E$: **bound** charge appears at its surface
($\\sigma_b=\\mathbf P\\cdot\\hat n$) and, if $\\mathbf P$ is nonuniform, in its volume
($\\rho_b=-\\nabla\\cdot\\mathbf P$). $\\mathbf D$ is defined so that Gauss's law for
$\\mathbf D$ only sees **free** charge -- the bound charge is already absorbed into $\\mathbf
D$'s definition. **Assumption:** linear, isotropic dielectric ($\\mathbf P\\parallel\\mathbf
E$, constant $\\chi_e$) -- this stays at the classical/macroscopic level; no quantum-chemical
polarizability mechanism is modeled.

**What would I actually measure in a lab?** Capacitance with vs. without a dielectric slab
between the plates ($C=\\epsilon_r C_0$) is the standard bench measurement of $\\epsilon_r$ --
no direct measurement of bound charge exists; it's inferred.

**MATLAB equivalent:** trivial scalar algebra here (`epsr = 1 + chi_e;`), so no dedicated
numerical routine -- the physics is in the symbolic relations, not a solver.

**Questions to ask:** Is there any FREE charge in this problem (there isn't, in the sphere
example below) -- if not, all the polarization is a *response*, not a *cause*. Does the
screening factor make physical sense as $\\epsilon_r\\to 1$ (should $\\to 1$, no screening)
and $\\epsilon_r\\to\\infty$ (should $\\to 0$, perfect screening, the conductor limit)?"""))

cells.append(co("""from griffiths.dielectrics import bound_surface_charge, displacement_field, clausius_mossotti, dielectric_sphere_in_field

P_s, theta_s = sp.symbols('P theta')
print("bound surface charge sigma_b = P . n_hat (uniformly polarized sphere):")
display(bound_surface_charge(P_s, theta_s))

E_s = sp.Symbol('E')
print("\\nD = eps0 E + P:"); display(displacement_field([E_s, 0, 0], [P_s, 0, 0]).T)

N_s, alpha_s = sp.symbols('N alpha', positive=True)
print("\\nClausius-Mossotti: eps_r(N, alpha) ="); display(clausius_mossotti(N_s, alpha_s))"""))

cells.append(co("""E0_s = sp.Symbol('E_0', positive=True)
print("dielectric sphere in a uniform field E0: interior field is UNIFORM, screened by 3/(eps_r+2)\\n")
for eps_r_val in (1.0, 2.0, 10.0, 1000.0):
    sol = dielectric_sphere_in_field(eps_r_val, E0_s)
    screening = float(sp.simplify(sol['E_in_over_E0']))
    print(f"eps_r = {eps_r_val:7.1f}:  E_in/E0 = {screening:.4f}   "
          f"({'no screening' if eps_r_val==1 else f'{screening*100:.1f}% of E0 survives inside'})")"""))

# ============================================================================
# PART 6 -- Maxwell (kept from the previous version of this notebook)
# ============================================================================
cells.append(md("""## Part 6 -- From electrostatics to Maxwell's equations

$$\\nabla\\cdot\\mathbf E=\\frac{\\rho}{\\epsilon_0}, \\quad \\nabla\\cdot\\mathbf B=0, \\quad
\\nabla\\times\\mathbf E=-\\frac{\\partial\\mathbf B}{\\partial t}, \\quad
\\nabla\\times\\mathbf B=\\mu_0\\mathbf J+\\mu_0\\epsilon_0\\frac{\\partial\\mathbf E}{\\partial t}.$$

Two **divergence** laws (where the fields come from) and two **curl** laws (how they drive
each other in time). Take $E=E(z,t)\\hat x$, $B=B(z,t)\\hat y$, no free charge or current:
Faraday and Ampere become two coupled first-order PDEs; cross-differentiate to eliminate $B$
and the electromagnetic **wave equation** appears, giving
$c=1/\\sqrt{\\mu_0\\epsilon_0}$ -- dimensional analysis: $[\\mu_0][\\epsilon_0]=$ s$^2$/m$^2$,
so $1/\\sqrt{\\mu_0\\epsilon_0}$ is a velocity, and numerically it comes out to $2.998\\times
10^8$ m/s.

**What would I actually measure in a lab?** None of these fields are measured "in the
abstract" -- what's actually measured is the force on a test charge (giving $E$), a torque
on a current loop or Hall-probe voltage (giving $B$), or (Part 10) the current a photodiode
outputs when light hits it.

**MATLAB equivalent:** these are symbolic derivations here (SymPy); MATLAB's Symbolic Math
Toolbox (`syms`, `diff`, `curl`) does the equivalent algebra."""))

cells.append(co("""for name, eqn in ed.maxwell_equations().items():
    print(f"{name:10s}:", eqn)"""))

cells.append(co("""wave, steps = ed.wave_equation_1d()
print("Faraday:", steps['faraday'])
print("Ampere :", steps['ampere'])
wave"""))

cells.append(co("""disp, k_w, n = ed.plane_wave_dispersion()
print("dispersion relation:", disp)
print("k(omega) =", k_w)
print("refractive index n =", n)
Z0 = ed.wave_impedance(medium=False)
print("vacuum impedance Z0 =", Z0, "=",
      float(Z0.subs({ed.mu0: 4e-7*np.pi, ed.eps0: 8.8541878128e-12})), "ohm")"""))

cells.append(co("""J, Sig = ed.ohms_law_tensor([[sp.Symbol('s_xx'), sp.Symbol('s_xy'), 0],
                               [sp.Symbol('s_yx'), sp.Symbol('s_yy'), 0],
                               [0, 0, sp.Symbol('s_zz')]])
print("sigma ="); sp.pprint(Sig)
print("J = sigma . E ="); sp.pprint(J)
print("Drude sigma(omega) =", ed.drude_conductivity())"""))

cells.append(co("""E0 = sp.Symbol('E_0', positive=True)
print("<S> =", ed.time_average_poynting(E0))

t_, x_field, A_field, phi_field = dg.make_field(N=1024, seed=3)
I_field = np.abs(x_field)**2
plt.figure(figsize=(9, 3))
plt.plot(t_, ed.to_decibels(I_field, ref=I_field.max()), lw=1.3)
plt.ylim(-40, 2); plt.xlabel('time t'); plt.ylabel('intensity (dB rel. peak)')
plt.title('a square-law detector reads intensity in decibels  (10 log10 I/I_max)')
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 7 -- Complex waves and k-vectors
# ============================================================================
cells.append(md("""## Part 7 -- Complex waves and wave vectors

$$U(\\mathbf r,t) = A\\, e^{i(\\mathbf k\\cdot\\mathbf r - \\omega t)}, \\qquad
|\\mathbf k| = \\frac{2\\pi}{\\lambda}, \\qquad e^{i\\theta}=\\cos\\theta + i\\sin\\theta \\;(\\text{Euler}).$$

$\\mathbf k$ points along propagation; its magnitude is fixed by $\\lambda$ regardless of
direction. Writing the field complex (rather than $\\cos(\\cdot)$ alone) buys algebra: real
and imaginary parts are two physical snapshots a quarter period apart -- and a detector that
only measures $|U|^2$ (Part 10) cannot tell them apart, which is exactly the ambiguity Part
12/17's phase retrieval resolves.

**What would I actually measure in a lab?** Interferometry (mixing the field with a
reference) can extract the phase directly; a bare photodiode cannot -- it only reads
$|U|^2$, discussed further in Part 10.

**MATLAB equivalent:** `exp(1i*theta)` for the complex exponential; `quiver` for the
$\\mathbf k$-vector plot below.

**Questions to ask:** Given only $|U(t)|^2$, is $\\phi(t)$ recoverable at all (Part 8's
shift-invariance theorem says NOT from one measurement alone)? What physically fixes
$|\\mathbf k|$ (the wavelength) versus what's free (the direction)?"""))

cells.append(co("""from dgs.taylor import taylor_series

theta_s7 = sp.Symbol('theta')
lhs_euler = taylor_series(sp.exp(sp.I*theta_s7), theta_s7, 0, 8)
rhs_euler = taylor_series(sp.cos(theta_s7), theta_s7, 0, 8) + sp.I*taylor_series(sp.sin(theta_s7), theta_s7, 0, 8)
print("Euler's formula as two interleaved Taylor series (order 8 truncation):")
print("e^{i theta} - (cos theta + i sin theta) simplifies to:", sp.simplify(lhs_euler - rhs_euler))"""))

cells.append(co("""from dgs.vector_calculus import complex_field_2d

res7 = complex_field_2d(field_type='EM_wave', N=40)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, key, title in zip(axs, ['Ereal_y', 'Eimag_y', 'amplitude', 'phase'],
                           ['Re[E_y] (bright side)', 'Im[E_y] (dark side)', '|E|', 'phase']):
    im = ax.imshow(res7[key], extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower',
                    cmap='RdBu_r' if 'E' in key else 'viridis')
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(res7['title']); plt.tight_layout(); plt.show()
print(res7['dark_side_lesson'])"""))

cells.append(co("""lam0_7 = 500e-9
k0_7 = 2*np.pi/lam0_7
angles7 = np.deg2rad([0, 30, 90, 150, 225])
fig, ax = plt.subplots(figsize=(5, 5))
for ang in angles7:
    ax.arrow(0, 0, np.cos(ang), np.sin(ang), head_width=0.06, length_includes_head=True)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal')
ax.set_title('propagation directions k_hat  (|k| = 2*pi/lambda is fixed, only direction varies)')
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
print(f"|k| = 2*pi/lambda = {k0_7:.3e} rad/m for lambda={lam0_7*1e9:.0f} nm -- same for all 5 arrows above")"""))

# ============================================================================
# PART 8 -- Fourier optics
# ============================================================================
cells.append(md("""## Part 8 -- Fourier optics

In the Fraunhofer (far-field) limit, the diffracted amplitude IS the Fourier transform of
the aperture: $\\tilde U(k_x,k_y)=\\mathcal F\\{U(x,y)\\}$, and a detector reads $|\\tilde
U|^2$. `dgs.fourier_optics`/`dgs.diffraction` already prove two exact 1D theorems used here
in 2D: **reciprocal scaling** (width $\\times\\sin\\theta_{\\rm null}=\\lambda$, so a
narrower aperture diffracts wider) and **shift-invariant intensity** (sliding the aperture
multiplies its transform by a pure phase, leaving $|\\tilde U|^2$ unchanged -- the aperture's
*position* is lost, exactly the phase-retrieval problem this repo's receiver solves, just in
time instead of space).

**Assumptions:** scalar diffraction theory (polarization ignored), far-field/Fraunhofer
(equivalently: a single Fourier-transforming lens).

**What would I actually measure in a lab?** A CCD/CMOS sensor in the focal plane of a lens
reads $|\\tilde U|^2$ -- the magnitude spectrum below -- but has NO way to read the phase
spectrum directly; phase-sensitive detection (interferometry, or the phase-retrieval
algorithms of Parts 12/13/17) is required to recover it.

**MATLAB equivalent:** `fft2`, `fftshift`, `ifft2` -- identical calls, identical semantics.

**Questions to ask:** Does a smaller aperture blur MORE or LESS (reciprocal scaling says
more)? Does the phase mask change the energy at each spatial frequency, or only its phase
(the latter -- exactly why it's invisible to an intensity-only sensor)?"""))

cells.append(co("""from dgs.imaging_inverse import synthetic_object

obj8 = synthetic_object(96)
F_obj = np.fft.fftshift(np.fft.fft2(obj8))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
axs[0].imshow(obj8, cmap='gray'); axs[0].set_title('object plane  U(x,y)')
axs[1].imshow(np.log1p(np.abs(F_obj)), cmap='viridis'); axs[1].set_title('|U~(kx,ky)|  (log scale)')
axs[2].imshow(np.angle(F_obj), cmap='twilight'); axs[2].set_title('phase of U~(kx,ky)')
for a in axs: a.axis('off')
plt.tight_layout(); plt.show()"""))

cells.append(co("""n8 = obj8.shape[0]
ky8, kx8 = np.mgrid[-n8//2:n8//2, -n8//2:n8//2]
kr8 = np.sqrt(kx8**2 + ky8**2)

def refocus(mask):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(F_obj*mask)))

circ_mask = (kr8 <= n8//6).astype(float)
rect_mask = ((np.abs(kx8) <= n8//6) & (np.abs(ky8) <= n8//6)).astype(float)
lowpass_mask = (kr8 <= n8//4).astype(float)
phase_mask = np.exp(1j*0.002*(kx8**2 + ky8**2))   # a quadratic (defocus-like) phase mask -- passes ALL frequencies

fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, mask, title in zip(axs, [circ_mask, rect_mask, lowpass_mask, phase_mask],
                            ['circular aperture', 'rectangular aperture', 'wider low-pass', 'quadratic phase mask']):
    ax.imshow(refocus(mask), cmap='gray'); ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show()
print("apertures REMOVE high spatial frequencies -> blur + ringing (Gibbs, from the hard cutoff).")
print("the phase mask keeps every frequency's ENERGY but scrambles its phase -> still blurs the")
print("image, but an intensity-only sensor cannot tell 'phase scrambled' from 'frequencies removed'.")"""))

# ============================================================================
# PART 9 -- Dispersion Taylor series
# ============================================================================
cells.append(md("""## Part 9 -- Dispersion and Taylor series

$$\\beta(\\omega) = \\beta_0 + \\beta_1(\\omega-\\omega_0) + \\frac{\\beta_2}{2}(\\omega-\\omega_0)^2
+ \\frac{\\beta_3}{6}(\\omega-\\omega_0)^3 + \\cdots$$

$\\beta_0\\to$ a constant phase (removable); $\\beta_1=1/v_g\\to$ pure group delay (removable
by a moving frame); $\\beta_2=d^2\\beta/d\\omega^2\\to$ **group-velocity dispersion**, the
first term that actually *reshapes* the pulse. This is ordinary-calculus Taylor expansion
applied to a propagation constant -- and it is exactly the algebra behind Part 6's bridge
$D=2\\pi\\beta_2 L$.

**Assumptions:** the expansion is only valid near $\\omega_0$ (narrowband pulse); higher
orders ($\\beta_3,\\ldots$) are dropped here, which is the **truncation error** of the model.

**What would I actually measure in a lab?** An autocorrelator or FROG measures pulse width
before/after a known length of fiber; fitting the width growth vs. propagation distance to
$\\propto\\sqrt{1+(z/z_0)^2}$ gives $\\beta_2$ experimentally, without ever measuring
$\\beta(\\omega)$ directly.

**MATLAB equivalent:** `diff(beta_sym, omega, k)` for symbolic derivatives (Symbolic Math
Toolbox); `fft`/`ifft` for the pulse-propagation sweep below.

**Questions to ask:** Does the pulse spread MORE for larger $|\\beta_2 L|$ (yes, verified
below)? Is $\\beta_2$'s sign (normal vs. anomalous dispersion) visible in the *magnitude* of
spreading alone (no -- both signs spread a transform-limited pulse identically; the sign
only shows up when the input is already chirped)?"""))

cells.append(co("""from dgs.taylor import dispersion_taylor

omega_s9, omega0_s9 = sp.symbols('omega omega_0', positive=True)
n0_s9, dn_s9 = sp.symbols('n_0 dn_domega', positive=True)
c_sym9 = sp.Symbol('c', positive=True)
beta_expr9 = (omega_s9/c_sym9) * (n0_s9 + dn_s9*(omega_s9 - omega0_s9))   # toy linear-index model
betas9 = dispersion_taylor(beta_expr9, omega_s9, omega0_s9, n=3)
for lab, b in zip(['beta_0 (phase)', 'beta_1 (1/v_g, group delay)', 'beta_2 (GVD)', 'beta_3'], betas9):
    print(lab, "="); display(sp.simplify(b))"""))

cells.append(co("""from dgs.em_dispersion import disperse_pulse, pulse_width

t9 = np.linspace(-40, 40, 4096)
pulse0 = np.exp(-t9**2/(2*3.0**2))
beta2Ls = [0, 50, 200, 800]
fig, ax = plt.subplots(figsize=(7, 3.6))
for b2L in beta2Ls:
    out9 = disperse_pulse(pulse0, t9, b2L, 1.0)
    ax.plot(t9, np.abs(out9)**2, label=f'beta2*L={b2L}  (width={pulse_width(t9, out9):.2f})')
ax.set_xlabel('t'); ax.set_ylabel('|E(t)|^2'); ax.legend(fontsize=8)
ax.set_title('a transform-limited pulse spreads under GVD  (Part 6 bridge: D = 2*pi*beta2*L)')
plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 10 -- Photodetector electronics
# ============================================================================
cells.append(md("""## Part 10 -- Photodetector and electronics

$$P_{\\rm opt}(t) \\to i_{\\rm PD}(t) = R_\\lambda P_{\\rm opt}(t) \\to V_{\\rm out}(t) = -R_f\\, i_{\\rm PD}(t).$$

$R_\\lambda=\\eta q\\lambda/(hc)$ is the responsivity (A/W); $R_f$ the transimpedance
amplifier's feedback resistor (the gain, V/A). The noise budget is a quadrature sum of shot
noise ($\\propto\\sqrt{I_{\\rm ph}}$, signal-dependent), thermal/Johnson noise of $R_f$
($\\propto 1/\\sqrt{R_f}$ -- bigger $R_f$ is QUIETER), and amplifier voltage noise acting
through the input capacitance. **The central trade-off:** $R_f\\cdot f_{3\\rm dB}=1/(2\\pi
C)$ is fixed by the capacitance alone -- gain and bandwidth are bought from the same budget.

**What would I actually measure in a lab?** An oscilloscope on the TIA output reads
$V_{\\rm out}(t)$ directly; the noise floor (scope in the dark, or averaged many traces)
gives the RMS noise current via $i_n=V_{n,\\rm rms}/R_f$, which can be compared against the
shot/thermal/amplifier budget below.

**MATLAB equivalent:** plain scalar/array arithmetic (`I_ph = Rlambda .* P_opt;`) -- no
dedicated toolbox function needed for these algebraic relations.

**Questions to ask:** At the chosen $P_{\\rm opt}$, is the receiver shot-noise-limited (the
ideal) or thermal/amplifier-limited? What does raising $R_f$ do to both the SNR and the
bandwidth simultaneously?"""))

cells.append(co("""from dgs.transimpedance_amplifier import responsivity, photocurrent, output_voltage, bandwidth_3db

lam_nm10, eta_qe10 = 1550.0, 0.85
Rlam10 = responsivity(lam_nm10, eta_qe10)
t10 = np.linspace(0, 2e-9, 500)
P_opt_t = 50e-6*(1 + 0.8*np.sin(2*np.pi*1e9*t10))   # 50 uW average optical power, 1 GHz tone
I_pd_t = Rlam10 * P_opt_t   # photocurrent() checks P_opt<0 with a scalar `if`, so apply the I=R*P formula directly for arrays
Rf10 = 2e4
V_out_t = output_voltage(I_pd_t, Rf10)
f3db10 = bandwidth_3db(Rf10, 0.5e-12)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
axs[0].plot(t10*1e9, P_opt_t*1e6); axs[0].set_xlabel('t (ns)'); axs[0].set_ylabel('P_opt (uW)')
axs[0].set_title(f'optical power  (R_lambda={Rlam10:.3f} A/W)')
axs[1].plot(t10*1e9, V_out_t*1e3); axs[1].set_xlabel('t (ns)'); axs[1].set_ylabel('V_out (mV)')
axs[1].set_title(f'TIA output  (R_f={Rf10/1e3:.0f} kOhm, f_3dB={f3db10/1e6:.0f} MHz)')
plt.tight_layout(); plt.show()"""))

cells.append(co("""from dgs.transimpedance_amplifier import (shot_noise_current, thermal_noise_current, amplifier_noise_current,
                                          total_noise_current, snr, noise_equivalent_power, sensitivity)

P_avg10, C10, en10, B10 = 50e-6, 0.5e-12, 2e-9, 1e9
i_shot = shot_noise_current(photocurrent(P_avg10, Rlam10), B10)
i_therm = thermal_noise_current(Rf10, B10)
i_amp = amplifier_noise_current(en10, C10, B10)
i_tot = total_noise_current(P_avg10, Rlam10, Rf10, C10, B10, e_n=en10)
plt.figure(figsize=(5, 3.2))
plt.bar(['shot', 'thermal', 'amplifier', 'total'], np.array([i_shot, i_therm, i_amp, i_tot])*1e9)
plt.ylabel('RMS current noise (nA)')
plt.title(f'noise budget @ B={B10/1e9:.0f} GHz  (SNR={snr(P_avg10, Rlam10, Rf10, C10, B10, e_n=en10):.1f})')
plt.tight_layout(); plt.show()
print(f"NEP = {noise_equivalent_power(Rlam10, Rf10, C10, B10, e_n=en10)*1e9:.3f} nW,   "
      f"sensitivity(SNR=7) = {sensitivity(7, Rlam10, Rf10, C10, B10, e_n=en10)*1e9:.1f} nW")"""))

# ============================================================================
# PART 11 -- Sampling / ADC
# ============================================================================
cells.append(md("""## Part 11 -- Sampling and computer engineering

**Nyquist-Shannon:** a signal band-limited to $f_{\\rm max}$ is fully reconstructible from
samples at $f_s\\ge 2f_{\\rm max}$. Below that, higher frequencies fold down and masquerade
as lower ones -- **aliasing**, shown below for a 100 Hz tone at 3x, exactly 2x, and 1.3x
sampling.

$$\\text{optical waveform} \\to \\text{ADC samples} \\to \\text{NumPy array} \\to \\text{C buffer}$$

A 2D image flattens to a 1D array with `index = y*W + x` (row-major); a fixed-resolution ADC
sample stream maps directly onto a C buffer:

```c
uint16_t samples[N];   // 16-bit ADC, one sample per element
```

**What would I actually measure in a lab?** An oscilloscope's own ADC does exactly this
pipeline; its datasheet ENOB (effective number of bits) is measured the same way as the
independent quantization-noise simulation below, not just quoted from the bit count.

**MATLAB equivalent:** `Fs`/`decimate`/`resample` for sample-rate changes; the SNR-vs-bits
formula is the same `6.02*N+1.76` used by every ADC datasheet.

**Questions to ask:** At 1.3x the signal frequency, does the sampled/reconstructed signal
look like a DIFFERENT (lower) frequency (yes -- that's the alias)? Does adding 2 bits of
resolution buy roughly 12 dB more SNR (the formula predicts exactly that)?"""))

cells.append(co("""from dgs.adc import ADC, sinusoidal

f_sig11 = 100.0
t11, analog11 = sinusoidal(f=f_sig11, duration=0.05)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, fs11, label in zip(axs, [300.0, 200.0, 130.0], ['fs=3f  (OK)', 'fs=2f  (exactly Nyquist)', 'fs=1.3f  (ALIASED)']):
    a11 = ADC(n_bits=12, fs=fs11)
    ts11, q11 = a11.convert(t11, analog11)
    ax.plot(t11*1e3, analog11, lw=1, alpha=0.6, label='analog')
    ax.step(ts11*1e3, q11, where='post', lw=1.3, label='sampled')
    ax.set_title(label); ax.legend(fontsize=7); ax.set_xlabel('t (ms)')
plt.suptitle('Nyquist-Shannon aliasing (100 Hz signal, three sample rates)'); plt.tight_layout(); plt.show()"""))

cells.append(co("""from dgs.adc_snr_bits import theoretical_snr_db, simulate_adc_snr

rows11 = []
for nb in (4, 6, 8, 10, 12, 16):
    a11b = ADC(n_bits=nb, fs=2000.0)
    ts11b, q11b = a11b.convert(t11, analog11)
    x_at11 = np.interp(ts11b, t11, analog11)
    rows11.append({"bits": nb, "theory 6.02N+1.76 (dB)": theoretical_snr_db(nb),
                    "ADC-class measured (dB)": a11b.sqnr_db(x_at11, q11b),
                    "independent simulation (dB)": simulate_adc_snr(nb)})
df_adc = pd.DataFrame(rows11)
display(df_adc)"""))

cells.append(co("""Wd, Hd = 8, 5
img2d = np.arange(Wd*Hd).reshape(Hd, Wd)
flat11 = img2d.reshape(-1)
y_idx, x_idx = 3, 5
assert flat11[y_idx*Wd + x_idx] == img2d[y_idx, x_idx]
print(f"img2d[{y_idx},{x_idx}] = {img2d[y_idx, x_idx]}   ==   flat[y*W+x] = flat[{y_idx*Wd+x_idx}] = {flat11[y_idx*Wd+x_idx]}")
print("this row-major flatten is exactly how a 2D sensor frame becomes one contiguous C buffer.")"""))

# ============================================================================
# PART 12 -- Inverse problems
# ============================================================================
cells.append(md("""## Part 12 -- Computational imaging forward model

$$\\mathbf y = H\\mathbf x + \\mathbf n, \\qquad
\\hat{\\mathbf x} = \\arg\\min_{\\mathbf x}\\left[\\|H\\mathbf x - \\mathbf y\\|_2^2 + \\lambda\\|\\mathbf x\\|_2^2\\right].$$

$\\mathbf x$ is the true object, $H$ a (here Gaussian) blur operator, $\\mathbf n$ additive
noise, $\\mathbf y$ the measurement -- `dgs.inverse_calculus.inverse_problem_framework()`
places this in the same family as this repo's own phase retrieval (GS = alternating
projections minimizing the same kind of objective). Naive inversion ($\\lambda=0$, divide by
$H$ in Fourier space) blows up wherever $|H|$ is small because noise divides by the same
tiny number as signal -- **ill-posed**. Tikhonov regularization trades bias (a smoother,
slightly wrong answer) for variance (noise suppression); `dgs.imaging_inverse` (built for
this notebook) implements the 2D version of the 1D Wiener-filter demo already in
`dgs.inverse_calculus.deconvolution_demo`.

**What would I actually measure in a lab?** A camera with a known point-spread function
(measured once, e.g. by imaging a pinhole) gives $H$; a single noisy exposure of the unknown
scene gives $\\mathbf y$ directly.

**MATLAB equivalent:** `deconvwnr(y, psf, lambda)` (Wiener/Tikhonov deconvolution, same
closed-form algebra as `tikhonov_deconvolve` below); `fft2`/`ifft2` for the manual version.

**Questions to ask:** Is there a $\\lambda$ that's clearly too small (noise-amplified,
"salt and pepper" reconstruction) and one clearly too large (over-smoothed, edges gone)?
Where exactly does the error-vs-$\\lambda$ curve bottom out, and is that minimum an INTERIOR
point (bias/variance tradeoff) rather than an endpoint?"""))

cells.append(co("""from dgs.imaging_inverse import gaussian_blur_kernel, apply_blur, add_gaussian_noise, tikhonov_deconvolve, reconstruction_error

x_true12 = synthetic_object(64)
kernel12 = gaussian_blur_kernel(9, sigma=1.5)
y_clean12 = apply_blur(x_true12, kernel12)
y_noisy12 = add_gaussian_noise(y_clean12, sigma=0.05, seed=1)

lambdas12 = np.logspace(-6, 1, 40)
errors12 = [reconstruction_error(x_true12, tikhonov_deconvolve(y_noisy12, kernel12, lam)) for lam in lambdas12]
best_lam12 = lambdas12[int(np.argmin(errors12))]
x_hat_best = tikhonov_deconvolve(y_noisy12, kernel12, best_lam12)

fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, im12, title in zip(axs, [x_true12, y_clean12, y_noisy12, x_hat_best],
                            ['true object x', 'blurred y=Hx', 'noisy measurement', 'Tikhonov reconstruction']):
    ax.imshow(im12, cmap='gray'); ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show()

plt.figure(figsize=(5.5, 3.5))
plt.loglog(lambdas12, errors12)
plt.axvline(best_lam12, ls='--', color='k', label=f'best lambda={best_lam12:.2e}')
plt.xlabel('lambda'); plt.ylabel('MSE(x_true, x_hat)'); plt.legend()
plt.title('naive inverse (small lambda) vs oversmoothed (large lambda): interior minimum')
plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 13 -- Differentiable physics (PyTorch)
# ============================================================================
cells.append(md("""## Part 13 -- Differentiable physics with PyTorch

"Differentiable optics" writes the forward model (a phase mask, then propagation, then a
detector) in ordinary tensor ops and lets autograd hand back $\\partial L/\\partial\\theta$
for free -- the chain rule applied automatically to a computation graph, instead of by hand.
`dgs.differentiable_optics_tutorial` stages this in four steps: (1) verify autograd against
a known derivative, (2) confirm gradients flow through a real optical forward model, (2b)
*design* a phase mask by gradient ascent (a tiny hologram), (4) recover phase from
intensity-only data by gradient descent and compare the result against classical
Gerchberg-Saxton on the identical problem.

**Assumptions:** the forward model (phase mask -> FFT -> intensity) is exactly the same
$H(f)=e^{i\\pi Df^2}$ operator used throughout this notebook -- differentiable optics and GS
are two different algorithms solving the SAME inverse problem, not two different physical
models.

**What would I actually measure in a lab?** Nothing new here vs. Part 12/17 -- the physical
measurement is still intensity-only; what's different is purely the *algorithm* used to
invert it.

**MATLAB equivalent:** MATLAB's Deep Learning Toolbox (`dlarray`, `dlgradient`) provides the
same automatic-differentiation machinery as `torch.autograd`.

**Questions to ask:** Does the autograd-computed derivative match the analytic/SymPy one to
numerical precision? Do gradient descent and GS converge to reconstructions with comparable
self-consistency error (judged against the MEASUREMENTS, not an unknown ground truth --
consistent with this repo's own verification philosophy)?"""))

cells.append(co("""import torch
from dgs.differentiable_optics_tutorial import (step1_scalar_autograd, step2_forward_model_is_differentiable,
                                                  step2b_design_a_focusing_phase_mask, step4_compare_to_gs)

s1 = step1_scalar_autograd()
x_sym13 = sp.Symbol('x')
sympy_deriv = float(sp.diff(x_sym13**2, x_sym13).subs(x_sym13, s1['x']))
h13 = 1e-4
fd_estimate = ((s1['x']+h13)**2 - (s1['x']-h13)**2) / (2*h13)
print(f"y = x^2 at x={s1['x']}:")
print(f"  autograd dy/dx = {s1['dy_dx_autograd']:.6f}")
print(f"  analytic  2x   = {s1['dy_dx_analytic']:.6f}")
print(f"  sympy diff(x^2)= {sympy_deriv:.6f}")
print(f"  finite diff    = {fd_estimate:.6f}")"""))

cells.append(co("""s2 = step2_forward_model_is_differentiable()
print("gradient flows through the forward optical model:", s2)

focus13 = step2b_design_a_focusing_phase_mask(n_iter=300)
print(f"\\nfocusing-mask design (gradient ASCENT on a lens/hologram):")
print(f"  target-pixel energy share {focus13['initial_share_at_target']:.4f} -> {focus13['final_share_at_target']:.4f}")"""))

cells.append(co("""cmp13 = step4_compare_to_gs()
print(f"GS (alternating projections) reconstruction MSE:  {cmp13['gs_reconstruction_mse']:.3e}")
print(f"gradient descent (autograd)  reconstruction MSE:   {cmp13['gd_reconstruction_mse']:.3e}")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(cmp13['phi_gs'], label='GS (alternating projections)')
ax.plot(cmp13['phi_gd'], label='gradient descent (autograd)', ls='--')
ax.set_xlabel('sample'); ax.set_ylabel('recovered phase (rad)'); ax.legend()
ax.set_title('two different algorithms, the identical inverse problem')
plt.tight_layout(); plt.show()"""))

# ============================================================================
# PART 14 -- Jalali connection
# ============================================================================
cells.append(md("""## Part 14 -- The Jalali-lab photonics connection

This notebook contains mathematical ingredients relevant to modern photonic time-stretch
work (Fourier transforms, dispersion, optical detection, high-speed sampling, inverse
problems), not a reproduction of any specific unpublished result. Three tiers, kept
distinct:

**1. Publicly documented work (verified via web search for this notebook, quoted with
sources).** F. Coppinger, A.S. Bhushan, B. Jalali, *"Photonic time stretch and its
application to analog-to-digital conversion,"* IEEE Trans. Microwave Theory Tech. **47**,
1309-1314 (1999): the paper demonstrated a photonic time-stretch preprocessor feeding a
**1 Gsample/s** electronic ADC, with stretch factors **M = 3, 6, and 8** shown. (Two
independent search-engine summaries of the abstract agree on these figures; the primary PDF
itself was not directly fetched, so treat this as secondary-source-verified, not
primary-source-read.) Sources:
[ResearchGate](https://www.researchgate.net/publication/3120820_Photonic_time_stretch_and_its_application_to_analog-to-digital_conversion),
[Nature Photonics review citing the same paper](https://www.nature.com/articles/nphoton.2017.76).

**Important discrepancy, flagged rather than silently repeated:** this repo's own
`dgs.coppinger_jalali_1999` module carries a docstring explicitly labeled *"PAPER ABSTRACT
(reproduced from memory)"* that instead states a 2 Gsample/s ADC and M=10 (20 GHz capture).
That does not match the verified abstract above -- it is used below only as this notebook's
own **illustrative worked example** (tier 3), not as a claim about the paper.

**2. General photonics/EE concepts used freely.** SMF-28 single-mode fiber's dispersion
parameter is independently well documented as $D\\approx 17\\,{\\rm ps/(nm\\cdot km)}$ near
1550 nm (any fiber-optics textbook or the Corning SMF-28 datasheet) -- this is a property of
the fiber, not of the 1999 paper specifically.

**3. This notebook's own educational simulation.** The $H(f)=e^{i\\pi Df^2}$ operator this
whole notebook builds toward, and the capstone in Part 17, are this repo's own construction
-- inspired by, but not a reproduction of, any lab's unpublished experimental setup.

**Questions to ask:** For any specific number quoted about a paper, is it from a source you
actually read, or "reproduced from memory"? Does the general physics (dispersion, GVD,
Fourier optics) hold regardless of whose specific experimental numbers are attached to it?"""))

cells.append(co("""from dgs.coppinger_jalali_1999 import maxwell_phasor_domain, coppinger_1999_stretch_factor

mx14 = maxwell_phasor_domain(freq_Hz=193e12, medium='SMF28')
print(f"SMF-28 @ 193 THz (1550 nm): n = {mx14['fields']['n_real']:.4f}  "
      f"(independently well-known value: ~1.4446)")
print(f"D (this module's own Taylor-expansion estimate) = {mx14['GVD']['D_ps_nm_km']:.1f} ps/(nm*km)  "
      f"(SMF-28 datasheet value: ~17 ps/(nm*km))")
print(f"internal self-consistency (Faraday's law):  error = {mx14['maxwell_verification']['faraday_error_frac']:.1e}")
print(f"internal self-consistency (Ampere's law):   error = {mx14['maxwell_verification']['ampere_error_frac']:.1e}")"""))

cells.append(co("""cp14 = coppinger_1999_stretch_factor(D1_ps_nm_km=17, L1_km=5, D2_ps_nm_km=17, L2_km=45,
                                     Delta_lambda_nm=10, f_ADC_GHz=1.0)   # f_ADC=1 GHz matches the VERIFIED paper value
print("this notebook's own illustrative worked example (tier 3 -- NOT quoted from the paper):")
print(f"  stretch factor M = {cp14['results']['M']:.1f}")
print(f"  captured RF bandwidth at a 1 Gsample/s ADC = {cp14['results']['B_RF_GHz']:.1f} GHz")
print()
print(cp14['H_f_connection'])"""))

# ============================================================================
# PART 17 -- Capstone
# ============================================================================
cells.append(md("""## Part 17 -- Final integrated capstone

$$\\boxed{\\text{synthetic optical object} \\to \\text{wave propagation} \\to \\text{dispersive transform}
\\to \\text{photodetector} \\to \\text{ADC} \\to \\text{digital samples} \\to \\text{inverse reconstruction}}$$

Every stage below reuses code already built earlier in this notebook or already tested
elsewhere in this repo: `dgs.dispersion_gs_prototype.make_field`/`disperse` (object + wave
propagation + the dispersive transform of Parts 6/9), `dgs.transimpedance_amplifier`
(Part 10's photodetector chain), `dgs.adc.ADC` (Part 11's sampling), and
`dgs.gs_core.retrieve_phase_with_history` (Part 12/13's inverse-problem family, this repo's
own canonical Gerchberg-Saxton implementation). **Assumption:** $|D|\\ge5000$ AND a unit-amplitude
(constant-envelope) signal are both required for GS to reliably converge in this
normalized-unit convention (see `dgs/gs_core.py`'s own kwarg-bounds warning) -- so the
object below is a synthetic QPSK optical-communication signal via
`gs_core.make_measurements`, this repo's own best-validated configuration, rather than an
arbitrary varying-amplitude pulse (tested separately: a varying-amplitude object still
converges GS to machine-precision self-consistency with the *measurements*, but can land on
a different phase that is equally consistent with intensity-only data -- a genuine
phase-retrieval ambiguity, not a bug, and a reason to prefer a well-conditioned signal for a
capstone demonstration).

**MATLAB pipeline sketch:**
```matlab
[t, x, A, phi] = make_field(N, seed);
x1 = ifft(fft(x) .* exp(1i*pi*D1*f.^2));   % disperse
x2 = ifft(fft(x) .* exp(1i*pi*D2*f.^2));
I1 = abs(x1).^2;  I2 = abs(x2).^2;          % square-law detector
Iph1 = Rlambda * P_peak * I1;  V1 = Iph1 * Rf;   % photodetector + TIA
V1q = quantize(V1, n_bits);                 % ADC
phi_rec = gerchberg_saxton(I1q, I2q, D1, D2, n_iter);
```

**Questions to ask:** Does the RMS phase error stay small even after passing through a
realistic (noisy, quantized) detector chain, not just the idealized clean measurements used
in Parts 6/9? Which stage of the pipeline dominates the final error -- photon shot noise, ADC
quantization, or the GS algorithm's own convergence limit?"""))

cells.append(co("""from dgs import gs_core
from dgs.transimpedance_amplifier import responsivity as tia_resp, output_voltage as tia_vout
from dgs.adc import ADC as ADC17
from dgs.dispersion_gs_prototype import compare_phase

# synthetic optical object: a QPSK optical-communication signal (this repo's own
# best-validated GS configuration -- unit_amplitude=True, |D|>=5000)
meas17 = gs_core.make_measurements(modulation='QPSK', n_symbols=64, sps=8,
                                    D1=-5000.0, D2=-5750.0, snr_db=25.0, rng_seed=0)
E17, phi17, D1_cap, D2_cap, unit_amp17 = (meas17['E'], meas17['phi_true'],
                                           meas17['D1'], meas17['D2'], meas17['unit_amplitude'])
t17 = meas17['t']
I1_true = np.abs(gs_core.disperse(E17, D1_cap))**2   # wave propagation + dispersive transform
I2_true = np.abs(gs_core.disperse(E17, D2_cap))**2

# photodetector: scale to physical optical power, run through responsivity + TIA
P_peak = 200e-6   # 200 uW peak optical power
Rlam17 = tia_resp(1550.0, 0.85); Rf17 = 1e4
# tia_iph()/photocurrent() checks P_opt<0 with a scalar `if`, so apply I=R*P directly for these arrays
V1 = tia_vout(Rlam17 * (P_peak*I1_true), Rf17)
V2 = tia_vout(Rlam17 * (P_peak*I2_true), Rf17)

# ADC: quantize at the field's own native sample rate (no resampling drift)
fs17 = 1.0/(t17[1] - t17[0])
_, V1_q = ADC17(n_bits=10, fs=fs17).convert(t17, V1)
_, V2_q = ADC17(n_bits=10, fs=fs17).convert(t17, V2)
N_final = min(len(V1_q), len(V2_q))
I1_meas = np.maximum(V1_q[:N_final], 0) / (Rf17*Rlam17*P_peak)
I2_meas = np.maximum(V2_q[:N_final], 0) / (Rf17*Rlam17*P_peak)

phi_rec17, errs17, _ = gs_core.retrieve_phase_with_history(I1_meas, I2_meas, D1_cap, D2_cap,
                                                             n_iter=80, unit_amplitude=unit_amp17)
rms_err17, phi_aligned17 = compare_phase(phi_rec17, phi17[:N_final], np.ones(N_final))
print(f"pipeline: QPSK object -> dispersive transform -> photodetector -> 10-bit ADC -> GS inverse")
print(f"RMS phase-recovery error (offset/twin ambiguity removed): {rms_err17:.4f} rad")"""))

cells.append(co("""fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(t17[:N_final], phi17[:N_final], label='true phase')
axs[0].plot(t17[:N_final], phi_aligned17, '--', label='GS-recovered (aligned)')
axs[0].set_xlabel('t'); axs[0].set_ylabel('phase (rad)'); axs[0].legend()
axs[0].set_title('capstone: recovered vs true phase, through a realistic detector chain')
axs[1].plot(errs17); axs[1].set_yscale('log')
axs[1].set_xlabel('GS iteration'); axs[1].set_ylabel('RMS amplitude error'); axs[1].set_title('GS convergence')
plt.tight_layout(); plt.show()

summary17 = pd.DataFrame([
    {"stage": "hidden object",          "quantity": "QPSK complex field E(t)=e^{i phi(t)}", "value": f"N={len(t17)} samples"},
    {"stage": "dispersive transform",   "quantity": "D1, D2 (normalized)",                  "value": f"{D1_cap}, {D2_cap}"},
    {"stage": "photodetector",          "quantity": "responsivity, R_f",                     "value": f"{Rlam17:.3f} A/W, {Rf17/1e3:.0f} kOhm"},
    {"stage": "ADC",                    "quantity": "bits, samples",                         "value": f"10-bit, N={N_final}"},
    {"stage": "inverse reconstruction", "quantity": "RMS phase error",                        "value": f"{rms_err17:.4f} rad"},
])
display(summary17)"""))

cells.append(md("""## Summary -- the repo *is* this pipeline

| Section of this notebook | This repo's code |
|---|---|
| vector calculus, electrostatics, Gauss, Poisson | `griffiths.vectors/electrostatics`, `dgs.gauss_law`, `dgs.poisson_2d` |
| fields in matter | `griffiths.dielectrics` |
| Maxwell's equations, wave equation, plane-wave dispersion | `griffiths.electrodynamics` |
| Fourier optics, dispersion Taylor series | `dgs.fourier_optics`/`diffraction`, `dgs.taylor`, `dgs.em_dispersion` |
| GVD $\\beta_2$ over length $L$ | the parameter $D=2\\pi\\beta_2 L$ |
| quadratic spectral phase | $H(f)=e^{i\\pi D f^2}$ |
| photodetector, TIA noise budget | `dgs.transimpedance_amplifier` |
| sampling, quantization | `dgs.adc`, `dgs.adc_snr_bits` |
| Tikhonov inverse problems | `dgs.imaging_inverse` |
| differentiable/autograd phase retrieval | `dgs.differentiable_optics_tutorial` |
| classical Gerchberg-Saxton phase retrieval | `dgs.gs_core` |

Gerchberg-Saxton (or its autograd twin, Part 13) then **inverts** this whole chain to
recover the phase $\\phi(t)$ a square-law detector threw away. Civilian optical metrology /
education."""))

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3.12 (torch)", "language": "python", "name": "py312"}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "electrodynamics_to_dispersion_operator.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
