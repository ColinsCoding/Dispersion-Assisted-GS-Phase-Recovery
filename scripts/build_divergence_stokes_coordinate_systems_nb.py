"""Build notebooks/divergence_stokes_coordinate_systems.ipynb -- completes the
FTC-as-vector-calculus-theorems sweep started in ftc_coordinate_systems.ipynb
(which did the gradient theorem in 4 systems). This notebook does the other
two: the divergence theorem and Stokes' theorem, each in Cartesian,
cylindrical, spherical, and elliptic coordinates -- 8 checks total.

Method: each side of each theorem is built by COMPOSING sympy pieces --
coordinate transform -> field -> curvilinear_div/curl -> volume/surface
element -- into one integrand via sp.lambdify, then integrated numerically
(scipy.integrate). Numeric quadrature is used uniformly (not symbolic
integration) after the gradient-theorem notebook's elliptic case showed
sympy silently choosing a wrong branch on a mixed hyperbolic/trig integral --
scipy.dblquad/tplquad has no such failure mode and keeps runtime predictable
across all 8 checks.

Sections:
  S1  Setup + the lambdify-and-compose methodology
  S2  Divergence theorem, 4 systems (shell regions)
  S3  Stokes' theorem, 4 systems (cap surfaces)
  S4  Summary table
"""

import json, pathlib

NB = pathlib.Path("notebooks/divergence_stokes_coordinate_systems.ipynb")
NB.parent.mkdir(exist_ok=True)

cells = []
def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "execution_count": None,
                              "metadata": {}, "outputs": [], "source": src})


# ── S1 ────────────────────────────────────────────────────────────────────────
md("""# Divergence and Stokes' Theorems Across Four Coordinate Systems

[`ftc_coordinate_systems.ipynb`](ftc_coordinate_systems.ipynb) verified the
**gradient theorem** (FTC for line integrals) in Cartesian, cylindrical,
spherical, and elliptic coordinates. This notebook completes the set:

$$\\textbf{Divergence theorem: }\\int_V(\\nabla\\cdot\\mathbf F)\\,dV=\\oint_S\\mathbf F\\cdot d\\mathbf a
\\qquad\\qquad
\\textbf{Stokes' theorem: }\\int_S(\\nabla\\times\\mathbf F)\\cdot d\\mathbf a=\\oint_{\\partial S}\\mathbf F\\cdot d\\boldsymbol\\ell$$

**Method.** Every check below is built the same way: **compose** the pieces
already in `griffiths/curvilinear.py` -- a field in local coordinates, its
`curvilinear_div`/`curvilinear_curl`, and the matching `volume_element`/
`surface_element` -- into a single sympy expression, `sp.lambdify` it to a
plain numeric function, and integrate with `scipy.integrate`
(dblquad/tplquad). Numeric quadrature is used everywhere, not symbolic
integration: the gradient-theorem notebook's elliptic case showed
`sp.integrate` silently pick a wrong branch on a mixed hyperbolic/trig
integrand, caught only by an independent numeric check. Composing the same
small set of pieces (transform, field, operator, element -> lambdify ->
quadrature) into all 8 checks below, instead of writing bespoke numeric code
per system, is also the cleanest way to keep 8 near-identical derivations
from drifting into 8 different sets of bugs.
""")

code("""\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

import numpy as np
import sympy as sp
from scipy import integrate
import pandas as pd

from griffiths.curvilinear import (
    r, theta, phi, s, phi_cyl, z_cyl, c_focal, u_ell, v_ell,
    spherical_scale_factors, cylindrical_scale_factors, elliptic_scale_factors,
    curvilinear_div, curvilinear_curl, volume_element, surface_element,
)

results = []
print("sympy", sp.__version__, "| scipy integrate ready")
""")

# ── S2: divergence theorem ────────────────────────────────────────────────────
md("""## Divergence Theorem

Each system uses a **shell** region -- bounded away from any coordinate
degeneracy (the axis, the origin, the foci) -- with a purely radial-like
field $F_1=f(q_1)$, $F_2=F_3=0$. That choice means only the $q_1=$const
faces carry net flux: the periodic angular coordinate's two "ends" are the
same physical surface with opposite orientation and cancel exactly, and
there's no $q_3$ (or $\\theta$) component to contribute anything on those
faces either.""")

md("### Cartesian: box $[0,a]\\times[0,b]\\times[0,c]$, $\\mathbf F=(x^2,y^2,z^2)$"),

code("""\
x, y, z = sp.symbols('x y z', real=True)
a_c, b_c, c_c = 1.0, 2.0, 1.5
Fx, Fy, Fz = x**2, y**2, z**2

div_cart = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)
div_cart_f = sp.lambdify((x, y, z), div_cart, "numpy")
LHS, _ = integrate.tplquad(lambda zz, yy, xx: div_cart_f(xx, yy, zz), 0, a_c, 0, b_c, 0, c_c)

Fx_f, Fy_f, Fz_f = (sp.lambdify((x, y, z), Fi, "numpy") for Fi in (Fx, Fy, Fz))
RHS = (integrate.dblquad(lambda zz, yy: Fx_f(a_c, yy, zz), 0, b_c, 0, c_c)[0]
       + integrate.dblquad(lambda zz, xx: Fy_f(xx, b_c, zz), 0, a_c, 0, c_c)[0]
       + integrate.dblquad(lambda yy, xx: Fz_f(xx, yy, c_c), 0, a_c, 0, b_c)[0])

print(f"LHS (volume integral of div F) = {LHS:.6f}")
print(f"RHS (flux through 6 faces)     = {RHS:.6f}")
results.append({"theorem": "divergence", "system": "Cartesian", "LHS": LHS, "RHS": RHS})
""")

md("### Cylindrical: shell $s\\in[s_1,s_2]$, $\\mathbf F=(s^2,0,0)$ (fiber cross-section)"),

code("""\
h_cyl = cylindrical_scale_factors()
F_cyl = [s**2, 0, 0]
div_cyl = curvilinear_div(F_cyl, (s, phi_cyl, z_cyl), h_cyl)
div_cyl_f = sp.lambdify((s, phi_cyl, z_cyl), sp.simplify(div_cyl * volume_element(h_cyl)), "numpy")

s1, s2, hgt = 0.5, 1.5, 1.0
LHS, _ = integrate.tplquad(lambda zz, pp, ss: div_cyl_f(ss, pp, zz), s1, s2, 0, 2*np.pi, 0, hgt)

Fs_f = sp.lambdify(s, F_cyl[0], "numpy")
dA_s_f = sp.lambdify(s, surface_element(h_cyl, 0), "numpy")
flux_out = Fs_f(s2) * dA_s_f(s2) * 2*np.pi * hgt
flux_in = Fs_f(s1) * dA_s_f(s1) * 2*np.pi * hgt
RHS = flux_out - flux_in

print(f"LHS = {LHS:.6f}   RHS (outer - inner flux) = {RHS:.6f}")
results.append({"theorem": "divergence", "system": "Cylindrical", "LHS": LHS, "RHS": RHS})
""")

md("### Spherical: shell $r\\in[r_1,r_2]$, $\\mathbf F=(r^2,0,0)$ (Mie-scattering-style radial field)"),

code("""\
h_sph = spherical_scale_factors()
F_sph = [r**2, 0, 0]
div_sph = curvilinear_div(F_sph, (r, theta, phi), h_sph)
div_sph_f = sp.lambdify((r, theta, phi), sp.simplify(div_sph * volume_element(h_sph)), "numpy")

r1, r2 = 0.5, 1.5
LHS, _ = integrate.tplquad(lambda pp, th, rr: div_sph_f(rr, th, pp), r1, r2, 0, np.pi, 0, 2*np.pi)

Fr_f = sp.lambdify(r, F_sph[0], "numpy")
dA_r_f = sp.lambdify((r, theta), surface_element(h_sph, 0), "numpy")
flux_out, _ = integrate.dblquad(lambda th, pp: Fr_f(r2)*dA_r_f(r2, th), 0, 2*np.pi, 0, np.pi)
flux_in, _ = integrate.dblquad(lambda th, pp: Fr_f(r1)*dA_r_f(r1, th), 0, 2*np.pi, 0, np.pi)
RHS = flux_out - flux_in

print(f"LHS = {LHS:.6f}   RHS = {RHS:.6f}")
results.append({"theorem": "divergence", "system": "Spherical", "LHS": LHS, "RHS": RHS})
""")

md("### Elliptic: shell $u\\in[u_1,u_2]$, $\\mathbf F=(u^2,0,0)$ (elliptical-core-fiber cross-section)"),

code("""\
h_ell = elliptic_scale_factors(c=sp.Integer(1))
F_ell = [u_ell**2, 0, 0]
div_ell = curvilinear_div(F_ell, (u_ell, v_ell, z_cyl), h_ell)
div_ell_f = sp.lambdify((u_ell, v_ell, z_cyl), sp.simplify(div_ell * volume_element(h_ell)), "numpy")

u1, u2, hgt2 = 0.4, 1.2, 1.0
LHS, _ = integrate.tplquad(lambda zz, vv, uu: div_ell_f(uu, vv, zz), u1, u2, 0, 2*np.pi, 0, hgt2)

Fu_f = sp.lambdify(u_ell, F_ell[0], "numpy")
dA_u_f = sp.lambdify((u_ell, v_ell), surface_element(h_ell, 0), "numpy")
flux_out, _ = integrate.dblquad(lambda vv, zz: Fu_f(u2)*dA_u_f(u2, vv), 0, hgt2, 0, 2*np.pi)
flux_in, _ = integrate.dblquad(lambda vv, zz: Fu_f(u1)*dA_u_f(u1, vv), 0, hgt2, 0, 2*np.pi)
RHS = flux_out - flux_in

print(f"LHS = {LHS:.6f}   RHS = {RHS:.6f}")
results.append({"theorem": "divergence", "system": "Elliptic", "LHS": LHS, "RHS": RHS})
""")

# ── S3: Stokes ────────────────────────────────────────────────────────────────
md("""## Stokes' Theorem

Each system uses a **cap** surface: the "radial-like" coordinate ($x/y$
plane's implicit radius, $s$, $\\theta$, or $u$) runs from its natural
minimum out to a finite cutoff, the periodic angle runs over its full range,
and the boundary is the single curve at that cutoff -- a rectangle
(Cartesian), a disk's rim (cylindrical), a latitude circle (spherical), or an
ellipse (elliptic).""")

md("### Cartesian: rectangle $[0,a]\\times[0,b]$ at $z=$const, $\\mathbf F=(-y,x,0)$"),

code("""\
Fx2, Fy2 = -y, x
curl_z_cart = sp.diff(Fy2, x) - sp.diff(Fx2, y)
LHS = float(curl_z_cart) * a_c * b_c

Fx2_f = sp.lambdify((x, y), Fx2, "numpy")
Fy2_f = sp.lambdify((x, y), Fy2, "numpy")
RHS = (integrate.quad(lambda xx: Fx2_f(xx, 0), 0, a_c)[0]
       + integrate.quad(lambda yy: Fy2_f(a_c, yy), 0, b_c)[0]
       + integrate.quad(lambda xx: Fx2_f(xx, b_c), a_c, 0)[0]
       + integrate.quad(lambda yy: Fy2_f(0, yy), b_c, 0)[0])

print(f"LHS (surface integral of curl) = {LHS:.6f}")
print(f"RHS (loop integral)            = {RHS:.6f}")
results.append({"theorem": "Stokes", "system": "Cartesian", "LHS": LHS, "RHS": RHS})
""")

md("### Cylindrical: disk $s\\in[0,R]$ at $z=$const, $\\mathbf F=s\\,\\hat\\phi$ (solid-body rotation)"),

code("""\
F_cyl2 = [0, s, 0]
curl_z_cyl = sp.simplify(curvilinear_curl(F_cyl2, (s, phi_cyl, z_cyl), h_cyl)[2])
integrand_cyl = sp.lambdify((s, phi_cyl), sp.simplify(curl_z_cyl * surface_element(h_cyl, 2)), "numpy")

R_cyl = 1.0
LHS, _ = integrate.dblquad(lambda pp, ss: integrand_cyl(ss, pp), 0, R_cyl, 0, 2*np.pi)

Fphi_f = sp.lambdify(s, F_cyl2[1], "numpy")
h_phi_f = sp.lambdify(s, h_cyl[1], "numpy")
RHS, _ = integrate.quad(lambda pp: Fphi_f(R_cyl)*h_phi_f(R_cyl), 0, 2*np.pi)

print(f"LHS = {LHS:.6f}   RHS = {RHS:.6f}")
results.append({"theorem": "Stokes", "system": "Cylindrical", "LHS": LHS, "RHS": RHS})
""")

md("### Spherical: polar cap $r=R$, $\\theta\\in[0,\\theta_0]$, $\\mathbf F=r\\sin\\theta\\,\\hat\\phi$"),

code("""\
F_sph2 = [0, 0, r*sp.sin(theta)]
curl_r_sph = sp.simplify(curvilinear_curl(F_sph2, (r, theta, phi), h_sph)[0])
integrand_sph = sp.lambdify((r, theta), sp.simplify(curl_r_sph * surface_element(h_sph, 0)), "numpy")

R_sph, theta0 = 1.0, np.pi/3
LHS, _ = integrate.dblquad(lambda th, pp: integrand_sph(R_sph, th), 0, 2*np.pi, 0, theta0)

Fphi_sph_f = sp.lambdify((r, theta), F_sph2[2], "numpy")
h_phi_sph_f = sp.lambdify((r, theta), h_sph[2], "numpy")
RHS, _ = integrate.quad(lambda pp: Fphi_sph_f(R_sph, theta0)*h_phi_sph_f(R_sph, theta0), 0, 2*np.pi)

print(f"LHS = {LHS:.6f}   RHS = {RHS:.6f}")
results.append({"theorem": "Stokes", "system": "Spherical", "LHS": LHS, "RHS": RHS})
""")

md("### Elliptic: cap $u\\in[0,u_0]$ at $z=$const, $\\mathbf F=\\sinh(u)\\,\\hat v$"),

code("""\
F_ell2 = [0, sp.sinh(u_ell), 0]
curl_z_ell = sp.simplify(curvilinear_curl(F_ell2, (u_ell, v_ell, z_cyl), h_ell)[2])
integrand_ell = sp.lambdify((u_ell, v_ell), sp.simplify(curl_z_ell * surface_element(h_ell, 2)), "numpy")

u0 = 0.8
LHS, _ = integrate.dblquad(lambda vv, uu: integrand_ell(uu, vv), 0, u0, 0, 2*np.pi)

Fv_ell_f = sp.lambdify(u_ell, F_ell2[1], "numpy")
h_v_ell_f = sp.lambdify((u_ell, v_ell), h_ell[1], "numpy")
RHS, _ = integrate.quad(lambda vv: Fv_ell_f(u0)*h_v_ell_f(u0, vv), 0, 2*np.pi)

print(f"LHS = {LHS:.6f}   RHS = {RHS:.6f}")
results.append({"theorem": "Stokes", "system": "Elliptic", "LHS": LHS, "RHS": RHS})
""")

# ── S4: summary ────────────────────────────────────────────────────────────────
md("## Summary: 8 Checks, 4 Systems x 2 Theorems"),

code("""\
df = pd.DataFrame(results)
df["abs_err"] = (df["LHS"] - df["RHS"]).abs()
df["rel_err"] = df["abs_err"] / df["LHS"].abs()
df
""")

code("""\
assert (df["rel_err"] < 1e-4).all(), "every LHS/RHS pair must agree to numeric-quadrature precision"
print("All 8 checks pass: divergence theorem and Stokes' theorem hold in all four coordinate systems.")
print()
print("Combined with ftc_coordinate_systems.ipynb's gradient-theorem sweep, all three")
print("FTC generalizations are now verified in Cartesian, cylindrical, spherical, and")
print("elliptic coordinates -- 12 checks total across the two notebooks.")
""")

# ── finalize ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB}  ({len(cells)} cells)")
print(f"Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace \"{NB}\"")
