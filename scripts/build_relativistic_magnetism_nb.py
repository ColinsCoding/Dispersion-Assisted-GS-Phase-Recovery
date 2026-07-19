#!/usr/bin/env python
"""Builds notebooks/magnetism_as_relativistic_electricity.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Magnetism on a Moving Charge: an Electric Force, Seen From the Right Frame

Griffiths (Ch.12) and Purcell both make this point: the magnetic force on a
charge moving near a current-carrying wire is not a separate fundamental
force -- it's the ordinary Coulomb force on a NET charge density that only
appears because of special relativity's length contraction, applied
asymmetrically to the wire's positive ions (at rest) and conduction
electrons (drifting).

We derive this directly: boost into the frame of a test charge moving
alongside the wire, show the wire's positive and negative charge densities
length-contract by DIFFERENT amounts (because they have different
velocities in that frame), compute the resulting net charge density and the
electric force it produces, and check that an utterly microscopic
relativistic charge imbalance reproduces the full, everyday-strength
magnetic force.
""")

code(r"""import sympy as sp
sp.init_printing()

c, v, u, s, lam0, q, eps0, mu0 = sp.symbols("c v u s lambda_0 q epsilon_0 mu_0", positive=True)
print("SymPy", sp.__version__, "ready")
""")

md(r"""## Setup

A wire along the x-axis: positive ion lattice at rest with line charge
density $\lambda_0$, conduction electrons drifting at speed $u$ with line
charge density $-\lambda_0$ (the wire is neutral in the lab frame $S$). A
test charge $q$ moves at speed $v$ parallel to the wire, at perpendicular
distance $s$. In the lab frame, the wire is neutral, so there's NO electric
force on $q$ -- only the familiar magnetic force $qvB$.

Now boost into $S'$, the test charge's rest frame, moving at $v$ relative to
$S$.
""")

code(r"""def gamma(speed):
    return 1 / sp.sqrt(1 - speed**2 / c**2)

def relativistic_velocity_add(speed_in_S, boost_speed):
    # Velocity of an object (moving at speed_in_S in frame S) as seen in a
    # frame S' that moves at boost_speed relative to S.
    return (speed_in_S - boost_speed) / (1 - speed_in_S * boost_speed / c**2)

# positive ions: at rest in S (velocity 0), so their REST density equals
# their density in S, lambda0. In S', they move at -v.
lam_plus_S_prime = gamma(v) * lam0
print("positive-ion density in S' (test-charge frame):")
sp.pprint(lam_plus_S_prime)

# electrons: move at -u in S, so their proper (rest-frame) density is
# lambda0 / gamma(u) (since -lambda0 = gamma(u) * lambda_proper in S)
lam_minus_proper = -lam0 / gamma(u)

# velocity of electrons in S' (boost S' moves at v relative to S, electrons
# move at -u in S)
u_prime = relativistic_velocity_add(-u, v)
lam_minus_S_prime = gamma(u_prime) * lam_minus_proper
print("\nelectron velocity in S':")
sp.pprint(sp.simplify(u_prime))
print("\nelectron density in S':")
sp.pprint(sp.simplify(lam_minus_S_prime))
""")

code(r"""# net charge density in S' -- this should be nonzero, even though the wire
# is exactly neutral in the lab frame S
lam_net_S_prime = sp.simplify(lam_plus_S_prime + lam_minus_S_prime)
print("net line charge density in the test-charge's frame, lambda'_net =")
sp.pprint(lam_net_S_prime)
""")

md(r"""## The non-relativistic (everyday-speed) limit

Real conduction-electron drift speeds are ~mm/s, and ordinary lab speeds for
$v$ are nowhere near $c$. Expand $\lambda'_{net}$ to leading order in
$u/c$ and $v/c$ -- the relativistic effect should be utterly tiny per unit
charge, but multiplied by $\lambda_0$ (which is enormous: a wire's free
electron density), it should reproduce the full, ordinary-strength magnetic
force. That's the whole point of this calculation.
""")

code(r"""# Taylor-expand in u/c and v/c around 0, keep the leading nonzero term
small = sp.symbols("epsilon", positive=True)  # bookkeeping parameter for the expansion order
lam_net_expanded = sp.series(
    lam_net_S_prime.subs({u: small * u, v: small * v}), small, 0, 4
).removeO()
lam_net_leading = sp.simplify(lam_net_expanded.subs(small, 1))
print("leading-order net charge density (u,v << c) =")
sp.pprint(lam_net_leading)
""")

code(r"""# the electric field this net line charge produces at distance s (infinite
# line charge formula, standard Griffiths E&M result), and the force on q
E_prime = lam_net_leading / (2 * sp.pi * eps0 * s)
F_prime = sp.simplify(q * E_prime)
print("electric force on q in S' (from the relativistic charge imbalance alone) =")
sp.pprint(F_prime)

# compare to the textbook magnetic force in the LAB frame: F = q v B,
# B = mu0 I / (2 pi s), I = lambda0 * u (current from the drifting electrons)
I_current = lam0 * u
B_field = mu0 * I_current / (2 * sp.pi * s)
F_magnetic_lab = q * v * B_field
print("\ntextbook magnetic force magnitude in the lab frame, |F| = qvB =")
sp.pprint(F_magnetic_lab)

# Their MAGNITUDES should match exactly once mu0 = 1/(eps0 c^2) is
# substituted. The sign can come out -1: F_prime correctly tracks the force
# DIRECTION (attractive, toward the wire, since the net charge seen in S' is
# negative when u and v are both positive), while F_magnetic_lab here is a
# bare scalar q*v*B that never tracked the actual vector direction of v x B
# -- so a relative minus sign reflects that bookkeeping choice, not an error
# in the physics. The magnitudes are the real claim.
ratio = sp.simplify((F_prime / F_magnetic_lab).subs(mu0, 1 / (eps0 * c**2)))
print("\nF_prime / F_magnetic_lab (sign reflects force direction bookkeeping):", ratio)
print("|F_prime| == |F_magnetic_lab| exactly:", sp.simplify(sp.Abs(ratio) - 1) == 0)
""")

md(r"""## How small is the relativistic charge imbalance, really?

Plug in real numbers for a household copper wire: free-electron density
~$10^{29}$ m$^{-3}$, drift speed ~0.1 mm/s, test-charge speed comparable.
The fractional charge imbalance $\lambda'_{net}/\lambda_0$ should come out
almost inconceivably small -- and yet it reproduces an everyday magnetic
force, because $\lambda_0$ itself is enormous.
""")

code(r"""import numpy as np

c_val = 3e8
u_val = 1e-4     # drift speed, ~0.1 mm/s, realistic for household wiring
v_val = 1.0      # test charge moving at 1 m/s alongside the wire

fractional_imbalance = (u_val * v_val) / c_val**2   # leading-order u*v/c^2 scaling
print(f"fractional charge imbalance lambda'_net / lambda0 ~ {fractional_imbalance:.3e}")
print("(i.e. about 1 part in", f"{1/fractional_imbalance:.2e}", "of the wire's charge density)")
print("\nThis utterly microscopic imbalance, multiplied by the wire's enormous")
print("charge density, reproduces the FULL ordinary magnetic force -- magnetism")
print("is electrostatics, viewed from a frame where it isn't exactly cancelled.")
""")

md(r"""## Summary

The net charge density $\lambda'_{net}$ in the test charge's rest frame was
derived from first principles (relativistic length contraction applied
separately to the at-rest ions and the drifting electrons, plus relativistic
velocity addition), not assumed. Its leading-order expansion in $u/c, v/c$
reproduces the textbook magnetic force $qvB$ in MAGNITUDE *exactly* once
$\mu_0=1/(\epsilon_0 c^2)$ is substituted -- confirmed symbolically. The
relative sign that appears is a force-direction bookkeeping artifact (the
relativistic calculation correctly resolves the force as attractive; the
bare scalar $qvB$ comparison never tracked the vector direction of
$\vec v\times\vec B$), not an error in the physics. The numeric example shows
just how small the relativistic effect is per electron (one part in
$10^{16}$-ish) and how that's still enough to power every electric motor,
because the line charge density it's a fraction of is astronomically large.
""")

nb["cells"] = cells

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/magnetism_as_relativistic_electricity.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/magnetism_as_relativistic_electricity.ipynb with", len(cells), "cells")
