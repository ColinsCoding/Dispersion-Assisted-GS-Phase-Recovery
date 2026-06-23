# Griffiths Problem 1.16 — a fully worked solution (every step)

**Problem.** Sketch the vector function `v = r̂ / r²` and compute its divergence. The
answer may surprise you — explain it.

---

## Step 0 — What the field looks like (the sketch)

`r̂` is the unit vector pointing radially *outward* from the origin, and `1/r²` is its
length. So at every point `v` points straight away from the origin, getting **weaker**
as you go out (length `1/r²`). The sketch: arrows radiating outward in all directions,
long near the center, short far away. It *looks* like it is spreading out everywhere —
hold that thought.

## Step 1 — Write v in Cartesian coordinates

To take a divergence with `∂/∂x + ∂/∂y + ∂/∂z`, put everything in `x, y, z`. Two facts:

- the radial unit vector is `r̂ = r⃗/r = (x x̂ + y ŷ + z ẑ)/r`,
- the radius is `r = (x² + y² + z²)^(1/2)`.

Therefore
```
v = r̂ / r² = r⃗ / r³ = (x/r³) x̂ + (y/r³) ŷ + (z/r³) ẑ ,
```
where `r³ = (x² + y² + z²)^(3/2)`. The three components are
```
v_x = x (x²+y²+z²)^(-3/2) ,   v_y = y (...)^(-3/2) ,   v_z = z (...)^(-3/2) .
```

## Step 2 — Differentiate the x-component (full product rule)

```
∂v_x/∂x = ∂/∂x [ x · (x²+y²+z²)^(-3/2) ] .
```
This is a product `x · f`, so use `(uf)' = u'f + uf'`:

**(a)** the `u' f` piece — derivative of the `x` out front:
```
(∂x/∂x) · (x²+y²+z²)^(-3/2) = 1 · (x²+y²+z²)^(-3/2) = 1/r³ .
```

**(b)** the `u f'` piece — `x` times the derivative of the power. Use the chain rule on
`(x²+y²+z²)^(-3/2)`: bring down the `-3/2`, lower the power by one, times the inner
derivative `∂/∂x (x²+y²+z²) = 2x`:
```
x · [ (-3/2)(x²+y²+z²)^(-5/2) · (2x) ]
  = x · ( -3x · (x²+y²+z²)^(-5/2) )
  = -3x² (x²+y²+z²)^(-5/2)
  = -3x² / r⁵ .
```

Add (a) + (b):
```
∂v_x/∂x = 1/r³ − 3x²/r⁵ .
```

## Step 3 — The other two components by symmetry

Nothing distinguished `x`; the same computation gives
```
∂v_y/∂y = 1/r³ − 3y²/r⁵ ,
∂v_z/∂z = 1/r³ − 3z²/r⁵ .
```

## Step 4 — Add them up and simplify

```
∇·v = (1/r³ − 3x²/r⁵) + (1/r³ − 3y²/r⁵) + (1/r³ − 3z²/r⁵)
    = 3/r³ − 3(x² + y² + z²)/r⁵ .
```
Now use `x² + y² + z² = r²`:
```
∇·v = 3/r³ − 3 r²/r⁵ = 3/r³ − 3/r³ = 0 .
```

**Result so far:** `∇·v = 0` — but read the fine print. Every term had `r` in a
denominator, so this is valid **only where r ≠ 0**. At the origin `r = 0` makes
`1/r³` and `1/r⁵` blow up and the calculation is meaningless (it is `∞ − ∞`).

## Step 5 — The surprise, stated plainly

The picture (Step 0) screams "this field diverges everywhere," yet the algebra says the
divergence is **zero** everywhere (off the origin). Both cannot be the full story. The
resolution comes from the *integral* form, not the local derivative.

## Step 6 — Test it with the divergence theorem (compute the flux)

The divergence theorem says `∫_V (∇·v) dτ = ∮_S v · da`. Pick the surface `S` = a sphere
of radius `R` centered on the origin (so the origin is *inside*). On that sphere:

- the field is `v = r̂ / R²` (radius is `R` there),
- the area element is `da = R² sinθ dθ dφ r̂` (outward, radial).

Their dot product (`r̂ · r̂ = 1`):
```
v · da = (r̂/R²) · (R² sinθ dθ dφ r̂) = (1/R²)(R²) sinθ dθ dφ = sinθ dθ dφ .
```
The `R²` cancels. Integrate over the whole sphere:
```
∮_S v · da = ∫₀^{2π} ∫₀^{π} sinθ dθ dφ
           = ( ∫₀^{2π} dφ ) ( ∫₀^{π} sinθ dθ )
           = ( 2π ) ( [−cosθ]₀^{π} ) = (2π)(1 − (−1)) = (2π)(2) = 4π .
```

**The flux through any sphere is `4π`, independent of the radius R.**

## Step 7 — Spot the contradiction (and resolve it)

The divergence theorem links the two sides:
```
∫_V (∇·v) dτ = ∮_S v · da = 4π .
```
But in Step 4 we found `∇·v = 0` everywhere. If that were the *whole* truth, the volume
integral on the left would be `∫ 0 dτ = 0`, not `4π`. **Contradiction.**

The only way out: `∇·v` is **not** actually zero everywhere. It is zero for `r ≠ 0`, but
it must carry a contribution at the **one point we excluded**, `r = 0`, and that
contribution must integrate to exactly `4π`.

## Step 8 — Name the object: the Dirac delta

What is zero everywhere except a single point, yet has a finite (nonzero) integral? The
three-dimensional **Dirac delta** `δ³(r⃗)`, defined by
```
δ³(r⃗) = 0 for r⃗ ≠ 0 ,    and    ∫ δ³(r⃗) dτ = 1 (over any volume containing the origin).
```
To make our volume integral come out to `4π`, the divergence must be `4π` times it:
```
┌───────────────────────────────────────┐
│   ∇ · ( r̂ / r² ) = 4π δ³(r⃗)          │
└───────────────────────────────────────┘
```
**Check:** `∫_V 4π δ³(r⃗) dτ = 4π · 1 = 4π`, matching the flux from Step 6. ✓  And for
`r ≠ 0` it is `4π · 0 = 0`, matching Step 4. ✓  Both halves are now consistent.

## Step 9 — Why it matters (the physics)

This is not a curiosity — it is **Gauss's law in disguise.** The electric field of a
point charge `q` at the origin is
```
E = (q / 4πε₀) · (r̂ / r²) .
```
Taking the divergence and using our boxed result:
```
∇·E = (q / 4πε₀) · 4π δ³(r⃗) = (q/ε₀) δ³(r⃗) = ρ/ε₀ ,
```
because the charge density of a point charge is `ρ = q δ³(r⃗)` (all the charge piled at
one point). So `∇·E = ρ/ε₀` works perfectly — **the divergence of E is concentrated
exactly where the charge sits.** Divergence really does measure the *source* of field
lines; for `r̂/r²` the source is the single point at the origin.

**Bonus identity.** Since `∇(1/r) = −r̂/r²`, our field is `v = −∇(1/r)`, so
`∇·v = −∇²(1/r)`. Comparing to the boxed result:
```
∇²(1/r) = −4π δ³(r⃗) ,
```
the **Green's function of the Laplacian** — the workhorse you will use all through
Chapters 2–3 to build potentials from charge distributions.

---

## One-line summary for the board

> `∇·(r̂/r²) = 0` in empty space, but the flux through **every** sphere is `4π` — so all
> the divergence is a `4π` Dirac-delta spike at the origin: `∇·(r̂/r²) = 4π δ³(r⃗)`. That
> is Gauss's law for a point charge.

*Verified numerically in `griffiths/inverse_square.py` (divergence = 0 off-origin; flux
= 4π through spheres of radius 0.5, 1, 5); test in `tests/test_inverse_square.py`.*
