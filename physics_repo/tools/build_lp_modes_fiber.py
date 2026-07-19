"""Generate notebooks/lp_modes_characteristic_equation.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# LP modes of a step-index fiber: the characteristic equation with modified Bessel $K$

The previous notebook used the Bessel zeros to get a fiber's single-mode limit. The full story needs two
Bessel families, because a guided mode is **oscillatory in the core** and **evanescent in the cladding**:

- Core ($r<a$, higher index $n_1$): the field is $J_\ell(u\,r/a)$, with $u=a\sqrt{n_1^2k_0^2-\beta^2}$
  (real, oscillatory) -- an ordinary Bessel function.
- Cladding ($r>a$, index $n_2$): the field must **decay**, so it is the **modified Bessel function**
  $K_\ell(w\,r/a)$, with $w=a\sqrt{\beta^2-n_2^2k_0^2}$ -- $K_\ell$ falls off like $e^{-w r/a}$.

Matching the field and its slope at $r=a$ (weak-guidance / LP approximation) gives the **characteristic
equation**
$$u\,\frac{J_{\ell+1}(u)}{J_\ell(u)}=w\,\frac{K_{\ell+1}(w)}{K_\ell(w)},\qquad u^2+w^2=V^2,$$
with the normalized frequency $V=a k_0\,\mathrm{NA}$. Its roots are the guided modes $\mathrm{LP}_{\ell m}$;
their cutoffs are (again) Bessel zeros, and $\mathrm{LP}_{11}$ cutting on at $V=2.405$ recovers the
single-mode limit. We verify $K_\ell$ solves the modified Bessel equation (SymPy), solve the characteristic
equation for the modes (SciPy), check the cutoffs, and plot the core-plus-cladding field profiles.
Self-contained: NumPy, SymPy, SciPy, Matplotlib."""),
setup_cell(),
co("""from scipy.special import jv, kv, jn_zeros
from scipy.optimize import brentq"""),

md(r"""## The modified Bessel function $K_\ell$: the evanescent cladding field

$K_\ell$ solves the **modified** Bessel equation $x^2y''+xy'-(x^2+\ell^2)y=0$ (note the $+x^2$: solutions
grow/decay rather than oscillate) and $K_\ell(x)\to\sqrt{\pi/2x}\,e^{-x}$ -- exponential decay, exactly what
a bound cladding field needs. SymPy confirms it solves the equation; the plot contrasts the oscillating
$J_\ell$ with the decaying $K_\ell$."""),
co("""x = sp.symbols('x', positive=True)
for order in (0, 1, 2):
    y = sp.besselk(order, x)
    residual = sp.simplify(x**2*sp.diff(y, x, 2) + x*sp.diff(y, x) - (x**2 + order**2)*y)
    assert residual == 0
    print(f"K_{order} solves the modified Bessel equation (residual {residual})")
print("K_0(1,3,5) =", np.round(kv(0, [1., 3., 5.]), 5), " -> exponential decay (evanescent)")"""),

md(r"""## Solving the characteristic equation for the LP modes

Write the equation as $F(u)=u\,J_{\ell+1}(u)\,K_\ell(w)-w\,K_{\ell+1}(w)\,J_\ell(u)=0$ (smooth, no poles),
scan $u\in(0,V)$ for sign changes, and bracket each root. Every root is a guided mode $\mathrm{LP}_{\ell m}$
with normalized propagation constant $b=1-(u/V)^2\in(0,1)$ (near 1 = tightly bound). Here $V=5$."""),
co("""def lp_modes(V, l_max=5):
    modes = []
    for l in range(l_max + 1):
        def F(u):
            w = np.sqrt(max(V**2 - u**2, 1e-12))
            return u*jv(l + 1, u)*kv(l, w) - w*kv(l + 1, w)*jv(l, u)
        us = np.linspace(1e-4, V - 1e-4, 6000)
        vals = np.array([F(u) for u in us])
        m = 0
        for i in range(len(us) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i]*vals[i + 1] < 0:
                u0 = brentq(F, us[i], us[i + 1], maxiter=200)
                m += 1
                modes.append({"l": l, "m": m, "u": round(u0, 4),
                              "w": round(float(np.sqrt(V**2 - u0**2)), 4), "b": round(1 - (u0/V)**2, 4)})
    return modes

V = 5.0
modes = lp_modes(V)
n1, n2 = 1.4457, 1.4378                                             # typical silica core/cladding
for row in modes:
    row["n_eff"] = round(float(np.sqrt(n2**2 + row["b"]*(n1**2 - n2**2))), 6)
print(f"V = {V}:  {len(modes)} guided LP modes")
print(pd.DataFrame(modes).to_string(index=False))
assert any(r["l"] == 0 and r["m"] == 1 for r in modes)             # LP01 always present
assert all(0 < r["b"] < 1 for r in modes)                          # bound modes"""),

md(r"""## Cutoffs are Bessel zeros; $\mathrm{LP}_{11}$ at $V=2.405$ is the single-mode limit

A mode cuts off when $w\to0$ (field no longer bound). For $\mathrm{LP}_{\ell m}$ with $\ell\ge1$ the cutoff
is the $m$-th zero of $J_{\ell-1}$; $\mathrm{LP}_{01}$ has no cutoff. So $\mathrm{LP}_{11}$ (first zero of
$J_0=2.405$) is the second mode to appear -- below it the fiber is single-mode. We confirm the mode count
jumps from 1 to 2 as $V$ crosses $2.405$."""),
co("""print("LP11 cutoff = first zero of J_0 =", round(jn_zeros(0, 1)[0], 4), " (single-mode below this)")
print("LP21/LP02 cutoff = first zero of J_1 =", round(jn_zeros(1, 1)[0], 4))
n_below = len(lp_modes(2.30, l_max=3))
n_above = len(lp_modes(2.50, l_max=3))
print(f"modes at V=2.30: {n_below}  (single-mode)   modes at V=2.50: {n_above}  (LP11 turns on)")
assert n_below == 1 and n_above == 2"""),

md(r"""## Mode field profiles: $J_\ell$ in the core, $K_\ell$ in the cladding

Each mode's radial field is $J_\ell(u r/a)$ inside and $[J_\ell(u)/K_\ell(w)]K_\ell(w r/a)$ outside --
continuous at $r=a$, oscillatory then exponentially decaying. This is the shape the light actually takes in
the fiber."""),
co("""def profile(l, u, w, r_over_a):
    core = jv(l, u*r_over_a)
    clad = jv(l, u)/kv(l, w)*kv(l, w*r_over_a)
    return np.where(r_over_a <= 1, core, clad)

ra = np.linspace(1e-3, 3, 400)
checked = []
for row in modes:
    l, u, w = row["l"], row["u"], row["w"]
    p = profile(l, u, w, ra)
    # continuity at r = a
    assert abs(profile(l, u, w, np.array([0.999]))[0] - profile(l, u, w, np.array([1.001]))[0]) < 0.02
    checked.append((l, row["m"]))
print("field profiles continuous at the core-cladding boundary for:", checked)"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
xs = np.linspace(0.05, 8, 400)
for l, col in zip((0, 1, 2), ("#4C78A8", "#E45756", "#54A24B")):
    ax[0].plot(xs, jv(l, xs), color=col, label=f"$J_{l}$ (core)")
    ax[0].plot(xs, kv(l, xs), "--", color=col, label=f"$K_{l}$ (cladding)")
ax[0].set_ylim(-0.6, 1.2); ax[0].axhline(0, color="gray", lw=0.6)
ax[0].set_xlabel("x"); ax[0].set_title("oscillating $J_\\ell$ vs decaying $K_\\ell$"); ax[0].legend(fontsize=7, ncol=2)
# field profiles of the first few modes
for row, col in zip(modes[:4], ("#4C78A8", "#E45756", "#54A24B", "#B279A2")):
    ax[1].plot(ra, profile(row["l"], row["u"], row["w"], ra), color=col, label=f"LP{row['l']}{row['m']}")
ax[1].axvline(1.0, ls=":", color="gray"); ax[1].set_xlabel("r / a"); ax[1].set_ylabel("field")
ax[1].set_title("LP mode profiles (core | cladding)"); ax[1].legend(fontsize=8)
# b-V style: number of guided modes vs V
Vs = np.linspace(1.0, 8.0, 60)
ax[2].plot(Vs, [len(lp_modes(V, l_max=6)) for V in Vs], color="#54A24B")
ax[2].axvline(2.405, ls=":", color="#E45756"); ax[2].text(2.45, 1.2, "2.405\\nsingle-mode", fontsize=8)
ax[2].set_xlabel("V number"); ax[2].set_ylabel("guided LP modes"); ax[2].set_title("mode count vs V")
plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- A guided fiber mode is **oscillatory in the core** ($J_\ell$) and **evanescent in the cladding**
  ($K_\ell$, the modified Bessel function that decays like $e^{-w r/a}$, SymPy-verified).
- Matching field and slope at the boundary gives the **characteristic equation**
  $u\,J_{\ell+1}(u)/J_\ell(u)=w\,K_{\ell+1}(w)/K_\ell(w)$ with $u^2+w^2=V^2$; its roots are the
  $\mathrm{LP}_{\ell m}$ modes (solved with SciPy, $b\in(0,1)$, $n_\mathrm{eff}$ tabulated).
- **Cutoffs are Bessel zeros**: $\mathrm{LP}_{11}$ at $V=2.405$ (first zero of $J_0$) sets the single-mode
  limit -- the mode count jumps 1 -> 2 there (verified).

Subject-verb-object: the core oscillates; the cladding decays; the boundary matches them; the Bessel zeros
count the modes."""),
]

write("lp_modes", "characteristic_equation", cells)
