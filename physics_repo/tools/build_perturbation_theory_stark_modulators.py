"""Generate notebooks/perturbation_theory_stark_modulators.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Perturbation theory and the Stark effect: the physics of photonic modulators

Griffiths' time-independent perturbation theory (Ch. 6) answers a device question: what happens to a
quantum system's energy levels when you apply an external electric field? The answer is the **Stark
effect**, and the Stark effect *is* how every electro-optic modulator works -- the component that imprints
an electrical signal onto light before it enters the dispersion / time-stretch stage of this instrument.

Two orders, two devices:

- **First order**, $E_n^{(1)}=\langle n|H'|n\rangle$. For a symmetric state this vanishes by parity --
  *unless* levels are **degenerate**, where degenerate perturbation theory gives a shift **linear** in the
  field. That linear Stark effect is the **Pockels effect**: $\Delta n\propto E$, the basis of lithium-
  niobate Mach-Zehnder modulators, which need a crystal *without* inversion symmetry.
- **Second order**, $E_n^{(2)}=\sum_{m\neq n}\dfrac{|\langle m|H'|n\rangle|^2}{E_n^{(0)}-E_m^{(0)}}$, always
  present and $\propto E^2$. This quadratic Stark shift is the **Kerr effect** and, in a quantum well, the
  **quantum-confined Stark effect (QCSE)** that red-shifts the absorption edge -- the basis of high-speed
  **electro-absorption modulators** in datacenter optical links.

We model the QCSE directly: a particle in a box (a quantum well) with an applied field. Perturbation theory
is verified term by term against **exact diagonalization**, we extract the polarizability, show where
perturbation theory breaks down, and then treat the degenerate (linear/Pockels) case.

Self-contained: NumPy, SymPy, Pandas, Matplotlib. Units $\hbar=m=L=1$; field strength $f=eE$ is the
perturbation parameter."""),
setup_cell(),

md(r"""## The unperturbed well and the dipole matrix elements (SymPy)

The infinite square well on $[0,L]$ has $\psi_n(x)=\sqrt{2/L}\,\sin(n\pi x/L)$ and
$E_n^{(0)}=n^2\pi^2\hbar^2/(2mL^2)$. A uniform field adds $H'=f\,(x-L/2)$ (dipole energy, referenced to the
well centre). The perturbation matrix elements are dipole integrals $\langle m|x|n\rangle$; SymPy gives the
closed form and the **parity selection rule**: $\langle n|H'|n\rangle=0$ (no first-order shift for a
symmetric well), and $\langle m|x|n\rangle\neq0$ only when $m+n$ is odd."""),
co("""x, L = sp.symbols('x L', positive=True)
psi = lambda k: sp.sqrt(2/L)*sp.sin(k*sp.pi*x/L)
me = lambda mm, nn: sp.integrate(psi(mm)*x*psi(nn), (x, 0, L))    # <m|x|n>

assert sp.simplify(me(1, 1) - L/2) == 0                          # <n|x|n> = L/2  -> <n|H'|n> = 0
assert sp.simplify(me(2, 1) - (-16*L/(9*sp.pi**2))) == 0         # m+n odd: nonzero dipole coupling
assert sp.simplify(me(3, 1)) == 0                                # m+n even: forbidden (parity)
print("<1|x|1> =", sp.simplify(me(1,1)), "  -> first-order Stark shift <n|H'|n> = <n|x|n> - L/2 = 0 (parity)")
print("<2|x|1> =", sp.simplify(me(2,1)), "   <3|x|1> =", me(3,1), " (selection rule: m+n odd)")

# closed form used to build the matrix numerically: <m|x|n> = -8 m n L / (pi^2 (m^2-n^2)^2) for m+n odd
def dipole_matrix(N, Lval=1.0):
    M = np.full((N, N), 0.0)
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i == j:                 M[i-1, j-1] = Lval/2
            elif (i + j) % 2 == 1:     M[i-1, j-1] = -8*i*j*Lval/(np.pi**2*(i**2 - j**2)**2)
    return M
Xmat = dipole_matrix(6)
assert abs(Xmat[1,0] - (-16/(9*np.pi**2))) < 1e-12               # matches SymPy <2|x|1>
print("dipole matrix (6x6) built; <2|x|1> numeric =", round(Xmat[1,0], 6))"""),

md(r"""## Second-order (quadratic) Stark shift = the quantum-confined Stark effect

With $H'=f(x-L/2)$, the ground state has $E_1^{(1)}=0$ and
$$E_1^{(2)}=f^2\sum_{m\ge2}\frac{|\langle m|x|1\rangle|^2}{E_1^{(0)}-E_m^{(0)}}<0,$$
a **downward, quadratic** shift -- the field polarizes the state and lowers its energy. The coefficient is
minus half the **polarizability**, $E_1^{(2)}=-\tfrac12\alpha f^2$. We compute the perturbation sum and
compare the predicted shift to **exact diagonalization** of $H_0+H'$ in the well basis. Perturbation theory
matches the exact shift at small field and departs at large field, where the level is no longer a small
correction."""),
co("""N = 40
E0 = np.array([n**2*np.pi**2/2 for n in range(1, N+1)])          # E_n^(0), hbar=m=L=1
X = dipole_matrix(N)
Hp = X - 0.5*np.eye(N)                                           # H' = (x - 1/2); diagonal is exactly 0

# second-order coefficient and polarizability from the PT sum (ground state n=1)
terms = np.array([X[0, m]**2/(E0[0] - E0[m]) for m in range(1, N)])
E2_coeff = terms.sum()                                          # E_1^(2) = E2_coeff * f^2  (<0)
alpha = -2*E2_coeff                                             # polarizability (>0)
print(f"E_1^(1) = {Hp[0,0]:.3e} (=0, parity)   E_1^(2)/f^2 = {E2_coeff:.6e}   polarizability alpha = {alpha:.6e}")
assert abs(Hp[0,0]) < 1e-15 and E2_coeff < 0

def exact_ground_shift(f):
    return np.linalg.eigvalsh(np.diag(E0) + f*Hp)[0] - E0[0]

for f in (0.5, 2.0):
    pt, ex = E2_coeff*f**2, exact_ground_shift(f)
    print(f"f={f}:  PT (2nd order) {pt:.6f}   exact {ex:.6f}   rel.err {abs(pt-ex)/abs(ex):.2e}")
    assert abs(pt - ex)/abs(ex) < 0.05                          # PT accurate at small field
# even in f (no linear term): shift(+f) == shift(-f)
assert abs(exact_ground_shift(3.0) - exact_ground_shift(-3.0)) < 1e-9
print("shift is even in f (E^(1)=0): quadratic Stark / QCSE red-shift confirmed")"""),

md(r"""## Convergence of the perturbation sum (Pandas)

The sum is dominated by the nearest coupled level ($m=2$): the parity rule kills even-$m+n$ terms and the
energy denominator suppresses distant ones. A three-term sum already captures the polarizability."""),
co("""rows, running = [], 0.0
for m in range(1, 8):
    if X[0, m] != 0.0:
        t = X[0, m]**2/(E0[0] - E0[m]); running += t
        rows.append({"level m": m+1, "|<m|x|1>|^2": round(X[0, m]**2, 6),
                     "E1-Em": round(E0[0]-E0[m], 3), "term": f"{t:.3e}",
                     "running E2/f^2": f"{running:.6e}"})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\\nfull sum E2/f^2 = {E2_coeff:.6e}  (m=2 alone gives {X[0,1]**2/(E0[0]-E0[1]):.6e})")"""),

md(r"""## Degenerate perturbation theory = linear Stark = the Pockels effect

When two states share an energy $E_0$ and the field couples them, first-order **degenerate** perturbation
theory diagonalizes the $2\times2$ block $\begin{psmallmatrix}E_0 & fV\\ fV & E_0\end{psmallmatrix}$, giving
$E_\pm=E_0\pm f|V|$ -- a splitting **linear** in the field. This is the linear Stark effect (as in hydrogen
$n=2$, where $2s$ and $2p$ are degenerate) and, in materials, the **Pockels effect**: a refractive-index
change $\Delta n\propto E$. It requires broken inversion symmetry -- exactly why lithium niobate (non-
centrosymmetric) is the workhorse Mach-Zehnder modulator crystal, while symmetric media show only the
quadratic Kerr/QCSE response above."""),
co("""E0d, V = 1.0, 0.3
def two_level(f):
    return np.linalg.eigvalsh(np.array([[E0d, f*V], [f*V, E0d]]))
for f in (0.0, 0.5, 1.0):
    ev = two_level(f)
    assert np.allclose(ev, [E0d - abs(f*V), E0d + abs(f*V)])    # E_+- = E0 +- f|V|  (linear)
    print(f"f={f}:  levels = {np.round(ev,3)}   splitting = {ev[1]-ev[0]:.3f}  (= 2 f|V|, linear in field)")
print("degenerate -> linear Stark (Pockels); non-degenerate -> quadratic Stark (Kerr/QCSE)")"""),

md(r"""## Engineering: from Stark shifts to modulator specifications

**Pockels / Mach-Zehnder (linear).** $\Delta n=-\tfrac12 n^3 r\,E$ with electro-optic coefficient $r$. A
Mach-Zehnder modulator biases one arm to a $\pi$ phase difference at the **half-wave voltage**
$V_\pi=\dfrac{\lambda\,d}{n^3 r\,L}$ ($d$ electrode gap, $L$ length). Lower $V_\pi$ = less drive power.

**QCSE / electro-absorption (quadratic).** The field red-shifts the exciton edge by
$\Delta E=-\tfrac12\alpha E^2$; at fixed laser wavelength this swings the material between transparent and
absorbing -- a compact, low-drive intensity modulator integrated in datacenter transceivers."""),
co("""# representative lithium-niobate Mach-Zehnder half-wave voltage
lam = 1.55e-6      # m (telecom C-band)
n_r = 2.2          # extraordinary index of LiNbO3
r33 = 30e-12       # m/V, electro-optic coefficient
d   = 8e-6         # m, electrode gap
Lm  = 2e-2         # m, interaction length
Vpi = lam*d/(n_r**3*r33*Lm)
print(f"LiNbO3 Mach-Zehnder V_pi ~ {Vpi:.2f} V  (linear Pockels; lower is better)")
assert 1.0 < Vpi < 10.0

# QCSE red-shift at a typical quantum-well field, using our dimensionless polarizability as the scaling law
E_field_rel = 3.0
print(f"QCSE: quadratic red-shift ~ -1/2 * alpha * E^2, here {-0.5*alpha*E_field_rel**2:.4f} (well units) "
      f"at f={E_field_rel}")
print("linear (Pockels) modulates PHASE; quadratic (QCSE) modulates ABSORPTION -> intensity")"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
# (1) quadratic Stark: PT parabola vs exact ground-state shift, and the breakdown at large field
fs = np.linspace(0, 30, 60)
ex = np.array([exact_ground_shift(f) for f in fs])
ax[0].plot(fs, E2_coeff*fs**2, "--", color="#4C78A8", label=r"2nd-order PT $-\frac{1}{2}\alpha f^2$")
ax[0].plot(fs, ex, color="#E45756", label="exact diagonalization")
ax[0].set_xlabel("field strength f"); ax[0].set_ylabel(r"ground-state shift $\Delta E_1$")
ax[0].set_title("quadratic Stark (QCSE): PT vs exact"); ax[0].legend(fontsize=8)
# (2) the well tilts under the field: potential and |psi_1|^2 pushed to one side
xg = np.linspace(0, 1, 300)
for f, col in [(0.0, "#4C78A8"), (20.0, "#E45756")]:
    H = np.diag(E0) + f*Hp
    w, v = np.linalg.eigh(H)
    # reconstruct ground-state wavefunction on the grid from the box basis
    basis = np.array([np.sqrt(2)*np.sin(k*np.pi*xg) for k in range(1, N+1)])
    psi1 = v[:, 0] @ basis
    ax[1].plot(xg, psi1**2/np.trapezoid(psi1**2, xg) + 0*f, color=col, label=f"f={f:.0f}")
ax[1].set_xlabel("x / L"); ax[1].set_ylabel(r"$|\psi_1(x)|^2$")
ax[1].set_title("field polarizes the ground state (dipole)"); ax[1].legend(fontsize=8)
# (3) degenerate case: linear splitting (Pockels) vs quadratic single-level (Kerr)
fs2 = np.linspace(-1.5, 1.5, 100)
ax[2].plot(fs2, [two_level(f)[0] for f in fs2], color="#E45756")
ax[2].plot(fs2, [two_level(f)[1] for f in fs2], color="#E45756", label="degenerate: linear (Pockels)")
ax[2].plot(fs2, E0d + E2_coeff*fs2**2*40, color="#4C78A8", label="non-degenerate: quadratic (Kerr/QCSE)")
ax[2].set_xlabel("field strength f"); ax[2].set_ylabel("energy")
ax[2].set_title("linear vs quadratic Stark"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()"""),

md(r"""## Exercises

1. **Third-order / higher fields.** Extend the exact diagonalization and fit $\Delta E_1=c_2 f^2+c_4 f^4+
   \dots$; identify the field where the $f^4$ term reaches 10% of the $f^2$ term (the perturbation-theory
   validity limit).
2. **Hydrogen linear Stark.** For the $n=2$ manifold ($2s$, $2p_0$), build the $H'=eEz$ block and show the
   $\pm3ea_0E$ linear splitting -- the atomic Pockels analogue.
3. **$V_\pi L$ figure of merit.** Compare lithium niobate ($r_{33}\approx30$ pm/V) with a hypothetical
   polymer ($r\approx100$ pm/V); which gives the lower $V_\pi$ at equal length, and why does bandwidth
   favor short devices?
4. **QCSE tunability.** With $\Delta E=-\tfrac12\alpha E^2$, estimate the field needed to red-shift an
   exciton by one linewidth, and relate it to the on/off contrast of an electro-absorption modulator.

## Summary

- **First-order** Stark shift vanishes for symmetric states by **parity** ($\langle n|H'|n\rangle=0$),
  verified in SymPy; the coupling $\langle m|x|n\rangle$ obeys the selection rule $m+n$ odd.
- **Second-order** perturbation theory gives the **quadratic** Stark / **quantum-confined Stark effect**:
  $E_1^{(2)}=-\tfrac12\alpha f^2<0$, matching **exact diagonalization** at small field and breaking down at
  large field (verified; polarizability extracted; sum dominated by the nearest level).
- **Degenerate** perturbation theory gives the **linear** Stark effect -- the **Pockels effect** requiring
  broken inversion symmetry (lithium niobate Mach-Zehnder modulators, $V_\pi\sim$ volts).
- Perturbation order maps onto device physics: linear $\to$ phase modulators (Pockels), quadratic $\to$
  absorption modulators (QCSE). The modulator that encodes the signal in this instrument is an engineered
  Stark effect.

Subject-verb-object: the field perturbs the levels; parity forbids the first-order shift; the second order
polarizes and red-shifts; degeneracy restores the linear response the modulator exploits."""),
]

write("perturbation_theory", "stark_modulators", cells)
