"""Generate notebooks/feynman_virtual_photon.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# The virtual photon: how a photon carries the electromagnetic force

Two electrons repel. In quantum electrodynamics (QED) that force is not action at a distance -- it is the
**exchange of a virtual photon**. One electron emits a photon, the other absorbs it, and the recoil at
each end *is* the force. The Feynman diagram for electron-electron scattering ($e^-e^-\to e^-e^-$,
Moller scattering) is two electron lines exchanging a single wavy photon line.

The photon in that diagram is **virtual**: it is *off the mass shell*. A real photon obeys $E=pc$, i.e.
its four-momentum satisfies $q^2=0$; the exchanged photon carries four-momentum $q=p_1-p_3$ with
$q^2\neq0$, so it could never exist as a free particle. It exists only for the brief instant permitted by
the energy-time uncertainty relation, and four-momentum **is** conserved exactly at every vertex -- the
"violation" is only of the free-particle energy-momentum relation, not of any conservation law.

This notebook makes that picture quantitative and self-contained:
- the photon **propagator** $\sim 1/q^2$ and its Fourier transform, which *is* the Coulomb potential;
- the massive generalization (Yukawa), showing **force range $\sim1/(\text{mediator mass})$**;
- the off-shell (spacelike) momentum transfer in $t$-channel scattering;
- the propagator squared reproducing the **Rutherford** $1/\sin^4(\theta/2)$ cross section;
- energy-time uncertainty setting the range; and a **drawn Feynman diagram**.

Self-contained: NumPy, SymPy, Pandas, Matplotlib."""),
setup_cell(),

md(r"""## The propagator is the potential: Coulomb from photon exchange

The amplitude for exchanging a boson of mass $\mu$ carries the **propagator** $G(q)=1/(q^2+\mu^2)$
(natural units $\hbar=c=1$). Fourier-transforming to position space gives the potential the exchange
produces:
$$V(r)=\int\frac{d^3q}{(2\pi)^3}\,\frac{e^{\,i\mathbf q\cdot\mathbf r}}{q^2+\mu^2}
=\frac{e^{-\mu r}}{4\pi r}\quad\text{(Yukawa)},\qquad
\mu\to0:\;V(r)=\frac{1}{4\pi r}\;\text{(Coulomb)}.$$
Equivalently $V$ solves the screened-Poisson (Helmholtz) equation $(-\nabla^2+\mu^2)V=\delta^3(\mathbf r)$
-- the position-space statement that $1/(q^2+\mu^2)$ is the propagator. SymPy verifies the radial
solution; NumPy verifies the Fourier integral."""),
co("""r, mu = sp.symbols('r mu', positive=True)
# For r>0, radial Laplacian: nabla^2 V = (1/r) (r V)''.  With w = r V, (-nabla^2+mu^2)V=0  <=>  w'' = mu^2 w.
w = sp.exp(-mu*r)/(4*sp.pi)                     # w = r V  for the Yukawa potential V=e^{-mu r}/(4 pi r)
assert sp.simplify(sp.diff(w, r, 2) - mu**2*w) == 0        # Yukawa solves (-nabla^2+mu^2)V=0 for r>0
w0 = sp.Rational(1,1)/(4*sp.pi)                             # mu->0 limit: w=1/(4pi) -> V=1/(4 pi r)
assert sp.simplify(sp.diff(w0, r, 2)) == 0                 # Coulomb solves Laplace eq for r>0
print("Yukawa V=e^{-mu r}/(4 pi r) solves (-nabla^2+mu^2)V=0;  Coulomb 1/(4 pi r) solves nabla^2 V=0")"""),

co("""# Numerical Fourier transform: V(r) = (1/(2 pi^2 r)) int_0^inf q sin(q r)/(q^2+mu^2) dq  =  e^{-mu r}/(4 pi r)
def V_fourier(rr, mval, qmax=1500.0, nq=600_000):
    q = np.linspace(1e-6, qmax, nq)
    integ = q*np.sin(q*rr)/(q**2 + mval**2)
    return np.trapz(integ, q)/(2*np.pi**2*rr) if hasattr(np,'trapz') else np.trapezoid(integ, q)/(2*np.pi**2*rr)
for mval, name in [(0.8, "Yukawa mu=0.8"), (1e-4, "Coulomb mu->0")]:
    rr = 1.3
    num = V_fourier(rr, mval)
    exact = np.exp(-mval*rr)/(4*np.pi*rr)
    print(f"{name:16s}: FT V({rr}) = {num:.5f}   exact e^(-mu r)/(4 pi r) = {exact:.5f}")
    assert abs(num-exact) < 1e-3"""),

md(r"""## Range of a force = Compton wavelength of the mediator

The exchanged mass sets the range: $V\propto e^{-\mu r}$ falls off over $R=1/\mu=\hbar/(Mc)$, the
**Compton wavelength** of the mediator. A **massless** photon ($M=0$) gives infinite range -- the
$1/r$ Coulomb law. A **massive** mediator gives a short-range force. This is Yukawa's 1935 argument
run in reverse: the ~1 fm range of the nuclear force implies a mediator of ~200 MeV (the pion, 140 MeV);
the ~$10^{-3}$ fm weak force implies the ~80 GeV $W$ boson."""),
co("""hbarc = C.HBAR*C.C/C.E*1e9                       # hbar c in MeV.fm
print(f"hbar c = {hbarc:.3f} MeV.fm")
assert abs(hbarc - 197.33) < 0.1
rows = []
for name, M_MeV, force in [("photon", 0.0, "electromagnetic"),
                           ("pion (Yukawa)", 140.0, "nuclear (residual strong)"),
                           ("W boson", 80400.0, "weak")]:
    R = np.inf if M_MeV == 0 else hbarc/M_MeV        # range = hbar/(M c) in fm
    rows.append({"mediator": name, "mass [MeV]": M_MeV, "force": force,
                 "range hbar/Mc [fm]": ("infinite" if R==np.inf else f"{R:.4g}")})
print(pd.DataFrame(rows).to_string(index=False))
assert abs(hbarc/140.0 - 1.409) < 0.01                # pion -> ~1.4 fm nuclear range"""),

md(r"""## The exchanged photon is off-shell (virtual)

In elastic $e^-e^-$ scattering an electron of momentum $\mathbf k$ scatters by angle $\theta$ keeping
$|\mathbf k|$. The four-momentum handed to the photon is $q=p_1-p_3$; its invariant is the Mandelstam
variable $t=q^2=-4k^2\sin^2(\theta/2)<0$ (spacelike). A **real** photon has $q^2=0$; here $q^2<0$ for any
$\theta>0$, so the photon is **virtual** -- it can never satisfy $E=pc$. Four-momentum is still conserved
exactly at each vertex ($p_1=p_3+q$); only the free-particle relation is off. SymPy confirms $t<0$ and the
photon's non-zero invariant mass-squared."""),
co("""k, theta = sp.symbols('k theta', positive=True)
t = -4*k**2*sp.sin(theta/2)**2                    # Mandelstam t = q^2 for elastic scattering
assert sp.simplify(t) == -4*k**2*sp.sin(theta/2)**2
# q^2 = t < 0 for 0<theta<pi (spacelike => virtual); a real photon would need q^2=0
val = t.subs({k:1, theta:sp.pi/2})
print(f"q^2 = t = -4 k^2 sin^2(theta/2);  at theta=90deg, k=1:  q^2 = {float(val):.3f}  (<0 => virtual)")
assert val < 0"""),

md(r"""## The propagator squared is the Rutherford cross section

The scattering amplitude is (vertex)$\times$(propagator)$\times$(vertex) $\sim e^2/q^2=e^2/t$. The cross
section is $|\mathcal M|^2\sim e^4/t^2$. Substituting $t=-4k^2\sin^2(\theta/2)$,
$$\frac{d\sigma}{d\Omega}\propto\frac{1}{t^2}=\frac{1}{16k^4\sin^4(\theta/2)}\propto\frac{1}
{\sin^4(\theta/2)},$$
the **Rutherford law**. The famous $1/\sin^4(\theta/2)$ that Rutherford measured off gold foil is
nothing but the **square of the $1/q^2$ virtual-photon propagator**. SymPy verifies the identity;
NumPy plots the steep forward rise."""),
co("""prop_sq = 1/t**2
rutherford = 1/(16*k**4*sp.sin(theta/2)**4)
assert sp.simplify(prop_sq - rutherford) == 0        # 1/q^4  ==  1/sin^4(theta/2) law
print("1/t^2 = 1/(16 k^4 sin^4(theta/2))  ->  dsigma/dOmega ~ 1/sin^4(theta/2)  (Rutherford)")
th = np.linspace(np.deg2rad(5), np.deg2rad(175), 400)
dsig = 1/np.sin(th/2)**4
print(f"cross section ratio, 10deg vs 90deg: {(1/np.sin(np.deg2rad(10)/2)**4)/(1/np.sin(np.deg2rad(90)/2)**4):.0f}x")"""),

md(r"""## Energy-time uncertainty: how the virtual photon is allowed to exist

The virtual photon violates $E=pc$ by an amount $\Delta E$, so it can exist only for $\Delta t\sim\hbar/
\Delta E$ (energy-time uncertainty), reaching a distance $R\sim c\,\Delta t\sim\hbar c/\Delta E$. For a
mediator of mass $M$ the minimum off-shell cost is $\Delta E\sim Mc^2$, giving $R\sim\hbar/(Mc)$ -- the
same Compton-wavelength range as the Yukawa fall-off, now from the uncertainty principle. The photon's
$M=0$ makes $R\to\infty$: the electromagnetic force has unlimited range because its mediator is massless.
"""),
co("""for name, M_MeV in [("pion", 140.0), ("W boson", 80400.0)]:
    dE = M_MeV                                        # min off-shell energy ~ M c^2 (MeV)
    R = hbarc/dE                                      # reach in fm
    print(f"{name:8s}: Delta E ~ {M_MeV:>7.1f} MeV  ->  reach hbar c/Delta E = {R:.4g} fm")
print("photon:  Delta E -> 0 (massless)  ->  reach -> infinity  (long-range Coulomb)")"""),

md(r"""## Plots: the Rutherford rise and the Feynman diagram"""),
co(r"""fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

# (left) Rutherford cross section from the squared propagator
ax[0].semilogy(np.rad2deg(th), dsig, color="#4C78A8", lw=2)
ax[0].set_xlabel("scattering angle theta [deg]"); ax[0].set_ylabel(r"$d\sigma/d\Omega \propto 1/\sin^4(\theta/2)$")
ax[0].set_title("propagator$^2$ = Rutherford cross section")

# (right) the t-channel Feynman diagram: two electrons exchange a virtual photon
axd = ax[1]; axd.set_xlim(0, 10); axd.set_ylim(0, 10); axd.axis("off")
axd.set_title("Moller scattering: $e^-e^-$ exchange a virtual photon")
arrow = dict(arrowstyle="-|>", color="#333", lw=1.8)
# electron fermion lines: incoming from the left to a vertex, then outgoing to the right
# lower line: p1 -> vertex(5,3) -> p3 ;  upper line: p2 -> vertex(5,7) -> p4
axd.annotate("", xy=(5,3), xytext=(1,1), arrowprops=arrow)   # p1 in
axd.annotate("", xy=(9,1), xytext=(5,3), arrowprops=arrow)   # p3 out
axd.annotate("", xy=(5,7), xytext=(1,9), arrowprops=arrow)   # p2 in
axd.annotate("", xy=(9,9), xytext=(5,7), arrowprops=arrow)   # p4 out
axd.text(0.5,0.5,"$e^-\\ p_1$",fontsize=11); axd.text(9.1,0.7,"$p_3$",fontsize=11)
axd.text(0.5,9.3,"$e^-\\ p_2$",fontsize=11); axd.text(9.1,9.1,"$p_4$",fontsize=11)
# wavy virtual-photon line between the two vertices (5,3)-(5,7)
yy = np.linspace(3, 7, 200); xx = 5 + 0.22*np.sin(2*np.pi*(yy-3)/0.8)
axd.plot(xx, yy, color="#E45756", lw=1.8)
axd.text(5.35, 5.0, r"virtual photon $\gamma^*$", color="#E45756", fontsize=10)
axd.text(5.35, 4.3, r"$q=p_1-p_3,\ q^2=t<0$", color="#E45756", fontsize=9)
for vx,vy in [(5,3),(5,7)]:
    axd.plot(vx,vy,'o',color="#333",ms=6)            # QED vertices (coupling ~ e)
plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- The electromagnetic force between two electrons is the **exchange of a virtual photon**; its Feynman
  diagram is two electron lines joined by one photon line, with a coupling $\sim e$ at each vertex.
- The photon carries the **propagator** $1/q^2$; its Fourier transform **is** the Coulomb potential
  $1/(4\pi r)$ (SymPy: Yukawa/Coulomb solve the screened-Poisson equation; NumPy: the Fourier integral).
- A massive mediator gives $e^{-\mu r}/(4\pi r)$ with range $\hbar/(Mc)$ -- **force range = mediator
  Compton wavelength** (photon massless $\Rightarrow$ infinite-range Coulomb; pion $\Rightarrow$ ~1.4 fm).
- The exchanged photon is **off-shell**, $q^2=t=-4k^2\sin^2(\theta/2)<0$: it can never be a free photon.
  Four-momentum is conserved exactly at each vertex; only $E=pc$ is (briefly) violated, licensed by
  energy-time uncertainty $\Delta t\sim\hbar/\Delta E$.
- The **square of the propagator** reproduces the **Rutherford** $1/\sin^4(\theta/2)$ cross section --
  the $1/q^2$ virtual photon is directly visible in a scattering measurement.

Subject-verb-object: the electron emits a virtual photon; the propagator sets the potential; the mediator
mass sets the range; the uncertainty principle licenses the exchange. (Like the Dirac notebook, this is
particle physics on its own terms -- not a dependency of the classical dispersion phase-recovery
pipeline.)"""),
]

write("feynman", "virtual_photon", cells)
