"""Generate notebooks/dirac_antimatter.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# The Dirac equation and the prediction of antimatter

Special relativity says a free particle's energy and momentum obey $E^2=(pc)^2+(mc^2)^2$, so
$E=\pm\sqrt{(pc)^2+(mc^2)^2}$ -- there is a **negative** root. Schrodinger's equation, first order in
time but second order in space, cannot represent this symmetrically; the Klein-Gordon equation
(second order in both) restores it but brings negative probability densities. In 1928 **Dirac**
insisted on an equation **first order in both** space and time,
$$i\hbar\,\partial_t\psi=\big(c\,\boldsymbol\alpha\cdot\mathbf p+\beta mc^2\big)\psi,$$
and squaring it back to $E^2=(pc)^2+(mc^2)^2$ forces $\boldsymbol\alpha,\beta$ to **anticommute** --
they cannot be numbers, they must be $4\times4$ matrices. The price of that four-component wavefunction
is two extra solutions with negative energy, which Dirac reinterpreted as a new particle of the same
mass and opposite charge: the **positron** (antielectron), found by Anderson in 1932.

This notebook derives that chain: the two energy roots, the Clifford algebra the matrices must satisfy,
the gamma matrices in SymPy, and the Dirac Hamiltonian's $\pm E$ eigenvalues (electron and positron) in
NumPy. Self-contained: NumPy, SymPy, Pandas, Matplotlib."""),
setup_cell(),

md(r"""## The two energy roots and the mass gap

$E^2=(pc)^2+(mc^2)^2$ has two branches: $E=+\sqrt{\cdots}$ (electron) and $E=-\sqrt{\cdots}$ (the
antiparticle branch), separated by a gap of $2mc^2$ at $p=0$. SymPy solves for both; the plot is the
relativistic dispersion relation."""),
co("""p, mc2, pc = sp.symbols('p m_c2 pc', real=True, positive=True)
E = sp.symbols('E', real=True)
roots = sp.solve(sp.Eq(E**2, pc**2 + mc2**2), E)
print("E =", roots, "  (two branches; gap 2 m c^2 at p=0)")
assert set(roots) == {sp.sqrt(pc**2+mc2**2), -sp.sqrt(pc**2+mc2**2)}

mc2_MeV = C.M_E*C.C**2/C.E/1e6                      # electron rest energy in MeV
print(f"electron rest energy m c^2 = {mc2_MeV:.4f} MeV")
print(f"mass gap 2 m c^2 = {2*mc2_MeV:.4f} MeV  (pair-production / annihilation energy)")
assert abs(mc2_MeV - 0.511) < 1e-3"""),

md(r"""## Why the coefficients must be matrices (Clifford algebra)

Write $E=c\,\boldsymbol\alpha\cdot\mathbf p+\beta mc^2$ and square it. The cross terms cancel and the
squares reproduce $E^2=(pc)^2+(mc^2)^2$ **only if**
$$\alpha_i\alpha_j+\alpha_j\alpha_i=2\delta_{ij}\,\mathbb 1,\quad
\alpha_i\beta+\beta\alpha_i=0,\quad \alpha_i^2=\beta^2=\mathbb 1.$$
No four ordinary numbers anticommute like this, so $\alpha_i,\beta$ are matrices -- the smallest that
work are $4\times4$. In the Dirac representation they are built from the $2\times2$ Pauli matrices."""),
co("""sx = sp.Matrix([[0,1],[1,0]]); sy = sp.Matrix([[0,-sp.I],[sp.I,0]]); sz = sp.Matrix([[1,0],[0,-1]])
I2, Z2, I4 = sp.eye(2), sp.zeros(2), sp.eye(4)
blk = lambda a,b,c,d: sp.Matrix(sp.BlockMatrix([[a,b],[c,d]]))
ax = blk(Z2,sx,sx,Z2); ay = blk(Z2,sy,sy,Z2); az = blk(Z2,sz,sz,Z2)   # alpha_i = [[0,sigma],[sigma,0]]
beta = blk(I2,Z2,Z2,-I2)                                              # beta = diag(I, -I)
alphas = [ax, ay, az]
anti = lambda M,Nn: M*Nn + Nn*M
for i in range(3):
    for j in range(3):
        assert anti(alphas[i], alphas[j]) == (2 if i==j else 0)*I4    # {a_i,a_j}=2 delta_ij
    assert anti(alphas[i], beta) == sp.zeros(4)                       # {a_i, beta}=0
    assert alphas[i]**2 == I4
assert beta**2 == I4
print("Clifford algebra verified: {a_i,a_j}=2 delta_ij,  {a_i,beta}=0,  a_i^2=beta^2=I")"""),

md(r"""## The gamma matrices and the covariant algebra

Define $\gamma^0=\beta$, $\gamma^i=\beta\alpha_i$. They satisfy the compact covariant form
$\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu}\mathbb 1$ with the metric $g=\mathrm{diag}(1,-1,-1,-1)$ -- the
defining relation of the Dirac (Clifford) algebra, from which all of relativistic quantum mechanics
follows."""),
co("""g0 = beta; gammas = [g0, beta*ax, beta*ay, beta*az]
metric = sp.diag(1,-1,-1,-1)
for mu in range(4):
    for nu in range(4):
        assert anti(gammas[mu], gammas[nu]) == 2*metric[mu,nu]*I4
print("{gamma^mu, gamma^nu} = 2 g^{mu nu} I  verified for all 16 pairs")"""),

md(r"""## The Dirac Hamiltonian: electron and positron energies

$H=c\,\boldsymbol\alpha\cdot\mathbf p+\beta mc^2$ is, for a fixed momentum, a $4\times4$ Hermitian
matrix. Its four eigenvalues are $\pm\sqrt{(pc)^2+(mc^2)^2}$, each **doubly degenerate** -- the two
signs are the electron and the positron, the double degeneracy is spin up/down. Diagonalizing it
numerically (units $c=m=\hbar=1$, momentum along $z$) reproduces the relativistic energies exactly."""),
co("""anp = lambda M: np.array(M.tolist(), dtype=complex)
AX, AY, AZ, B = anp(ax), anp(ay), anp(az), anp(beta)
def dirac_H(px, py, pz, mass=1.0, c=1.0):
    return c*(AX*px + AY*py + AZ*pz) + B*mass*c**2

rows = []
for pz in (0.0, 0.5, 1.0, 2.0):
    ev = np.sort(np.linalg.eigvalsh(dirac_H(0,0,pz)))
    Erel = np.sqrt(pz**2 + 1.0)                       # sqrt((pc)^2+(mc^2)^2), c=m=1
    rows.append({"p_z": pz, "eigenvalues": np.round(ev,3).tolist(), "+-sqrt(p^2+1)": round(Erel,3)})
    assert np.allclose(ev, [-Erel,-Erel,+Erel,+Erel])  # two -E (positron), two +E (electron)
print(pd.DataFrame(rows).to_string(index=False))
print("\\neigenvalues are +-sqrt((pc)^2+(mc^2)^2), each doubly degenerate (spin up/down)")"""),

md(r"""## Antimatter: reading the negative-energy branch

Dirac could not discard the negative-energy solutions (a positive-energy electron would cascade into
them). His resolution -- and the modern Feynman-Stuckelberg reading -- is that a filled negative-energy
state behaves as a **positron**: same mass $m_e$, charge $+e$, and (by CPT) an identical spectrum to the
electron. Consequences that follow directly:

- **Pair production**: a photon of energy $\ge 2m_ec^2=1.022\ \mathrm{MeV}$ can create an $e^-e^+$ pair.
- **Annihilation**: $e^-+e^+\to\gamma\gamma$, two $511\ \mathrm{keV}$ photons back-to-back (the basis of
  PET imaging).
- **Antihydrogen**: a positron bound to an antiproton has the same $-13.6/n^2\ \mathrm{eV}$ levels as
  hydrogen (`dgs/hydrogen_atom.py` flags this CPT identity)."""),
co("""E_photon_for_pair = 2*mc2_MeV
print(f"pair-production threshold (photon energy): {E_photon_for_pair:.3f} MeV")
print(f"annihilation photons: 2 x {mc2_MeV*1e3:.1f} keV, emitted back-to-back")
# a 511 keV gamma-ray wavelength (for scale): lambda = hc/E
lam_pm = C.H*C.C/(mc2_MeV*1e6*C.E)*1e12
print(f"511 keV photon wavelength = {lam_pm:.3f} pm  (a gamma ray)")
assert abs(E_photon_for_pair - 1.022) < 2e-3"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
pp = np.linspace(-3, 3, 400)
ax[0].plot(pp, np.sqrt(pp**2+1), color="#4C78A8", label="electron  $+\\sqrt{(pc)^2+(mc^2)^2}$")
ax[0].plot(pp, -np.sqrt(pp**2+1), color="#E45756", label="positron  $-\\sqrt{\\cdots}$")
ax[0].fill_between(pp, -1, 1, color="gray", alpha=0.12)
ax[0].annotate("", xy=(0,1), xytext=(0,-1), arrowprops=dict(arrowstyle="<->"))
ax[0].text(0.1, 0, "gap $2mc^2$", va="center")
ax[0].set_xlabel("momentum $p$ (units $mc$)"); ax[0].set_ylabel("energy $E$ (units $mc^2$)")
ax[0].set_title("Dirac dispersion: two branches, mass gap"); ax[0].legend(fontsize=8)
# eigenvalue branches vs momentum from the 4x4 Hamiltonian
ps = np.linspace(0, 3, 60)
evs = np.array([np.sort(np.linalg.eigvalsh(dirac_H(0,0,p))) for p in ps])
for k in range(4):
    ax[1].plot(ps, evs[:,k], ".", color=["#E45756","#E45756","#4C78A8","#4C78A8"][k], ms=3)
ax[1].plot(ps, np.sqrt(ps**2+1), "k-", lw=0.8); ax[1].plot(ps, -np.sqrt(ps**2+1), "k-", lw=0.8)
ax[1].set_xlabel("momentum $p_z$"); ax[1].set_ylabel("Hamiltonian eigenvalues")
ax[1].set_title("4x4 Dirac H: +-E, each 2-fold (spin)")
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- Relativity's $E^2=(pc)^2+(mc^2)^2$ has a **negative-energy root**; Dirac's first-order equation keeps
  it and forces the coefficients $\alpha_i,\beta$ to obey the **Clifford algebra**
  $\{\alpha_i,\alpha_j\}=2\delta_{ij}$, $\{\alpha_i,\beta\}=0$ -- so they are $4\times4$ matrices
  (verified in SymPy, with $\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu}$).
- The Dirac Hamiltonian's eigenvalues are $\pm\sqrt{(pc)^2+(mc^2)^2}$, doubly degenerate: **electron and
  positron, spin up and down** (verified in NumPy).
- The negative branch is **antimatter** -- the positron, $m_ec^2=0.511\ \mathrm{MeV}$, with pair
  production at $1.022\ \mathrm{MeV}$ and $511\ \mathrm{keV}$ annihilation gammas.

Subject-verb-object: relativity demands a negative root; Dirac linearizes the equation; the algebra
forces matrices; the matrices predict the positron. (Note: this is elementary-particle physics on its
own terms -- it is *not* a dependency of the classical dispersion phase-recovery pipeline, whose only
quantum input is photon shot noise.)"""),
]

write("dirac", "antimatter", cells)
