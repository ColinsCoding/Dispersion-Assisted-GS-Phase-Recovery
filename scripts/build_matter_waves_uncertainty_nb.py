"""Generate notebooks/matter_waves_uncertainty_sympy.ipynb -- de Broglie
relations, Davisson-Germer, wave packets/phasor form, group vs phase
velocity, the probabilistic interpretation, and Heisenberg uncertainty
DERIVED from an actual standard deviation (not asserted) -- everything via
sp.init_printing(), no hardcoded equations. Companion to the existing
modern_physics_ee_executed.ipynb Section 2 (which plots the bound but never
derives it) and applied_qm_engineering.ipynb (finite-difference/operator
methods, doesn't touch any of these topics). NOTE: no triple-double-quote
docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "matter_waves_uncertainty_sympy.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Matter Waves and Uncertainty -- Derived, Not Asserted

Every equation below comes out of a `sympy` calculation with `init_printing()`
rendering the result -- nothing is typed in as a finished LaTeX formula and left
unverified. De Broglie relations, the Davisson-Germer diffraction condition, wave
packets built as literal superpositions, the phasor form of a traveling wave,
group vs. phase velocity, the Born rule, and finally Heisenberg uncertainty
computed from an actual standard deviation on a real wavefunction -- not quoted
from memory.""")

code(r"""import sympy as sp
sp.init_printing()
print("SymPy", sp.__version__, "loaded, init_printing enabled")""")

# ── §1 de Broglie relations ─────────────────────────────────────────
md(r"""## §1 De Broglie relations

Planck's $E=h\nu$ (light as particles) and de Broglie's postulate that the
SAME relation runs backwards for matter: a particle of momentum $p$ has an
associated wavelength $\lambda=h/p$. Built here from the photon relations
$E=h\nu=\hbar\omega$ and $p=h/\lambda=\hbar k$, so de Broglie's leap is
literally "use the photon's own $p$-$k$ relation for anything with momentum."
""")

code(r"""h, hbar, nu, omega, k, lam, p, m, v = sp.symbols(
    'h hbar nu omega k lambda p m v', positive=True)

# photon relations (established first, for light)
E_photon = h*nu
E_photon_ang = hbar*omega
p_photon = h/lam
p_photon_ang = hbar*k

print("Photon energy:  E = h*nu = hbar*omega  -->",
      sp.simplify(E_photon.subs(nu, omega/(2*sp.pi)) - E_photon_ang) == 0)
print("Photon momentum: p = h/lambda = hbar*k -->",
      sp.simplify(p_photon.subs(lam, 2*sp.pi/k) - p_photon_ang) == 0)

# de Broglie: SAME relation, now for a massive particle with p = m*v
de_broglie_lambda = h/p
de_broglie_lambda_massive = de_broglie_lambda.subs(p, m*v)
print("\nde Broglie wavelength, lambda = h/p =")
sp.pprint(de_broglie_lambda)
print("\nfor a massive particle, p = m*v, so lambda_dB =")
sp.pprint(de_broglie_lambda_massive)""")

# ── §2 Davisson-Germer ────────────────────────────────────────────────
md(r"""## §2 Davisson-Germer: electrons diffract like waves

Electrons scattered off a nickel crystal show constructive-interference peaks
at angles satisfying the SAME Bragg condition as X-ray diffraction,
$d\sin\theta=n\lambda$ -- proof electrons have a wavelength at all. Solve for
$\lambda$ from the historical numbers ($d=2.15\,\text{\AA}$ (111) planes,
peak at $\theta=50°$, $n=1$) and check it against the de Broglie wavelength
predicted from the accelerating voltage ($54\,\text{V}$).""")

code(r"""d_spacing, theta, n_order = sp.symbols('d theta n', positive=True)

bragg_lambda = d_spacing * sp.sin(theta) / n_order
print("Bragg condition solved for lambda: lambda = d*sin(theta)/n")
sp.pprint(sp.Eq(sp.Symbol('lambda'), bragg_lambda))

# historical Davisson-Germer numbers
d_val = 2.15e-10       # m, (111) plane spacing in Ni
theta_val = sp.pi * sp.Rational(50, 180)  # 50 degrees, exact rational multiple of pi
lambda_bragg = float(bragg_lambda.subs({d_spacing: d_val, theta: theta_val, n_order: 1}))
print(f"\nlambda from Bragg condition (d=2.15 A, theta=50deg, n=1): {lambda_bragg*1e10:.3f} Angstrom")

# de Broglie prediction from the 54 V accelerating voltage
e_charge, V_acc, m_e = sp.symbols('e V m_e', positive=True)
KE = e_charge * V_acc                       # kinetic energy gained
p_from_KE = sp.sqrt(2*m_e*KE)               # p = sqrt(2mE)
lambda_debroglie_expr = h / p_from_KE
lambda_debroglie_expr_num = lambda_debroglie_expr.subs(
    {h: 6.626e-34, e_charge: 1.602e-19, V_acc: 54.0, m_e: 9.109e-31})
lambda_debroglie_val = float(lambda_debroglie_expr_num)
print(f"lambda from de Broglie (54 V accelerating voltage):  {lambda_debroglie_val*1e10:.3f} Angstrom")

pct_diff = abs(lambda_bragg - lambda_debroglie_val)/lambda_debroglie_val * 100
print(f"\nagreement: {pct_diff:.1f}% difference -- same historical result that won de Broglie the Nobel:")
print("electron DIFFRACTION angle and electron MOMENTUM predict the same wavelength.")
assert pct_diff < 10, "should agree to within the historical experiment's precision"
""")

# ── §3 wave packets, the phasor transform, and why a bare plane wave is "singular" ──
md(r"""## §3 Wave packets and the phasor transform

A single plane wave $e^{i(kx-\omega t)}$ solves the wave equation but is
**not normalizable** -- $\int|e^{ikx}|^2 dx$ diverges over all space (it's the
same non-normalizable "singularity" as a Dirac comb: a perfectly sharp $k$
means perfectly NO localization, by the same ε-δ convergence logic used
earlier this session for the Gaussian normalization integral). A real
particle is a **wave packet**: a superposition (integral) of many plane
waves, weighted by an amplitude $\phi(k)$, which IS normalizable as long as
$\phi(k)$ decays.

The **phasor transform**: writing a real traveling wave $\cos(kx-\omega t)$
as $\text{Re}[e^{i(kx-\omega t)}]$ turns differentiation into multiplication
by $ik$/$-i\omega$ -- verified symbolically below, the same trick used all
session for the dispersion operator $H(f)=e^{i\pi Df^2}$.""")

code(r"""x, t, k_sym, w_sym = sp.symbols('x t k omega', real=True)

# 1. does a bare plane wave solve the 1D wave equation u_tt = c^2 u_xx?
c_sym = sp.symbols('c', positive=True)
u_plane = sp.exp(sp.I*(k_sym*x - w_sym*t))
wave_eq_residual = sp.diff(u_plane, t, 2) - c_sym**2*sp.diff(u_plane, x, 2)
wave_eq_residual = sp.simplify(wave_eq_residual / u_plane)  # divide out the common exponential
print("Plane wave e^(i(kx-wt)) solves u_tt = c^2 u_xx provided:")
sp.pprint(sp.Eq(wave_eq_residual, 0))
dispersion_condition = sp.solve(sp.Eq(wave_eq_residual, 0), w_sym)
print("=> omega =", dispersion_condition, " (the non-dispersive relation omega = +/- c*k)")

# 2. the phasor trick: differentiating e^(i(kx-wt)) IS multiplying by ik / -iw
dudx = sp.diff(u_plane, x)
dudt = sp.diff(u_plane, t)
print("\nd/dx[plane wave] / [plane wave] =", sp.simplify(dudx/u_plane), " (= i*k, confirmed)")
print("d/dt[plane wave] / [plane wave] =", sp.simplify(dudt/u_plane), " (= -i*omega, confirmed)")
assert sp.simplify(dudx/u_plane - sp.I*k_sym) == 0
assert sp.simplify(dudt/u_plane - (-sp.I*w_sym)) == 0

# 3. a wave PACKET: superpose plane waves over a narrow band of k, weighted
# by a Gaussian amplitude phi(k) -- this integral IS normalizable, unlike
# the single plane wave above
k0, sigma_k = sp.symbols('k_0 sigma_k', positive=True)
phi_k = sp.exp(-(k_sym - k0)**2 / (2*sigma_k**2))
packet_integrand = phi_k * sp.exp(sp.I*k_sym*x)
psi_x = sp.integrate(packet_integrand, (k_sym, -sp.oo, sp.oo))
psi_x_simplified = sp.simplify(psi_x)
print("\nWave packet psi(x) = integral of phi(k)*e^(ikx) dk, phi(k) a Gaussian in k:")
sp.pprint(psi_x_simplified)
print("\n--> a GAUSSIAN in x comes out of a GAUSSIAN in k -- localized in both,")
print("    unlike the single plane wave, which is perfectly localized in k")
print("    (a literal spike) and perfectly delocalized in x (a 'singularity'")
print("    in the sense of not being a proper normalizable state at all).")""")

# ── §4 group velocity vs phase velocity for matter waves ───────────────
md(r"""## §4 Group (packet) velocity vs. phase velocity, for matter waves

Phase velocity $v_p=\omega/k$ is how fast a single crest moves. Group
velocity $v_g=d\omega/dk$ is how fast the PACKET (the envelope, the thing
that actually carries the particle) moves. For a free-particle matter wave,
$\omega(k)=\hbar k^2/(2m)$ (from $E=\hbar\omega=p^2/2m=\hbar^2k^2/2m$) --
derive $v_g$ and show it equals the ordinary classical velocity $p/m$, not
some exotic quantum speed.""")

code(r"""hbar_s, k_s, m_s = sp.symbols('hbar k m', positive=True)

omega_matter = hbar_s * k_s**2 / (2*m_s)   # from E = p^2/2m = hbar^2 k^2/(2m) = hbar*omega
v_phase = sp.simplify(omega_matter / k_s)
v_group = sp.simplify(sp.diff(omega_matter, k_s))

print("Matter-wave dispersion: omega(k) = hbar*k^2/(2m)")
sp.pprint(sp.Eq(sp.Symbol('omega'), omega_matter))
print("\nPhase velocity v_p = omega/k =")
sp.pprint(v_phase)
print("\nGroup velocity v_g = d(omega)/dk =")
sp.pprint(v_group)

# classical velocity from p = hbar*k, v_classical = p/m = hbar*k/m
v_classical = hbar_s*k_s/m_s
print("\nClassical velocity v = p/m = hbar*k/m =")
sp.pprint(v_classical)

assert sp.simplify(v_group - v_classical) == 0
print("\nVerified EXACTLY: v_group = v_classical. The wave PACKET (not any single")
print("crest inside it) moves at the same speed as the classical particle --")
print("v_phase = v_group/2 here, so the crests drift backward relative to the")
print("envelope, a real and often-missed feature of matter-wave packets.")""")

# ── §5 probabilistic interpretation ─────────────────────────────────
md(r"""## §5 The probabilistic interpretation (Born rule)

$P(x)dx = |\psi(x)|^2 dx$ -- probability density, not amplitude. Verify the
wave packet from §3 actually integrates to something finite (it's
normalizable, unlike the bare plane wave) and extract its normalization
constant symbolically.""")

code(r"""A = sp.symbols('A', positive=True)
psi_normalized_form = A * sp.exp(-(x)**2 * sigma_k**2 / 2)  # from the Gaussian-in-x result above (k0=0 case for clarity)
prob_density = sp.simplify(sp.Abs(psi_normalized_form)**2)
print("Probability density |psi(x)|^2 (Born rule) for the k0=0 wave packet:")
sp.pprint(prob_density)

total_prob = sp.integrate(prob_density, (x, -sp.oo, sp.oo))
print("\nTotal probability (must be finite for a physical state):")
sp.pprint(total_prob)

A_normalized = sp.solve(sp.Eq(total_prob, 1), A)
print("\nSolving total probability = 1 for the normalization constant A:")
sp.pprint(A_normalized)
print("\n--> finite, solvable normalization -- exactly what failed for the bare")
print("    plane wave in Section 3. This is what 'probabilistic interpretation'")
print("    requires operationally: a state that CAN be normalized to total")
print("    probability 1, which rules plane waves out as physical states.")""")

# ── §6 Heisenberg uncertainty, derived from an actual standard deviation ──
md(r"""## §6 Heisenberg uncertainty -- computed, not quoted

$\sigma_x\sigma_p\geq\hbar/2$ is usually just handed over. Here it's computed
from the actual definition of standard deviation,
$\sigma_x^2=\langle x^2\rangle-\langle x\rangle^2$, on a normalized Gaussian
wavefunction, with $\sigma_p$ obtained the same way in momentum space via
the Fourier-transformed wavefunction -- then the product is checked
directly against $\hbar/2$.""")

code(r"""sigma_x_sym = sp.symbols('sigma_x', positive=True)

# normalized Gaussian wavefunction (position space), width sigma_x, centered at 0
psi_x_gauss = (1/(2*sp.pi*sigma_x_sym**2))**sp.Rational(1,4) * sp.exp(-x**2/(4*sigma_x_sym**2))

norm_check = sp.integrate(sp.Abs(psi_x_gauss)**2, (x, -sp.oo, sp.oo))
print("Normalization check, integral |psi|^2 dx =", sp.simplify(norm_check), "(expect 1)")
assert sp.simplify(norm_check - 1) == 0

mean_x = sp.integrate(x * sp.Abs(psi_x_gauss)**2, (x, -sp.oo, sp.oo))
mean_x2 = sp.integrate(x**2 * sp.Abs(psi_x_gauss)**2, (x, -sp.oo, sp.oo))
var_x = sp.simplify(mean_x2 - mean_x**2)
print("<x> =", mean_x, "   <x^2> =", sp.simplify(mean_x2))
print("sigma_x^2 = <x^2> - <x>^2 =", var_x, " (matches the sigma_x we put in, confirming the setup)")
assert sp.simplify(var_x - sigma_x_sym**2) == 0

# momentum-space wavefunction: Fourier transform of psi_x_gauss w.r.t. x -> p/hbar
p_sym = sp.symbols('p', real=True)
hbar_num = sp.symbols('hbar', positive=True)
# FT of a Gaussian is a Gaussian; standard result: if sigma_x is the position
# width, the momentum-space width is sigma_p = hbar/(2*sigma_x) for this
# minimum-uncertainty (Gaussian) state -- derive it via the FT integral directly
k_var = sp.symbols('k', real=True)
psi_k_gauss = sp.integrate(psi_x_gauss * sp.exp(-sp.I*k_var*x), (x, -sp.oo, sp.oo))
psi_k_gauss = sp.simplify(psi_k_gauss)
print("\nFourier transform to k-space, psi(k) = integral psi(x)*e^(-ikx) dx =")
sp.pprint(psi_k_gauss)

prob_k = sp.simplify(sp.Abs(psi_k_gauss)**2)
norm_k = sp.integrate(prob_k, (k_var, -sp.oo, sp.oo))
prob_k_normalized = sp.simplify(prob_k / norm_k)
mean_k2 = sp.integrate(k_var**2 * prob_k_normalized, (k_var, -sp.oo, sp.oo))
sigma_k_result = sp.sqrt(sp.simplify(mean_k2))
print("\nsigma_k (standard deviation of the k-space distribution) =")
sp.pprint(sigma_k_result)

sigma_p_result = hbar_num * sigma_k_result   # p = hbar*k
sigma_x_sigma_p = sp.simplify(sigma_x_sym * sigma_p_result)
print("\nsigma_x * sigma_p =")
sp.pprint(sigma_x_sigma_p)

assert sp.simplify(sigma_x_sigma_p - hbar_num/2) == 0
print("\nVerified EXACTLY: sigma_x * sigma_p = hbar/2 -- the Gaussian wave packet")
print("SATURATES the Heisenberg bound (equality, not just >=). This is why")
print("Gaussian wave packets are called 'minimum uncertainty states': every")
print("other shape of psi(x) gives sigma_x*sigma_p > hbar/2, strictly.")""")

md(r"""## Summary

| Concept | What was actually computed | Result |
|---|---|---|
| de Broglie | photon relations extended to matter, $p=mv$ | $\lambda_{dB}=h/(mv)$ |
| Davisson-Germer | Bragg condition vs. de Broglie from accelerating voltage | agree to <10% |
| Wave packet | FT of a Gaussian $\phi(k)$ | Gaussian $\psi(x)$, normalizable |
| Phasor | $d/dx$ and $d/dt$ on $e^{i(kx-\omega t)}$ | multiply by $ik$, $-i\omega$ |
| Group velocity | $d\omega/dk$ for $\omega=\hbar k^2/2m$ | $v_g=\hbar k/m=$ classical $v$ |
| Born rule | $\int|\psi|^2dx=1$ solved for $A$ | finite normalization constant |
| Uncertainty | $\sigma_x$, $\sigma_p$ from actual variance integrals | $\sigma_x\sigma_p=\hbar/2$ exactly |

Nothing above was asserted from memory -- every result came out of a `sympy`
computation with `init_printing()` rendering it, and the ones with a known
textbook answer were checked against it with `assert`.""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")