"""Extend notebooks/matter_waves_chapter5_sympy_torch.ipynb (covers de Broglie
wavelength, phase velocity, both Heisenberg uncertainty forms) with:
  - Davisson-Germer: the electron-diffraction condition, checked against the
    real 1927 experimental numbers
  - an actual wave PACKET construction (2-wave superposition), deriving group
    velocity vs phase velocity from a trig identity instead of asserting it
  - probabilistic interpretation (Born rule) connected to Delta_x as a
    standard deviation, reusing dgs.robertson_uncertainty (not duplicating it)
NOTE: no triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "matter_waves_chapter5_sympy_torch.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

dg_md = md(r"""## 5. Davisson-Germer: detecting electron waves

de Broglie's $\lambda=h/p$ is a claim, not just a formula, and Davisson-Germer
(1927) is the experiment that made it real: fire electrons at a nickel
crystal (atomic spacing $d$) and look for a DIFFRACTION peak, exactly like
X-rays off a crystal lattice (Bragg condition $d\sin\theta = n\lambda$). If
electrons are waves, the peak angle is fixed by their de Broglie wavelength;
if they're just particles, there's no reason for one.""")

dg_code = code(r"""d_lattice, theta_s, n_order, lam = sp.symbols('d theta n lambda', positive=True)
bragg = sp.Eq(d_lattice * sp.sin(theta_s), n_order * lam)
print("Bragg/Davisson-Germer diffraction condition:")
sp.pprint(bragg)

# real 1927 experiment numbers: 54 eV electrons on nickel, d = 2.15 Angstrom,
# observed peak at theta = 50 degrees, first order (n=1)
E_eV = 54.0
d_val = 2.15e-10  # m

E_joules = E_eV * 1.602176634e-19
p_val = np.sqrt(2 * m_electron * E_joules)   # nonrelativistic, 54 eV is very nonrelativistic
lambda_debroglie = h_val / p_val
print(f"\n54 eV electron: p = {p_val:.4e} kg m/s, lambda_debroglie = {lambda_debroglie*1e10:.3f} Angstrom")

theta_predicted = sp.asin(1 * lambda_debroglie / d_val)
theta_predicted_deg = float(sp.deg(theta_predicted))
print(f"predicted diffraction angle (n=1): theta = {theta_predicted_deg:.1f} degrees")
print(f"Davisson & Germer's actual measured peak: 50 degrees")
print(f"agreement within {abs(theta_predicted_deg-50):.1f} degrees "
      f"-- de Broglie's formula predicts a REAL, measured diffraction angle,")
print("not just a dimensional-analysis curiosity.")""")

wp_md = md(r"""## 6. A real wave packet: group velocity vs phase velocity, derived not asserted

Section 2 stated that phase velocity isn't the packet's actual speed. Here's
the derivation: superpose just TWO waves of slightly different $k,\omega$
(the simplest possible wave packet) and watch an envelope and a carrier
appear, moving at two DIFFERENT speeds, via nothing but a trig identity.

$$\cos(k_1x-\omega_1t)+\cos(k_2x-\omega_2t) = 2\cos\!\left(\frac{\Delta k}{2}x-\frac{\Delta\omega}{2}t\right)\cos(\bar k x-\bar\omega t)$$

The first factor (slowly varying) is the **envelope**, moving at
$v_g=\Delta\omega/\Delta k\to d\omega/dk$; the second (fast oscillation) is
the **carrier**, moving at $v_p=\bar\omega/\bar k$.""")

wp_code = code(r"""xs, ts = sp.symbols('x t', real=True)
k1, k2, w1, w2 = sp.symbols('k_1 k_2 omega_1 omega_2', real=True)

wave_sum = sp.cos(k1*xs - w1*ts) + sp.cos(k2*xs - w2*ts)

# sympy's sum-to-product identity, applied directly (not hand-waved)
wave_sum_rewritten = sp.simplify(sp.expand_trig(wave_sum) - wave_sum)  # sanity: expand shouldn't change it
factored = sp.trigsimp(wave_sum.rewrite(sp.cos))

k_avg, w_avg, dk, dw = sp.symbols('kbar omegabar Delta_k Delta_omega', real=True)
# construct the claimed product form and verify it EQUALS the original sum
# (rather than asking sympy to discover the identity, which it may not do
# unprompted for a sum of two generic cosines -- verify the claim directly)
claimed_form = 2*sp.cos(dk/2*xs - dw/2*ts) * sp.cos(k_avg*xs - w_avg*ts)
substituted_claim = claimed_form.subs({
    dk: k1 - k2, dw: w1 - w2, k_avg: (k1+k2)/2, w_avg: (w1+w2)/2
})
difference = sp.simplify(sp.expand_trig(substituted_claim) - sp.expand_trig(wave_sum))
print("cos(k1 x-w1 t) + cos(k2 x-w2 t)  vs.  2cos(Dk/2 x-Dw/2 t)*cos(kbar x-wbar t):")
print("difference simplifies to:", difference)
assert difference == 0
print("\nVerified: the sum of two waves EXACTLY equals an envelope times a carrier.")
print("Envelope moves at (Dw/2)/(Dk/2) = Delta_omega/Delta_k -> d(omega)/d(k) as the")
print("two waves merge continuously -- THIS is the group velocity, derived from the")
print("wave packet's own structure, not asserted as 'the speed that matches the particle'.")
print("Carrier moves at wbar/kbar = the phase velocity from Section 2 -- a DIFFERENT number.")""")

prob_md = md(r"""## 7. Probabilistic interpretation, and $\Delta x$ as a standard deviation

Born's rule: $P(x)\,dx=|\psi(x)|^2\,dx$ is a genuine probability density, so
$\Delta x$ in the uncertainty relation isn't a vague "spread" -- it's the
actual standard deviation $\sigma_x=\sqrt{\langle x^2\rangle-\langle x\rangle^2}$
of that distribution. `dgs.robertson_uncertainty` already computes
$\sigma_x\sigma_p$ numerically (via FFT) for an arbitrary $\psi(x)$ --
reused here rather than re-derived, to show a Gaussian wave packet saturates
the bound exactly at $\hbar/2$ (the minimum-uncertainty state).""")

prob_code = code(r"""import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path.cwd().parent))
from dgs.robertson_uncertainty import position_momentum_uncertainty

x_grid = np.linspace(-20, 20, 4096)
sigma0 = 1.0
psi_gaussian = np.exp(-x_grid**2 / (4*sigma0**2))   # a real Gaussian wave packet

prod, bound = position_momentum_uncertainty(x_grid, psi_gaussian, hbar=1.0)
print(f"Gaussian wave packet: sigma_x * sigma_p = {prod:.6f}")
print(f"Heisenberg minimum hbar/2 = {bound:.6f}")
print(f"ratio = {prod/bound:.6f}  (1.0 means the Gaussian SATURATES the bound exactly --")
print("the minimum-uncertainty wave packet, same physics as Section 6's envelope,")
print("now with Delta_x properly identified as the probability distribution's own")
print("standard deviation, not just 'some width'.)")""")

def find_index(snippet):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and snippet in c.source:
            return i
    raise ValueError(f"not found: {snippet!r}")

idx_summary = find_index("## Summary")
cells.insert(idx_summary, prob_code)
cells.insert(idx_summary, prob_md)
cells.insert(idx_summary, wp_code)
cells.insert(idx_summary, wp_md)
cells.insert(idx_summary, dg_code)
cells.insert(idx_summary, dg_md)

nb.cells = cells
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH} with {len(cells)} cells")
