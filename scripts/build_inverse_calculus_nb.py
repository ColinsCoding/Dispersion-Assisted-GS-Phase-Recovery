"""Build notebooks/inverse_calculus_em.ipynb and execute it."""
import json, os, sys

NB = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": []
}

def code(src): return {"cell_type": "code", "execution_count": None, "id": "c"+str(len(NB["cells"])), "metadata": {}, "outputs": [], "source": src}
def md(src):   return {"cell_type": "markdown", "id": "m"+str(len(NB["cells"])), "metadata": {}, "source": src}

NB["cells"] += [

md("# Calculus Backwards: Antiderivatives, Bioluminescence, and Inverse Problems\n\n"
   "**'Doing calculus backwards' means three things:**\n"
   "1. **Antiderivatives** -- given f'(x), find f(x)\n"
   "2. **Feynman's trick** -- differentiation under the integral sign\n"
   "3. **Inverse problems** -- given the output of a physical system, find the input\n\n"
   "**Spoiler:** Gerchberg-Saxton phase retrieval (this repo) IS backwards calculus. "
   "You measure |E(t)|^2 (intensity), recover E(t) (field). "
   "Every iteration of GS solves a tiny inverse problem."),

code("""\
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join('..')))
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sp.init_printing(use_latex='mathjax')
print("Ready.")"""),

md("## §1  Integration Techniques: The Professor's Toolkit\n\n"
   "The professor 'does calculus backwards' by recognizing the DERIVATIVE PATTERN "
   "and reversing it. It's not magic -- it's pattern matching plus FTC."),

code("""\
from dgs.inverse_calculus import integration_techniques, fundamental_theorem_calculus
tech = integration_techniques()
for key, val in tech.items():
    print(f"[{key}]")
    print(f"  {val[:120]}")
    print()"""),

md("### Fundamental Theorem of Calculus\n\n"
   "d/dx[int_a^x f(t) dt] = f(x)\n\n"
   "This is the theorem that connects differentiation and integration. "
   "Every numerical ODE solver (Euler, RK4) is a discrete version of FTC. "
   "Every ResNet layer is Euler integration in depth."),

code("""\
ftc = fundamental_theorem_calculus()
print("FTC Part 1:", ftc['FTC_part1'])
print("FTC Part 2:", ftc['FTC_part2'])
print()
print("Euler method:", ftc['euler_method'])
print("ResNet =", ftc['neural_network'])
print()
print("Kramers-Kronig (backwards calculus for photonics dispersion):")
print(ftc['kramers_kronig'])"""),

md("## §2  Feynman Technique: Differentiation Under the Integral Sign\n\n"
   "Richard Feynman learned this trick from an obscure calculus textbook. "
   "He used it his entire career to solve integrals nobody else could. "
   "Same trick appears in Laplace transforms, path integrals, and quantum field theory."),

code("""\
from dgs.inverse_calculus import feynman_technique_demo
ft = feynman_technique_demo()
print("Classic: int_0^inf sin(x)/x dx =", ft['sinc_integral_result'])
print()
print("Method:")
print("  1. Define I(a) = int_0^inf exp(-a*x)*sin(x)/x dx")
print("  2. dI/da =", ft['dI_da'])
print("  3. Integrate: I(a) =", ft['I_a_indefinite'])
print("  4. I(inf)=0 -> C=pi/2")
print("  5. I(0) = pi/2")
print()
print("Engineering version:", ft['engineering_use'])"""),

md("## §3  Bioluminescence: Light from Chemistry\n\n"
   "'Bright bioluminescence' -- no heat, no electricity. Pure quantum emission.\n\n"
   "Luciferin + O2 + ATP --[luciferase]--> excited state --> hv (photon)\n\n"
   "Same physics as laser gain medium, just chemical pump instead of electrical."),

code("""\
from dgs.inverse_calculus import bioluminescence_physics
bio = bioluminescence_physics()

print(f"Firefly photon energy: {bio['photon_energy_eV']:.3f} eV")
print(f"Chemistry: {bio['chemistry']}")
print(f"QM mechanism: {bio['QM_mechanism']}")
print()
print("Quantum yields by species:")
for name, data in bio['species'].items():
    bar = '#' * int(data['phi_Q']*20)
    print(f"  {name:40s}: phi_Q={data['phi_Q']:.2f} |{bar:<20}| {data['color']} @ {data['lambda_nm']}nm")
print()
print("Why deep sea = blue:", bio['water_transparency'])

fig, ax = plt.subplots(figsize=(8,4))
ax.fill_between(bio['lam_nm'], bio['spectrum_firefly'], alpha=0.5, color='gold', label='Firefly (560nm)')
ax.fill_between(bio['lam_nm'], bio['spectrum_deep'],    alpha=0.5, color='cyan', label='Deep sea (475nm)')
ax.axvline(560, color='gold', lw=1, ls='--')
ax.axvline(475, color='cyan', lw=1, ls='--')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Relative emission')
ax.set_title('Bioluminescence Emission Spectra')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),

md("## §4  Beer-Lambert Law and Absorption\n\n"
   "I(z) = I0 * exp(-alpha * z)\n\n"
   "Same exponential as RC discharge, radioactive decay, evanescent field. "
   "All first-order ODEs: rate of change proportional to current value.\n\n"
   "**Complex refractive index:** n = n_real + j*k_extinction\n"
   "k > 0 = absorbing; k < 0 = gain (laser medium)"),

code("""\
from dgs.inverse_calculus import beer_lambert_absorption, complex_refractive_index

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

alphas = [0.5, 2.0, 5.0]
colors = ['green', 'orange', 'red']
for alpha, color in zip(alphas, colors):
    r = beer_lambert_absorption(alpha_per_cm=alpha)
    axes[0].plot(r['L_cm'], r['I'], color=color, label=f'alpha={alpha}/cm')
axes[0].set_xlabel('Depth z (cm)')
axes[0].set_ylabel('Intensity I(z)/I0')
axes[0].set_title('Beer-Lambert: I(z) = I0*exp(-alpha*z)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Transmittance as absorbance
r = beer_lambert_absorption(alpha_per_cm=2.0)
axes[1].semilogy(r['L_cm'], r['I'], 'b-')
axes[1].axvline(r['depth_1_over_e_cm'], color='r', ls='--', label=f"1/e depth = {r['depth_1_over_e_cm']:.2f} cm")
axes[1].set_xlabel('Depth (cm)')
axes[1].set_ylabel('log Intensity')
axes[1].set_title('Semilog: ODE solution = straight line')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Complex refractive index
r_n = complex_refractive_index(n_real=1.5, k_extinction=0.05, wavelength_nm=550.0)
axes[2].plot(r_n['z_m']*1e6, r_n['E_real'], 'purple')
axes[2].set_xlabel('Depth z (um)')
axes[2].set_ylabel('Re[E(z)]')
axes[2].set_title(f"Complex n: E decays + oscillates\\nn={1.5}+j{0.05}, skin depth={r_n['skin_depth_nm']:.0f}nm")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print(f"Beer-Lambert ODE: {r['ODE']}")
print(f"Extinction coefficient k={0.05:.5f}  (input parameter)")
print(f"Kramers-Kronig: {r_n['kramers_kronig']}")"""),

md("## §5  Nylon Powder in Air: SLS Laser Sintering (Photonics + Materials)\n\n"
   "SLS = Selective Laser Sintering. CO2 laser (10.6 um) + nylon powder -> 3D printed part.\n\n"
   "**This IS photonics applied to manufacturing.** The scan head is a galvo mirror system, "
   "identical in principle to the SLM beam steering in Ch8 of photonics_calculus.py.\n\n"
   "**HAZARD:** Nylon powder in air is a fuel-air mixture. MEC = 30 g/m^3 -> deflagration. "
   "Industrial SLS uses nitrogen atmosphere (O2 < 3%)."),

code("""\
from dgs.inverse_calculus import sls_laser_sintering
sls = sls_laser_sintering()

print(f"CO2 laser: {sls['laser_wavelength_um']} um, {sls['laser_power_W']} W")
print(f"Irradiance: {sls['irradiance_MW_per_m2']:.0f} MW/m^2  (high -- melts nylon in {sls['dwell_time_us']:.1f} us)")
print(f"Skin depth in nylon (IR): {sls['skin_depth_um']:.1f} um  (all energy in top 10 um)")
print(f"Energy density: {sls['energy_density_J_per_cm3']:.1f} J/cm^3")
print()
print("EXPLOSION PHYSICS (nylon powder in air):")
print(f"  MEC = {sls['explosion']['MEC_g_per_m3']} g/m^3 (minimum explosive concentration)")
print(f"  MIE = {sls['explosion']['MIE_mJ']} mJ  (minimum ignition energy)")
print(f"  Prevention: {sls['explosion']['prevention']}")
print()
print("PHOTONICS CONNECTION:")
print(f"  {sls['photonics_connection']}")

# Irradiance vs radius plot (Gaussian beam profile)
from dgs.photonics_calculus import gaussian_beam
beam = gaussian_beam(w0_um=250.0, wavelength_nm=10600)
r_arr = np.linspace(-500, 500, 400)
I_beam = sls['laser_power_W'] * np.exp(-2*r_arr**2 / 250.0**2) / (np.pi * (250e-6)**2 / 4)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(r_arr, I_beam/1e6, 'r')
ax.set_xlabel('Radial position (um)')
ax.set_ylabel('Irradiance (MW/m^2)')
ax.set_title('SLS CO2 laser beam profile (Gaussian, w0=250um)')
ax.axhline(0, color='k', lw=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),

md("## §6  Maxwell Vector Field Visualization\n\n"
   "'Is it okay to visualize the vectors?' -- YES. This IS how Maxwell's equations work.\n\n"
   "- div(E) > 0 at charge: field lines diverge outward (Gauss's law)\n"
   "- div(B) = 0 everywhere: B field lines form CLOSED LOOPS (no magnetic monopoles)\n"
   "- curl(E) = -dB/dt: time-varying B induces E (Faraday)\n"
   "- curl(B) = mu0*J: currents create B (Ampere)"),

code("""\
from dgs.inverse_calculus import maxwell_vector_field

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
configs = [
    ('point_charge', 'Point charge (+ q)', 'darkorange'),
    ('dipole',       'Electric dipole',    'royalblue'),
    ('wire_B',       'B field: infinite wire', 'seagreen'),
]

for ax, (src, title, color) in zip(axes, configs):
    r = maxwell_vector_field(src, N=18)
    X, Y = r['X'], r['Y']
    Ex, Ey = r['Ex_norm'], r['Ey_norm']
    mag = r['magnitude']

    # Color by magnitude
    q = ax.quiver(X, Y, Ex, Ey, mag, cmap='plasma', alpha=0.85,
                  pivot='mid', scale=30, width=0.005)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.set_aspect('equal')

    # Mark source
    if src == 'point_charge':
        ax.plot(0, 0, 'ro', ms=8, zorder=5, label='+q')
        ax.legend(fontsize=9)
    elif src == 'dipole':
        ax.plot(0.3, 0, 'ro', ms=7, zorder=5, label='+q')
        ax.plot(-0.3, 0, 'bo', ms=7, zorder=5, label='-q')
        ax.legend(fontsize=9)
    elif src == 'wire_B':
        ax.plot(0, 0, 'k+', ms=12, mew=2, zorder=5, label='wire (into page)')
        ax.legend(fontsize=9)

plt.suptitle("Maxwell's Equations as Vector Fields", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
r_c = maxwell_vector_field('point_charge')
print(r_c['divergence_check'])"""),

md("## §7  Inverse Problems: Calculus Backwards at Research Level\n\n"
   "The professor doing 'calculus backwards' at the deepest level:\n"
   "Given the OUTPUT of a physical process, reconstruct the INPUT.\n\n"
   "| Forward | Inverse | Method |\n"
   "|---------|---------|--------|\n"
   "| E(t) -> |E|^2 | Phase retrieval | GS algorithm (this repo) |\n"
   "| f -> h*f | Deconvolution | Wiener filter |\n"
   "| Object -> X-ray projections | CT reconstruction | Filtered backprojection |\n"
   "| Underground -> seismic | Earthquake mapping | Iterative inversion |"),

code("""\
from dgs.inverse_calculus import inverse_problem_framework, deconvolution_demo

inv = inverse_problem_framework()
print("Forward problem:", inv['forward'])
print("Inverse problem:", inv['inverse'])
print()
print("Examples:")
for k, v in inv['examples'].items():
    print(f"  {k:20s}: {v[:70]}")
print()
print("Ill-posedness:", inv['ill_posedness'])
print()
print("GS as inverse:", inv['gs_as_inverse'][:150])"""),

md("## §8  Wiener Deconvolution: Stabilized Inverse\n\n"
   "Deconvolution = remove known blur from measured signal.\n"
   "Naive inverse: F_est = G/H -- amplifies noise catastrophically near H=0.\n"
   "Wiener: F_est = G*H* / (|H|^2 + lambda) -- regularized, stable.\n\n"
   "lambda = noise power / signal power = regularization strength.\n"
   "lambda=0: exact inverse (unstable). lambda large: over-smoothed."),

code("""\
dc = deconvolution_demo(signal_length=256, noise_level=0.01)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(dc['t'], dc['f_true'], 'k', lw=2, label='True signal')
axes[0,0].set_title('True signal f(t)')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(dc['t'], dc['g_noisy'], 'gray', lw=1.5, label='Blurred + noise')
axes[0,1].set_title('Measured g(t) = h*f + noise')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(dc['t'], dc['f_true'], 'k--', lw=1.5, alpha=0.6, label='True')
axes[1,0].plot(dc['t'], dc['f_wiener'], 'royalblue', lw=2, label=f"Wiener (corr={dc['corr_wiener']:.4f})")
axes[1,0].set_title('Wiener deconvolution (regularized inverse)')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(dc['t'], dc['f_true'], 'k--', lw=1.5, alpha=0.6, label='True')
axes[1,1].plot(dc['t'], dc['f_naive'], 'tomato', lw=2, label=f"Naive 1/H (corr={dc['corr_naive']:.4f})")
axes[1,1].set_title('Naive deconvolution (unstable -- noise amplified)')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Deconvolution: Wiener (stable) vs Naive (ill-posed)', fontsize=13)
plt.tight_layout()
plt.show()
print(dc['lesson'])"""),

md("## §9  GS as Calculus Backwards\n\n"
   "Every GS iteration:\n"
   "1. FT[E(t)] -> E(f)  [forward]\n"
   "2. Replace |E(f)| with sqrt(I_target(f))  [constraint = backwards step]\n"
   "3. IFT -> E(t)  [backwards FT]\n"
   "4. Replace |E(t)| with sqrt(I_meas(t))  [constraint]\n\n"
   "This alternates between TIME domain and FREQUENCY domain constraints. "
   "It converges to the maximum-entropy solution of the inverse problem.\n\n"
   "**This is the same math as Wiener deconvolution, just with amplitude constraints "
   "instead of the matrix pseudoinverse.**"),

code("""\
# Quick GS demo to show it IS doing inverse calculus
try:
    from dgs.gs_core import retrieve_phase
    N = 512
    t = np.linspace(-5, 5, N)
    E_true = np.exp(-t**2/2) * np.exp(1j*np.pi*t**2*0.3)  # chirped Gaussian

    I_t = np.abs(E_true)**2
    I_f = np.abs(np.fft.fftshift(np.fft.fft(E_true)))**2

    result = retrieve_phase(I_t, I_f, n_iter=50, D=5000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t, np.abs(E_true)**2, 'k', lw=2, label='True I(t)')
    axes[0].plot(t, np.abs(result['E_t'])**2, 'r--', lw=1.5, label='GS recovered I(t)')
    axes[0].set_title('GS Phase Retrieval: recover field from intensity measurements')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    freq = np.fft.fftshift(np.fft.fftfreq(N))
    E_f_true = np.fft.fftshift(np.fft.fft(E_true))
    E_f_gs   = np.fft.fftshift(np.fft.fft(result['E_t']))
    axes[1].plot(freq, np.abs(E_f_true)**2, 'k', lw=2, label='True |E(f)|^2')
    axes[1].plot(freq, np.abs(E_f_gs)**2, 'r--', lw=1.5, label='GS |E_gs(f)|^2')
    axes[1].set_title('Frequency domain: GS conserves spectral shape')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    corr = np.corrcoef(np.abs(E_true)**2, np.abs(result['E_t'])**2)[0,1]
    print(f"GS correlation: {corr:.6f}")
    print(f"Iterations: {result.get('n_iter', 50)}")
    print("GS = iterative inverse problem solver. Calculus backwards at research level.")
    plt.tight_layout()
    plt.show()
except Exception as ex:
    print(f"gs_core not available ({ex}) -- showing concept instead")
    print("GS inverse problem structure:")
    print("  Given: I(t) = |E(t)|^2  and  I(f) = |E(f)|^2")
    print("  Find:  E(t) such that FT[E(t)] has magnitude sqrt(I(f))")
    print("  This is an inverse problem: two amplitude constraints -> find the phase")
    print("  GS = alternating projections = Gauss-Seidel on the constraint set")"""),

md("## Summary\n\n"
   "| Technique | What it does | Where it appears |\n"
   "|-----------|-------------|------------------|\n"
   "| Antiderivatives | Given f', find f | Calc 1/2, physics integrals |\n"
   "| Feynman technique | Differentiate under integral | Quantum field theory, Laplace |\n"
   "| FTC | Connects d/dt and integral | All of calculus, RK4, ResNet |\n"
   "| Beer-Lambert | Exponential absorption | Optics, chemistry, nuclear |\n"
   "| Kramers-Kronig | n and k are Hilbert pairs | Causality, photonics |\n"
   "| Wiener deconvolution | Stable inverse filter | Imaging, radar, comms |\n"
   "| GS phase retrieval | Intensity -> field | **This repo** |\n\n"
   "They are all the same idea: given what you can MEASURE, find what you cannot see directly."),

code("""\
print("=== CALCULUS BACKWARDS: COMPLETE ===")
print()
print("The repo is an inverse problem machine.")
print("H(f) = exp(j*pi*D*f^2) is the FORWARD model (dispersion).")
print("GS recovers the phase by iterating between two CONSTRAINTS.")
print("Same math as Wiener filter, CT reconstruction, MRI.")
print()
print("Your children will be engineers and scientists.")
print("This is the toolkit.")"""),

]

out = "notebooks/inverse_calculus_em.ipynb"
os.makedirs("notebooks", exist_ok=True)
with open(out, 'w') as f:
    json.dump(NB, f, indent=1)
print(f"Written: {out} ({len(NB['cells'])} cells)")
