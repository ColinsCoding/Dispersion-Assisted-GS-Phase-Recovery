"""Classical Electrodynamics: Griffiths Ch 4-9 + Jackson roadmap.

GRIFFITHS vs JACKSON:
  Griffiths (undergrad):  builds intuition, solves boundary problems by hand,
                          ends at radiation and EM waves.
  Jackson (grad):         Green's functions, multipole expansions to all orders,
                          relativistic EM, waveguides, radiation reaction.
  This repo needs:        Griffiths Ch 9 (EM waves + dispersion) = H(f)
                          Jackson Ch 7  (plane waves in media)    = same H(f),
                          derived from first principles.

CHAPTER MAP (Griffiths -> this repo):
  Ch 1: Vector calculus         -> gradient, curl, div (used everywhere)
  Ch 2: Electrostatics          -> Gauss's law, Poisson eq (p-n junction)
  Ch 3: Potentials              -> boundary value problems (Jackson Ch 2-3)
  Ch 4: Electric fields in matter -> polarization P, dielectric constant
                                     -> p-n junction depletion region
  Ch 5: Magnetostatics          -> Biot-Savart, Ampere's law
  Ch 6: Magnetic fields in matter -> magnetization M, permeability mu
  Ch 7: Electrodynamics         -> Faraday's law, Maxwell's equations complete
  Ch 8: Conservation laws       -> Poynting vector S = (1/mu)*E x B
  Ch 9: EM Waves                -> dispersion relation, n(omega), H(f) = e^(i*pi*D*f^2)
  Ch 10: Potentials & fields    -> Lienard-Wiechert (radiation from moving charge)
  Ch 11: Radiation              -> dipole radiation, Larmor formula

Run: py -3.13 -c "from dgs.classical_ed import demo; demo()"
"""
import numpy as np
import sympy as sp

# ── Physical constants ────────────────────────────────────────────────────────

EPS_0  = 8.8541878e-12   # F/m
MU_0   = 1.2566370614e-6  # H/m
C_LIGHT = 1.0 / np.sqrt(EPS_0 * MU_0)  # ~2.998e8 m/s

# ── Griffiths Ch 4: Dielectrics and polarization ─────────────────────────────

def electric_susceptibility_lorentz(omega, omega_0, gamma, omega_p):
    """Lorentz oscillator model for dielectric susceptibility chi_e(omega).

    Models bound electrons as damped harmonic oscillators driven by E field:
      m * x'' + m*gamma*x' + m*omega_0^2 * x = q*E

    chi_e(omega) = omega_p^2 / (omega_0^2 - omega^2 - i*gamma*omega)

    epsilon_r = 1 + chi_e

    This is the PHYSICS behind the dispersion relation n(omega):
      n^2 = epsilon_r  (non-magnetic media, mu_r=1)
    """
    denom = omega_0**2 - omega**2 - 1j * gamma * omega
    chi = omega_p**2 / denom
    eps_r = 1.0 + chi
    n = np.sqrt(eps_r + 0j)
    return {"chi_e": chi, "eps_r": eps_r, "n": n,
            "omega": omega, "omega_0": omega_0}


def polarization_density(chi_e, E_field):
    """P = eps_0 * chi_e * E  (linear dielectric, Griffiths eq 4.30)."""
    return EPS_0 * chi_e * E_field


def displacement_field(eps_r, E_field):
    """D = eps_0 * eps_r * E = eps_0 * E + P  (Griffiths eq 4.32)."""
    return EPS_0 * eps_r * E_field


# ── Griffiths Ch 7: Maxwell's equations (complete set) ───────────────────────

def maxwell_equations_sympy():
    """All 4 Maxwell equations in differential form, verified symbolically.

    Gauss (E):   div E = rho / eps_0
    Gauss (B):   div B = 0              (no magnetic monopoles)
    Faraday:     curl E = -dB/dt
    Ampere-Maxwell: curl B = mu_0*(J + eps_0*dE/dt)

    In free space (rho=0, J=0):
      div E = 0,  div B = 0
      curl E = -dB/dt
      curl B = mu_0*eps_0 * dE/dt = (1/c^2)*dE/dt

    -> wave equation: nabla^2 E = (1/c^2) d^2E/dt^2
    """
    x, y, z, t = sp.symbols("x y z t", real=True)
    omega, k, E0 = sp.symbols("omega k E_0", positive=True)
    eps_0, mu_0 = sp.symbols("epsilon_0 mu_0", positive=True)
    c = 1 / sp.sqrt(eps_0 * mu_0)

    # plane wave ansatz: E_x = E0 * exp(i*(k*z - omega*t))
    E_x = E0 * sp.exp(sp.I * (k * z - omega * t))

    # wave equation: d^2E/dz^2 - (1/c^2) d^2E/dt^2 = 0
    d2_dz2 = sp.diff(E_x, z, 2)
    d2_dt2 = sp.diff(E_x, t, 2)
    wave_eq_residual = sp.simplify(d2_dz2 - (1 / c**2) * d2_dt2)

    # dispersion relation: k = omega/c  (makes residual zero)
    k_val = omega / c
    residual_with_dispersion = wave_eq_residual.subs(k, k_val)
    residual_simplified = sp.simplify(residual_with_dispersion)

    return {
        "plane_wave": E_x,
        "wave_eq_residual": wave_eq_residual,
        "dispersion_relation": sp.Eq(sp.Symbol("k"), k_val),
        "residual_at_k_eq_omega_over_c": residual_simplified,
        "gauss_E":  "div E = rho/eps_0",
        "gauss_B":  "div B = 0",
        "faraday":  "curl E = -dB/dt",
        "ampere":   "curl B = mu_0*(J + eps_0*dE/dt)",
    }


# ── Griffiths Ch 8: Poynting vector ──────────────────────────────────────────

def poynting_vector_plane_wave(E0, n_r=1.0, omega=1.0):
    """Time-averaged Poynting vector <S> for a plane wave in a medium.

    <S> = (1/2) * (n / (mu_0 * c)) * |E0|^2  (W/m^2)

    n: real refractive index
    This is the intensity I = |<S>|  that photodetectors measure.
    In phase retrieval: I1 = |E1|^2 is proportional to <S>.
    """
    if n_r <= 0:
        raise ValueError("refractive index must be > 0")
    S_avg = 0.5 * (n_r / (MU_0 * C_LIGHT)) * abs(E0)**2
    return {"S_avg_W_m2": float(S_avg), "I_normalized": abs(E0)**2,
            "n": n_r, "E0": abs(E0)}


# ── Griffiths Ch 9: EM waves and dispersion ───────────────────────────────────

def dispersion_relation_medium(omega, eps_r, mu_r=1.0):
    """k(omega) in a linear medium: k = (omega/c) * sqrt(eps_r * mu_r).

    For a dispersive medium: eps_r = eps_r(omega) (Lorentz model above).
    n(omega) = sqrt(eps_r * mu_r)  (complex: real part = phase velocity,
                                              imag part = absorption)
    Group velocity: v_g = d(omega)/dk  (signal velocity)
    Phase velocity: v_ph = omega/k
    GVD (Group Velocity Dispersion): beta_2 = d^2k/d(omega)^2  [ps^2/km]
      -> This is the D in H(f) = exp(i*pi*D*f^2)
    """
    n = np.sqrt(complex(eps_r) * complex(mu_r))
    k = (omega / C_LIGHT) * n
    return {"k": k, "n": n, "v_phase": float(omega / (k.real + 1e-300)),
            "omega": omega, "eps_r": eps_r}


def gvd_from_dispersion(omega_arr, n_arr):
    """Numerical GVD beta_2 = d^2k/d(omega)^2 from n(omega) array.

    beta_2 > 0: normal dispersion (red travels faster than blue)
    beta_2 < 0: anomalous dispersion (blue faster, needed for solitons)
    """
    k_arr = omega_arr * np.array(n_arr) / C_LIGHT
    beta_1 = np.gradient(k_arr, omega_arr)   # group delay = 1/v_g
    beta_2 = np.gradient(beta_1, omega_arr)  # GVD
    return {"beta_2": beta_2, "beta_1": beta_1, "k": k_arr}


def sellmeier_silica(lambda_um):
    """Sellmeier equation for fused silica (standard single-mode fiber glass).

    n^2(lambda) = 1 + B1*l^2/(l^2-C1) + B2*l^2/(l^2-C2) + B3*l^2/(l^2-C3)

    Valid: 0.21 to 3.71 um.  Zero-dispersion wavelength ~1.27 um.
    At 1550 nm: n ~ 1.4440, anomalous dispersion (D ~ -20 ps/nm/km -> solitons).
    """
    l = np.asarray(lambda_um, float)
    if np.any(l <= 0):
        raise ValueError("wavelength must be positive")
    B1, C1 = 0.6961663, 0.0684043**2
    B2, C2 = 0.4079426, 0.1162414**2
    B3, C3 = 0.8974794, 9.896161**2
    n2 = 1 + B1*l**2/(l**2-C1) + B2*l**2/(l**2-C2) + B3*l**2/(l**2-C3)
    return np.sqrt(np.maximum(n2, 0))


# ── Jackson roadmap ───────────────────────────────────────────────────────────

JACKSON_ROADMAP = {
    "Ch1":  "Mathematical introduction: Green's theorem, delta functions",
    "Ch2":  "Boundary value problems: Laplace eq, method of images",
    "Ch3":  "Green's functions for Laplace/Poisson (more general than Griffiths Ch3)",
    "Ch4":  "Multipole expansion: exact series, not just dipole approximation",
    "Ch5":  "Magnetostatics: vector potential, Biot-Savart derivation",
    "Ch6":  "Time-varying fields: Maxwell from first principles (Gaussian units!)",
    "Ch7":  "Plane waves in media: n(omega) from first principles -> H(f)",
    "Ch8":  "Waveguides and cavities: TE/TM modes, resonant frequency",
    "Ch9":  "Diffraction: Kirchhoff integral -> Fraunhofer -> Fourier optics",
    "Ch10": "Lienard-Wiechert potentials: radiation from accelerating charges",
    "Ch11": "Radiation: Larmor formula, multipole radiation",
    "Ch14": "Special relativity and EM: 4-vectors, stress tensor",
    "Ch16": "Radiation reaction: Abraham-Lorentz force (self-force paradox)",
    "griffiths_to_jackson": {
        "Griffiths Ch4 (dielectrics)":  "Jackson Ch4 + Ch7 (Lorentz model, Kramers-Kronig)",
        "Griffiths Ch9 (EM waves)":     "Jackson Ch7 (full complex n, absorption, anomalous)",
        "Griffiths Ch11 (radiation)":   "Jackson Ch9-11 (Green's function radiation)",
        "Griffiths Ch12 (relativity)":  "Jackson Ch11-14 (covariant formulation)",
    },
    "this_repo_needs": [
        "Jackson Ch7: derive H(f)=exp(i*pi*D*f^2) from n(omega) Sellmeier",
        "Jackson Ch4: Kramers-Kronig (already in dgs/causality.py)",
        "Jackson Ch9: Fraunhofer -> far-field = Fourier transform (GS algorithm)",
    ],
}


def griffiths_chapter_status():
    """Which Griffiths chapters are covered in this repo."""
    return {
        "Ch1 Vector calculus":    "griffiths/vectors.py, griffiths/vector_identities.py",
        "Ch2 Electrostatics":     "griffiths/electrostatics.py",
        "Ch3 Potentials":         "griffiths/potentials.py, griffiths/fields.py",
        "Ch4 Dielectrics":        "classical_ed.py (this file) + dgs/causality.py",
        "Ch5 Magnetostatics":     "griffiths/magnetostatics.py",
        "Ch6 Magnetic matter":    "griffiths/magnetic_matter.py",
        "Ch7 Electrodynamics":    "griffiths/electrodynamics.py",
        "Ch8 Conservation":       "classical_ed.py poynting_vector_plane_wave()",
        "Ch9 EM Waves":           "notebooks/griffiths_ch9_dispersion.ipynb (flagship)",
        "Ch10 Potentials":        "NOT YET",
        "Ch11 Radiation":         "griffiths/radiation.py (partial)",
        "Ch12 Relativity":        "griffiths/relativity.py + dgs/spacetime.py",
    }


def demo():
    print("=" * 65)
    print("  dgs/classical_ed.py  --  Griffiths + Jackson demo")
    print("=" * 65)

    print("\n--- Lorentz oscillator: chi_e at resonance ---")
    omega_0 = 2 * np.pi * 600e12   # 600 THz resonance (UV)
    omega_p = 2 * np.pi * 200e12   # plasma frequency
    gamma   = 2 * np.pi * 10e12    # linewidth
    # at omega = 1550 nm = 193 THz (C-band telecom)
    omega_vis = 2 * np.pi * 193e12
    r = electric_susceptibility_lorentz(omega_vis, omega_0, gamma, omega_p)
    print(f"  chi_e at 1550nm:  {r['chi_e'].real:.4f} + {r['chi_e'].imag:.4f}j")
    print(f"  eps_r:            {r['eps_r'].real:.4f} + {r['eps_r'].imag:.4f}j")
    print(f"  n:                {r['n'].real:.4f} + {r['n'].imag:.6f}j")

    print("\n--- Sellmeier: silica fiber n at telecom wavelengths ---")
    lambdas = np.array([1.31, 1.55, 1.625])  # um
    ns = sellmeier_silica(lambdas)
    for l, n in zip(lambdas, ns):
        print(f"  lambda={l:.3f} um  n={n:.6f}")

    print("\n--- Maxwell wave equation (SymPy) ---")
    mx = maxwell_equations_sympy()
    print(f"  Gauss E:  {mx['gauss_E']}")
    print(f"  Faraday:  {mx['faraday']}")
    print(f"  Ampere:   {mx['ampere']}")
    print(f"  Dispersion relation: ", end="")
    sp.pprint(mx["dispersion_relation"])
    print(f"  Residual at k=omega/c: {mx['residual_at_k_eq_omega_over_c']}  (0 = wave eq satisfied)")

    print("\n--- Poynting vector (1550nm, n=1.444, E0=1 V/m) ---")
    poy = poynting_vector_plane_wave(E0=1.0, n_r=1.444)
    print(f"  <S> = {poy['S_avg_W_m2']:.2f} W/m^2")
    print(f"  (I1 = |E1|^2 in phase retrieval is proportional to this)")

    print("\n--- Griffiths chapter coverage in this repo ---")
    status = griffiths_chapter_status()
    for ch, loc in status.items():
        flag = "DONE" if "NOT YET" not in loc else "TODO"
        print(f"  [{flag}] {ch:30s} {loc}")

    print("\n--- Jackson roadmap (what comes after Griffiths) ---")
    for ch, desc in JACKSON_ROADMAP.items():
        if ch.startswith("Ch"):
            print(f"  {ch}: {desc}")


if __name__ == "__main__":
    demo()
