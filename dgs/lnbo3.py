"""
LiNbO3 (Lithium Niobate) material handler.
Sellmeier equations, electro-optic (EO) tensor, Pockels effect,
MZM phase shift, and connection to Coppinger1999 Eq(3).

Physics chain:
  Applied voltage V -> delta_n via r33 (EO coefficient)
  -> phase shift delta_phi = 2*pi*delta_n*L/lambda
  -> MZM output field: Eq(3) of Coppinger1999
  -> This IS the a*cos(2*pi*fm*t) modulation term

Hermitian connection:
  The phase modulator acts as exp(j*phi(t)) -- a UNITARY operator.
  |exp(j*phi)| = 1: no energy added, just phase rotated.
  Measurement of |E|^2 is a Hermitian observable (real eigenvalues = intensities).
  Photon number operator n_hat = a_dag * a is Hermitian -> real energy levels.
"""
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Sellmeier coefficients for LiNbO3 (Zelmon 1997)
# n^2 = A + B/(lambda^2 - C) - D*lambda^2
# lambda in micrometers
# ---------------------------------------------------------------------------
SELLMEIER = {
    'ordinary': {
        'A': 4.9048, 'B': 0.11768, 'C': 0.04750, 'D': 0.01276,
        'ref': 'Zelmon et al., JOSAB 1997'
    },
    'extraordinary': {
        'A': 4.5820, 'B': 0.09921, 'C': 0.04432, 'D': 0.01348,
        'ref': 'Zelmon et al., JOSAB 1997'
    }
}

# EO tensor elements [pm/V] (lithium niobate at 1550 nm, room temp)
EO_TENSOR = {
    'r13': 8.6,   # pm/V
    'r22': 3.4,   # pm/V
    'r33': 30.8,  # pm/V -- LARGEST: used in z-cut MZM
    'r51': 28.0,  # pm/V
    'ref': 'Wong, Properties of Lithium Niobate, 2002'
}

# Crystal structure
CRYSTAL = {
    'symmetry': 'trigonal 3m (rhombohedral)',
    'space_group': 'R3c',
    'Curie_temp_C': 1210,
    'density_gcm3': 4.628,
    'transparency_um': (0.4, 5.0),
    'damage_threshold_GWcm2': 0.5,
}


def sellmeier(wavelength_um, polarization='extraordinary'):
    """
    Refractive index of LiNbO3 via Sellmeier equation (Zelmon 1997).
    wavelength_um: wavelength in micrometers (e.g., 1.55 for 1550nm)
    Returns n (real).
    """
    c = SELLMEIER[polarization]
    lam2 = wavelength_um**2
    n2 = c['A'] + c['B']/(lam2 - c['C']) - c['D']*lam2
    return np.sqrt(n2)


def gvd_lnbo3(wavelength_um=1.55, polarization='extraordinary', delta=0.001):
    """
    GVD beta2 [ps^2/mm] for LiNbO3 via numerical second derivative of n(lambda).
    d^2n/dlambda^2 -> beta2 = (lambda^3 / 2*pi*c^2) * d^2n/dlambda^2
    """
    lam = wavelength_um
    n_plus = sellmeier(lam + delta, polarization)
    n_minus = sellmeier(lam - delta, polarization)
    n0 = sellmeier(lam, polarization)
    d2n_dlam2 = (n_plus - 2*n0 + n_minus) / delta**2  # 1/um^2

    c_um_ps = 2.998e2  # speed of light [um/ps]: 3e8 m/s * 1e6 um/m / 1e12 ps/s = 300 um/ps
    # beta2 = (lam^3 / 2*pi*c^2) * d2n/dlam2, units: ps^2/um -> convert to ps^2/mm
    beta2_ps2_um = (lam**3) / (2*np.pi * c_um_ps**2) * d2n_dlam2
    beta2_ps2_mm = beta2_ps2_um * 1e3  # 1/um = 1e3/mm
    return beta2_ps2_mm


def eo_phase_shift(V, L_mm, wavelength_um=1.55, polarization='extraordinary'):
    """
    Electro-optic phase shift in LiNbO3 waveguide (Pockels effect).
    delta_phi = pi * n_e^3 * r33 * V * L / (lambda * d)
    Simplified for waveguide with electrode gap d=6um (typical).

    V: applied voltage [V]
    L_mm: interaction length [mm]
    Returns delta_phi [rad].

    Connection to Coppinger Eq(3):
      V(t) = V_rf * cos(2*pi*fm*t)
      -> delta_phi(t) = (pi/V_pi) * V_rf * cos(2*pi*fm*t)
      -> E_out = E_in * exp(j*delta_phi(t))
      Small signal: exp(j*eps) ~ 1 + j*eps ~ 1 + j*(pi*a)*cos(2*pi*fm*t)
      This is the intensity modulation term if MZM is biased at quadrature.
    """
    n_e = sellmeier(wavelength_um, 'extraordinary')
    r33_m_per_V = EO_TENSOR['r33'] * 1e-12  # pm/V -> m/V
    lam_m = wavelength_um * 1e-6
    d_m = 6e-6  # electrode gap [m]
    L_m = L_mm * 1e-3
    delta_phi = np.pi * n_e**3 * r33_m_per_V * V * L_m / (lam_m * d_m)
    return delta_phi


def vpi(L_mm, wavelength_um=1.55):
    """
    Half-wave voltage V_pi: voltage needed for pi phase shift.
    V_pi = lambda * d / (n_e^3 * r33 * L)
    Lower V_pi = more efficient modulator.
    Typical LiNbO3: V_pi * L ~ 10 V*cm.
    """
    n_e = sellmeier(wavelength_um, 'extraordinary')
    r33_m_per_V = EO_TENSOR['r33'] * 1e-12
    lam_m = wavelength_um * 1e-6
    d_m = 6e-6
    L_m = L_mm * 1e-3
    Vpi = lam_m * d_m / (n_e**3 * r33_m_per_V * L_m)
    return Vpi


def mzm_lnbo3(V_bias, V_rf, fm_ghz, t_ps, L_mm=50.0, wavelength_um=1.55):
    """
    Full MZM transfer function using actual LiNbO3 EO physics.
    Connects to Coppinger Eq(3): a = V_rf/V_pi * sin(pi*V_bias/V_pi)

    Returns:
      E_out: complex field (phase modulation form)
      I_out: intensity |E_out|^2
      a_eff: effective linear modulation depth (Coppinger Eq 3 coefficient)
      V_pi_val: half-wave voltage [V]
    """
    V_pi_val = vpi(L_mm, wavelength_um)
    omega_m = 2*np.pi * fm_ghz * 1e9  # rad/s
    t_s = t_ps * 1e-12  # ps -> s

    # Phase in each arm of MZM
    phi_DC = np.pi * V_bias / V_pi_val
    phi_RF = np.pi * V_rf / V_pi_val * np.cos(omega_m * t_s)

    # MZM field transfer: E_out = E_in/2 * (e^j*phi_+ + e^j*phi_-)
    # For push-pull: phi_+ = phi_RF/2 + phi_DC/2, phi_- = -phi_RF/2 + phi_DC/2
    E_out = np.cos(phi_DC/2 + phi_RF/2) * np.exp(1j * phi_DC/2)
    I_out = np.abs(E_out)**2

    # Coppinger modulation depth (small signal at quadrature)
    a_eff = (np.pi * V_rf / V_pi_val) * np.sin(phi_DC)
    return E_out, I_out, a_eff, V_pi_val


# ---------------------------------------------------------------------------
# Hermitian operators: the physics of measurement
# ---------------------------------------------------------------------------
def hermitian_observable_demo():
    """
    Why |E|^2 is a Hermitian observable (quantum optics perspective).

    Photon number operator: n_hat = a_dag @ a
    Hermitian: n_hat = n_hat.dag -> real eigenvalues (measured photon counts)

    The PHASE of E cannot be directly measured (no Hermitian phase operator).
    Phase retrieval (GS algorithm) recovers the non-observable phase
    from TWO intensity measurements |E1|^2 and |E2|^2.

    Matrix mechanics connection (Python vs Java):
      Python/NumPy: matrix operations are native, Hermitian check = np.allclose(A, A.conj().T)
      Java: no native complex matrix; need Apache Commons Math or JBlas library
      SymPy: sp.Matrix([[1+1j, 2j], [-2j, 1-1j]]).is_hermitian -> True
    """
    # Build Hermitian matrix (energy operator)
    H_matrix = np.array([
        [1.0 + 0j, 2j      ],
        [-2j,      3.0 + 0j]
    ])
    is_hermitian = np.allclose(H_matrix, H_matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

    # Energy ratio: E2/E1 (the "symbol ratio" user asked about)
    energy_ratio = eigenvalues[1] / eigenvalues[0]

    return {
        'H_matrix': H_matrix,
        'is_hermitian': is_hermitian,
        'eigenvalues_real': eigenvalues,  # always real for Hermitian!
        'energy_ratio_E2_E1': energy_ratio,
        'connection': 'H(f)=exp(j*pi*D*f^2) is UNITARY (|eigenvalues|=1), not Hermitian.'
                      ' Measurement |E|^2 projects onto Hermitian number operator.',
        'python_check': "np.allclose(A, A.conj().T)",
        'sympy_check': "sp.Matrix(A).is_hermitian",
        'java_equivalent': "No native complex matrix -- use Apache Commons Math ComplexMatrix"
    }


def sympy_hermitian_symbolic():
    """SymPy symbolic Hermitian matrix and energy eigenvalue ratio."""
    E1, E2 = sp.symbols('E_1 E_2', positive=True)
    # Generic 2x2 Hermitian (real diagonal, complex off-diagonal)
    alpha, beta, gamma_r, gamma_i = sp.symbols('alpha beta gamma_r gamma_i', real=True)
    H = sp.Matrix([
        [alpha,              gamma_r + sp.I*gamma_i],
        [gamma_r - sp.I*gamma_i, beta             ]
    ])
    # Eigenvalues of Hermitian matrix are always real
    lam = sp.Symbol('lambda', real=True)
    char_poly = sp.det(H - lam*sp.eye(2))
    eigs = sp.solve(char_poly, lam)
    ratio = sp.simplify(eigs[1] / eigs[0])
    return {'H_symbolic': H, 'char_poly': char_poly,
            'eigenvalues': eigs, 'ratio': ratio}


# ---------------------------------------------------------------------------
# Python vs Java: symbol/operator comparison table
# ---------------------------------------------------------------------------
PYTHON_VS_JAVA = {
    'complex_number':  {'python': 'z = 3 + 4j',
                        'java': 'new Complex(3, 4)  // Apache Commons'},
    'matrix_multiply': {'python': 'C = A @ B',
                        'java': 'A.multiply(B)'},
    'hermitian_check': {'python': 'np.allclose(A, A.conj().T)',
                        'java': 'matrix.isHermitian(tol)  // JBlas'},
    'eigenvalues':     {'python': 'np.linalg.eigh(A)  // real for Hermitian',
                        'java': 'new EigenDecomposition(A).getRealEigenvalues()'},
    'symbolic_math':   {'python': 'sp.symbols("x"); sp.diff(x**2, x)',
                        'java': 'No built-in CAS; use Symja or Mathematica JLink'},
    'fft':             {'python': 'np.fft.fft(signal)',
                        'java': 'new FastFourierTransformer().transform(signal, DFT)'},
    'torch_autograd':  {'python': 'loss.backward()  // gradient flows through H(f)',
                        'java': 'No native autograd; use DL4J or TensorFlow Java'},
    'verdict':         'Python wins for physics/ML. Java wins for production Android/server.',
}


def demo():
    print("=== LiNbO3 Material Handler ===")
    for pol in ('ordinary', 'extraordinary'):
        n = sellmeier(1.55, pol)
        print(f"  n({pol}, 1550nm) = {n:.5f}")

    b2 = gvd_lnbo3(1.55)
    print(f"  GVD beta2 = {b2:.4f} ps^2/mm")

    Vpi = vpi(L_mm=50.0)
    print(f"  V_pi (L=50mm) = {Vpi:.2f} V")

    phi = eo_phase_shift(V=Vpi/2, L_mm=50.0)
    print(f"  Phase shift at V=Vpi/2: {phi:.4f} rad (expect pi/2={np.pi/2:.4f})")

    print("\n=== Hermitian Observable Demo ===")
    h = hermitian_observable_demo()
    print(f"  Is Hermitian: {h['is_hermitian']}")
    print(f"  Eigenvalues (real): {h['eigenvalues_real']}")
    print(f"  Energy ratio E2/E1: {h['energy_ratio_E2_E1']:.4f}")
    print(f"  Python check: {h['python_check']}")
    print(f"  Java equivalent: {h['java_equivalent']}")

    print("\n=== Python vs Java Symbol Comparison ===")
    for key, val in PYTHON_VS_JAVA.items():
        if key == 'verdict':
            print(f"  Verdict: {val}")
        else:
            print(f"  {key}:")
            print(f"    Python: {val['python']}")
            print(f"    Java:   {val['java']}")


if __name__ == '__main__':
    demo()
