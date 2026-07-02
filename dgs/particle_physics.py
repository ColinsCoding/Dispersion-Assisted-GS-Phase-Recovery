"""
Particle Physics: Standard Model + Cosmology + Field Theory

Theory-first curriculum map (read before engineering):
  Level 0 (Math foundations):     complex_numbers, linear_algebra, calculus
  Level 1 (Classical fields):     vector_calculus.py (curl/div/grad, Maxwell)
  Level 2 (Quantum mechanics):    modern_physics.py (QM 1D/3D, tunneling)
  Level 3 (Quantum information):  quantum_information.py (qubits, Chern, Majorana)
  Level 4 (Particle physics):     THIS FILE -- Standard Model, Feynman diagrams
  Level 5 (Quantum field theory): QFT -- creation/annihilation, path integrals
  Level 6 (Engineering):          dispersion_gs, gs_core, phase retrieval

The causality thread:
  Vector calculus: div(E)=rho/eps0 -- charge CAUSES field (causal PDE)
  Relativity:      nothing > c     -- causality PROTECTED by speed limit
  QM:              |psi|^2 = prob  -- individual events random, ensemble causal
  QFT:             time-ordering T -- ensures causal propagation of fields
  Standard Model:  gauge invariance -- FORCES arise from demanding local symmetry
  Cosmology:       arrows of time  -- thermodynamic + quantum + cosmological causality

Connection to this repo:
  Photon IS the gauge boson of electromagnetism (U(1) gauge field)
  H(f)=exp(j*pi*D*f^2): dispersion of photon in medium = QED in dense matter
  Gauge invariance -> conservation of charge -> current -> Maxwell's equations
  Higgs mechanism -> photon mass=0 (in vacuum) -> light travels at c
  Strong force: SU(3) color -> gluon field -> same math as non-abelian gauge theory
               used in topological photonics (SU(2) Berry phase in photonic crystal)
  Neutrino mixing: PMNS matrix = unitary rotation = same as qubit gate = beam splitter
  Particle classification: same group theory as crystallography -> photonic band structure
"""
import math
import numpy as np
import sympy as sp

# Physical constants
c     = 2.998e8       # m/s
hbar  = 1.0546e-34    # J*s
e     = 1.602e-19     # C
eV    = 1.602e-19     # J
MeV   = eV * 1e6
GeV   = eV * 1e9
me    = 9.109e-31     # kg
mp    = 1.673e-27     # kg
mn    = 1.675e-27     # kg
kB    = 1.381e-23     # J/K
G     = 6.674e-11     # m^3 kg^-1 s^-2 (gravitational constant)
alpha_fs = 1/137.036  # fine structure constant (coupling of EM)


# ============================================================
# Standard Model: Particle Classification
# ============================================================

def standard_model_particles():
    """
    The Standard Model: 17 fundamental particles + their properties.

    FERMIONS (spin-1/2, obey Fermi-Dirac statistics, Pauli exclusion):
      Quarks (6 flavors, 3 colors each = 18 quark states + 18 antiquarks):
        up (u):      mass ~2.2 MeV/c^2,  charge +2/3
        down (d):    mass ~4.7 MeV/c^2,  charge -1/3
        charm (c):   mass ~1.27 GeV/c^2, charge +2/3
        strange (s): mass ~95 MeV/c^2,   charge -1/3
        top (t):     mass ~173 GeV/c^2,  charge +2/3  [heaviest known particle]
        bottom (b):  mass ~4.18 GeV/c^2, charge -1/3
      Leptons (6 flavors, no color):
        electron (e):         mass 0.511 MeV/c^2, charge -1
        muon (mu):            mass 105.7 MeV/c^2, charge -1
        tau:                  mass 1777 MeV/c^2,  charge -1
        electron neutrino:    mass < 2 eV,         charge 0
        muon neutrino:        mass < 0.19 MeV,     charge 0
        tau neutrino:         mass < 18 MeV,        charge 0

    BOSONS (spin integer, obey Bose-Einstein statistics, no Pauli exclusion):
      Force carriers (spin-1 gauge bosons):
        photon (gamma):  mass 0,          charge 0   [EM force, U(1)]
        gluon (g):       mass 0,  charge 0 [strong force, SU(3), 8 colors]
        W+, W-:          mass 80.4 GeV,   charge +/-1  [weak force]
        Z0:              mass 91.2 GeV,   charge 0     [weak force]
      Scalar boson (spin-0):
        Higgs (H):       mass 125.1 GeV,  charge 0   [gives mass to W,Z,fermions]
      GRAVITY: NOT in Standard Model. Graviton (spin-2) is hypothetical.

    Forces and their ranges:
      Electromagnetic: infinite range (photon mass=0), coupling alpha=1/137
      Strong:          ~1 fm = 10^-15 m (gluon self-interaction, color confinement)
      Weak:            ~0.001 fm = 10^-18 m (W,Z mass=80-91 GeV -> short range)
      Gravity:         infinite range but ~10^-38 weaker than EM (not in SM)

    Symmetry group: SU(3)_C x SU(2)_L x U(1)_Y
      SU(3)_C: strong force (color)
      SU(2)_L: weak isospin (left-handed doublets)
      U(1)_Y:  hypercharge
      After Higgs mechanism: SU(2)_L x U(1)_Y -> U(1)_EM (electroweak symmetry breaking)
    """
    quarks = {
        'up':      {'mass_MeV': 2.2,      'charge': +2/3, 'gen': 1, 'color': True},
        'down':    {'mass_MeV': 4.7,      'charge': -1/3, 'gen': 1, 'color': True},
        'charm':   {'mass_MeV': 1270,     'charge': +2/3, 'gen': 2, 'color': True},
        'strange': {'mass_MeV': 95,       'charge': -1/3, 'gen': 2, 'color': True},
        'top':     {'mass_MeV': 173000,   'charge': +2/3, 'gen': 3, 'color': True},
        'bottom':  {'mass_MeV': 4180,     'charge': -1/3, 'gen': 3, 'color': True},
    }
    leptons = {
        'electron':          {'mass_MeV': 0.511,    'charge': -1, 'gen': 1},
        'muon':              {'mass_MeV': 105.7,    'charge': -1, 'gen': 2},
        'tau':               {'mass_MeV': 1777,     'charge': -1, 'gen': 3},
        'e_neutrino':        {'mass_eV':  '<2',     'charge':  0, 'gen': 1},
        'mu_neutrino':       {'mass_eV':  '<190000','charge':  0, 'gen': 2},
        'tau_neutrino':      {'mass_eV':  '<18e6',  'charge':  0, 'gen': 3},
    }
    bosons = {
        'photon': {'mass': 0,      'charge': 0,  'spin': 1, 'force': 'EM',    'range': 'infinite'},
        'gluon':  {'mass': 0,      'charge': 0,  'spin': 1, 'force': 'strong','range': '1 fm', 'colors': 8},
        'W+':     {'mass_GeV': 80.4, 'charge':+1,'spin': 1, 'force': 'weak',  'range': '0.001 fm'},
        'W-':     {'mass_GeV': 80.4, 'charge':-1,'spin': 1, 'force': 'weak',  'range': '0.001 fm'},
        'Z0':     {'mass_GeV': 91.2, 'charge': 0,'spin': 1, 'force': 'weak',  'range': '0.001 fm'},
        'Higgs':  {'mass_GeV': 125.1,'charge': 0,'spin': 0, 'force': 'mass',  'role': 'Higgs mechanism'},
        'graviton':{'mass': 0,     'charge': 0,  'spin': 2, 'force': 'gravity','status': 'hypothetical'},
    }

    # Composite particles (hadrons)
    hadrons = {
        'proton':  {'quarks': 'uud', 'mass_MeV': 938.3, 'charge': +1, 'baryon': True},
        'neutron': {'quarks': 'udd', 'mass_MeV': 939.6, 'charge':  0, 'baryon': True},
        'pion+':   {'quarks': 'u_dbar', 'mass_MeV': 139.6, 'charge': +1, 'meson': True},
        'pion0':   {'quarks': 'uu_ddbar_mix', 'mass_MeV': 135.0, 'charge': 0, 'meson': True},
        'kaon+':   {'quarks': 'u_sbar', 'mass_MeV': 493.7, 'charge': +1, 'meson': True},
    }

    # Forces and couplings
    forces = {
        'electromagnetic': {
            'mediator': 'photon', 'mass_mediator': 0,
            'coupling': alpha_fs, 'coupling_name': 'alpha = 1/137',
            'range': 'infinite (1/r^2)',
            'symmetry': 'U(1)_EM',
            'EE_connection': 'Maxwell equations are the classical limit of QED',
        },
        'strong': {
            'mediator': 'gluon (8 types)', 'mass_mediator': 0,
            'coupling': 0.118, 'coupling_name': 'alpha_s(M_Z) ~ 0.118',
            'range': '~1 fm (confinement)',
            'symmetry': 'SU(3)_color',
            'key': 'gluons carry color charge -> self-interact -> confinement -> quarks bound in hadrons',
        },
        'weak': {
            'mediator': 'W+/W-/Z0', 'mass_mediator_GeV': 80,
            'coupling': 0.03, 'coupling_name': 'alpha_w ~ 1/30',
            'range': '~0.001 fm',
            'symmetry': 'SU(2)_L x U(1)_Y -> U(1)_EM after Higgs',
            'key': 'Mediates beta decay: n -> p + e- + nu_e. Changes quark flavor (d->u).',
        },
        'gravity': {
            'mediator': 'graviton (hypothetical)', 'mass_mediator': 0,
            'coupling': 6e-39, 'coupling_name': 'G_N/hbar*c ~ 6e-39',
            'range': 'infinite (1/r^2)',
            'symmetry': 'general covariance (GR)',
            'key': 'Not in Standard Model. Quantizing GR remains unsolved.',
        },
    }

    return {
        'quarks': quarks, 'leptons': leptons, 'bosons': bosons,
        'hadrons': hadrons, 'forces': forces,
        'symmetry_group': 'SU(3)_C x SU(2)_L x U(1)_Y',
        'total_particles': 17,
        'generations': 3,
        'repo_connection': {
            'photon': 'H(f)=exp(j*pi*D*f^2): dispersion of photons in fiber = QED in matter',
            'Higgs': 'Photon mass=0 in vacuum (no Higgs coupling) -> c is universal -> causality',
            'gauge': 'Demanding local gauge invariance FORCES the existence of the photon field',
        },
    }


# ============================================================
# The Four Forces: Field Equations
# ============================================================

def four_forces_field_equations():
    """
    Classical and quantum field equations for all four fundamental forces.

    ELECTROMAGNETISM (Maxwell / QED):
      curl(E) = -dB/dt                    [Faraday]
      curl(B) = mu0*J + mu0*eps0*dE/dt    [Ampere-Maxwell]
      div(E) = rho/eps0                   [Gauss]
      div(B) = 0                          [no monopoles]
      Gauge field: A_mu = (phi/c, A_vec)
      Maxwell: d^mu F_{mu nu} = -mu0 J_nu
      QED Lagrangian: L = -F_{mu nu}^2/4 + psi_bar(i*gamma^mu*D_mu - m)*psi
      D_mu = d_mu - i*e*A_mu   [covariant derivative]

    STRONG FORCE (QCD):
      Same structure as QED but with SU(3) gauge field:
      L = -1/4 G^a_{mu nu}G^{a,mu nu} + sum_q psi_bar_q(i*gamma^mu*D_mu - m_q)*psi_q
      G^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu + g_s*f^{abc}*A^b_mu*A^c_nu
      The f^{abc} term (gluon self-coupling) is NEW vs QED -- causes confinement.
      Alpha_s runs: large at low energy (confinement) -> small at high energy (asymptotic freedom)

    WEAK FORCE (electroweak = Weinberg-Salam):
      Gauge group: SU(2)_L x U(1)_Y
      After Higgs: W+/W-/Z0 get mass, photon stays massless.
      W mass: M_W = g*v/2 where v=246 GeV (Higgs VEV).
      CKM matrix: quark flavor mixing 3x3 unitary matrix
      PMNS matrix: neutrino mixing 3x3 unitary matrix [same math as CKM]
        theta_12=33.4 deg, theta_23=49 deg, theta_13=8.6 deg

    GRAVITY (General Relativity, not in SM):
      G_{mu nu} + Lambda*g_{mu nu} = 8*pi*G/c^4 * T_{mu nu}
      G_{mu nu} = R_{mu nu} - g_{mu nu}*R/2   [Einstein tensor]
      R_{mu nu}: Ricci curvature tensor
      T_{mu nu}: stress-energy tensor (energy + momentum + pressure)
      Lambda: cosmological constant (dark energy)
      Prediction: gravitational waves (LIGO 2015), black holes, expanding universe
    """
    # Symbolically: Maxwell equations in covariant form
    x_sym = sp.Symbol('x')
    mu_sym = sp.Symbol('mu')
    F = sp.Symbol('F_{mu nu}')
    J = sp.Symbol('J^nu')
    A = sp.Symbol('A_mu')

    maxwell_covariant = sp.Eq(
        sp.Symbol('partial^mu F_{mu nu}'),
        -sp.Symbol('mu_0') * J
    )
    gauge_invariance = sp.Eq(
        sp.Symbol("A_mu'"),
        A + sp.Symbol('partial_mu lambda')
    )

    # Fine structure constant and running coupling
    Q_arr = np.logspace(1, 5, 200)   # 10 MeV to 100 TeV
    alpha_QED = alpha_fs / (1 - alpha_fs/(3*np.pi)*np.log(Q_arr/0.511))
    alpha_s = 0.118 / np.maximum(0.3, 1 + 0.118*7/(2*np.pi)*np.log(Q_arr/91200))

    # Yukawa potential (massive mediator) vs Coulomb (massless)
    r_fm = np.linspace(0.01, 5, 400)   # fm
    hbar_c_fm_MeV = 197.3   # hbar*c in MeV*fm
    m_W_MeV = 80400
    V_EM = -alpha_fs / r_fm   # Coulomb, arbitrary units
    V_weak = -alpha_fs * np.exp(-m_W_MeV * r_fm / hbar_c_fm_MeV) / r_fm  # Yukawa

    # CKM matrix (quark mixing)
    theta_12 = np.radians(13.04)
    theta_23 = np.radians(2.38)
    theta_13 = np.radians(0.201)
    delta_CP = np.radians(68.0)   # CP violation phase

    c12,s12 = np.cos(theta_12), np.sin(theta_12)
    c23,s23 = np.cos(theta_23), np.sin(theta_23)
    c13,s13 = np.cos(theta_13), np.sin(theta_13)
    eid = np.exp(1j*delta_CP)

    CKM = np.array([
        [c12*c13,                       s12*c13,                   s13*np.exp(-1j*delta_CP)],
        [-s12*c23 - c12*s23*s13*eid,    c12*c23 - s12*s23*s13*eid, s23*c13              ],
        [ s12*s23 - c12*c23*s13*eid,   -c12*s23 - s12*c23*s13*eid, c23*c13              ],
    ])

    # PMNS matrix (neutrino mixing)
    th12,th23,th13 = np.radians(33.4), np.radians(49.0), np.radians(8.6)
    c12n,s12n = np.cos(th12), np.sin(th12)
    c23n,s23n = np.cos(th23), np.sin(th23)
    c13n,s13n = np.cos(th13), np.sin(th13)

    PMNS = np.array([
        [c12n*c13n,                s12n*c13n,               s13n],
        [-s12n*c23n - c12n*s23n*s13n, c12n*c23n - s12n*s23n*s13n, s23n*c13n],
        [ s12n*s23n - c12n*c23n*s13n,-c12n*s23n - s12n*c23n*s13n, c23n*c13n],
    ])
    CKM_unitary = np.allclose(CKM @ CKM.conj().T, np.eye(3), atol=1e-10)
    PMNS_unitary = np.allclose(PMNS @ PMNS.conj().T, np.eye(3), atol=1e-10)

    return {
        'maxwell_covariant': str(maxwell_covariant),
        'gauge_invariance': str(gauge_invariance),
        'running_coupling': {
            'Q_MeV': Q_arr.tolist(),
            'alpha_QED': alpha_QED.tolist(),
            'alpha_s_QCD': alpha_s.tolist(),
            'note': 'alpha_QED increases with energy; alpha_s DECREASES (asymptotic freedom)',
        },
        'Yukawa_vs_Coulomb': {
            'r_fm': r_fm.tolist(),
            'V_EM': V_EM.tolist(),
            'V_weak': V_weak.tolist(),
            'lesson': 'Massive mediator (W,Z) -> exponential suppression at long range',
        },
        'CKM_matrix': CKM.tolist(),
        'PMNS_matrix': PMNS.tolist(),
        'CKM_unitary': bool(CKM_unitary),
        'PMNS_unitary': bool(PMNS_unitary),
        'PMNS_beam_splitter_analogy': (
            'PMNS matrix is unitary 3x3 -- same as a 3-port beam splitter network.\n'
            'Neutrino oscillation nu_e -> nu_mu -> nu_tau is INTERFERENCE of mass eigenstates.\n'
            'Same physics as interferometer: U matrix maps between bases.\n'
            'Same math as photonic beam splitter in quantum_information.py.'
        ),
    }


# ============================================================
# Feynman Diagrams and Scattering Amplitudes
# ============================================================

def feynman_diagrams():
    """
    Feynman diagrams: pictorial representation of perturbation theory.

    Rules:
      External lines: incoming/outgoing particles (on-shell: p^2 = m^2 c^2)
      Internal lines (propagators): virtual particles (off-shell: p^2 != m^2 c^2)
      Vertices: interaction coupling constant (e for EM, g_s for QCD, g_w for weak)
      Each diagram: contribution to scattering amplitude M
      Cross section: sigma ~ |M|^2

    Key processes:
      Compton scattering: gamma + e- -> gamma + e-  (2 diagrams at tree level)
        QED vertex: e, so M ~ e^2 ~ alpha = 1/137
      Electron-positron annihilation: e+ + e- -> 2 gamma  (1 tree diagram)
      Pair production: gamma + gamma -> e+ + e-  (related by crossing symmetry)
      Beta decay: n -> p + e- + nu_e
        = d -> u + W- -> u + e- + nu_e  (weak vertex: g_w)
      Higgs production: gg -> H  (via top quark loop -- loop diagram!)
        = gluon fusion: dominant at LHC

    Perturbation theory: M = M_tree + M_1loop + M_2loop + ...
      Each loop adds factor ~ alpha/(2*pi) (for QED) or alpha_s/(2*pi) (QCD)
      Tree level: O(alpha^n) where n = number of vertices
      Loop corrections: infrared and ultraviolet divergences -> renormalization

    Renormalization: absorb UV infinities into redefined particle masses and couplings.
      Running coupling: coupling constant DEPENDS on energy scale Q (renormalization group)
      alpha(Q) ~ alpha/(1 - alpha/(3*pi)*ln(Q/m_e))  [QED one-loop]
      This IS why alpha_s is large at low Q (confinement) and small at high Q (asymptotic freedom)
    """
    # Lowest-order e+e- -> mu+mu- cross section
    # sigma = 4*pi*alpha^2/(3*s) where s = (E_cm)^2
    E_cm_GeV = np.logspace(-1, 2, 200)   # 0.1 to 100 GeV
    s = (E_cm_GeV * GeV)**2
    hbar_c_m = hbar * c   # J*m = 1.974e-16 GeV*m
    hbar_c_GeV_m = 0.1973e-15   # 0.1973 fm * GeV  [natural units]

    sigma_QED = 4*np.pi*alpha_fs**2 * (hbar_c_GeV_m**2) / (3*s/(GeV**2)) * 1e31   # nb

    # Z resonance peak at sqrt(s) = M_Z = 91.2 GeV
    M_Z = 91.2; Gamma_Z = 2.495   # GeV
    BW_Z = (Gamma_Z**2/4) / ((E_cm_GeV - M_Z)**2 + Gamma_Z**2/4)
    sigma_Z_peak = 41.5 * BW_Z   # nb, peak cross section 41.5 nb

    # Running of alpha_s
    n_f = 5   # light quark flavors at M_Z
    b0 = 11 - 2*n_f/3   # one-loop beta function coefficient
    alpha_s_MZ = 0.118
    Q = np.logspace(1, 4, 300)   # 10 MeV to 10 TeV
    alpha_s_running = alpha_s_MZ / np.maximum(0.3, 1 + alpha_s_MZ * b0/(2*np.pi) * np.log(Q/91200))

    return {
        'e_plus_e_minus_sigma_QED_nb': sigma_QED.tolist(),
        'E_cm_GeV': E_cm_GeV.tolist(),
        'Z_resonance': {
            'sigma_Z_nb': sigma_Z_peak.tolist(),
            'M_Z_GeV': M_Z,
            'Gamma_Z_GeV': Gamma_Z,
            'lesson': 'Breit-Wigner resonance: same shape as resonator S21 in EE',
        },
        'alpha_s_running_Q_MeV': Q.tolist(),
        'alpha_s_running': alpha_s_running.tolist(),
        'key_processes': {
            'Compton':    'gamma + e- -> gamma + e-: energy transfer = Compton shift',
            'annihilation': 'e+ + e- -> 2*gamma: 511 keV each -> PET scanner',
            'beta_decay': 'd -> u + W- -> u + e- + nu_e: weak force changes quark flavor',
            'Higgs_ggH':  'gg -> H via top loop: how Higgs was discovered at LHC (2012)',
        },
        'Breit_Wigner_resonator_analogy': (
            'Z boson: sigma(E) ~ Gamma^2/4 / [(E-M_Z)^2 + Gamma^2/4]\n'
            'Resonator: |S21|^2 ~ (kappa_ext/2)^2 / [(omega-omega_r)^2 + (kappa/2)^2]\n'
            'IDENTICAL FORM. Resonance = unstable state = finite lifetime = Gamma = 1/tau.\n'
            'Gamma*tau = hbar (energy-time uncertainty). Same physics at all scales.'
        ),
    }


# ============================================================
# Conservation Laws and Symmetries (Noether's theorem)
# ============================================================

def conservation_laws_noether():
    """
    Noether's theorem: every continuous symmetry -> conserved quantity.

    Translation in time    -> Conservation of energy
    Translation in space   -> Conservation of momentum
    Rotation in space      -> Conservation of angular momentum
    U(1) gauge symmetry    -> Conservation of electric charge
    SU(3) color symmetry   -> Conservation of color charge
    Baryon number B        -> (approximately) conserved
    Lepton number L        -> (approximately) conserved
    CPT                    -> EXACT symmetry (deep consequence of QFT)

    DISCRETE SYMMETRIES:
      C (charge conjugation): particle <-> antiparticle
      P (parity): spatial reflection (r -> -r)
      T (time reversal): t -> -t
      CP violation: CKM phase delta_CP != 0 -> matter/antimatter asymmetry!
        This is WHY the universe has more matter than antimatter.
        A causal story: if CP were exact, big bang would make equal matter + antimatter
        -> they would annihilate -> only photons remain -> NO MATTER, NO US.
        CP violation is why you exist.
      CPT: ALWAYS conserved in any Lorentz-invariant QFT.

    APPROXIMATE SYMMETRIES (broken by interactions):
      Isospin SU(2): u and d are "the same" to strong force (broken by mass difference)
      Chiral symmetry: massless quarks have L/R symmetry (broken by QCD condensate)
      Flavor SU(6): all quark flavors same (broken by large mass differences)

    Conservation law checks for particle reactions:
      B (baryon number): quarks B=1/3, antiquarks B=-1/3, leptons B=0
      L (lepton number): leptons L=1, antileptons L=-1, quarks L=0
      Q (charge): must sum to same before and after
      Energy-momentum: 4-momentum p^mu conserved at each vertex
    """
    # Check conservation laws for reaction: p + p -> p + p + pi0
    # p=uud, B=1 each; pi0=uu_bar, B=0; charge: p has Q=1 each
    reaction = {
        'initial': {'particles': ['p', 'p'], 'B': 2, 'L': 0, 'Q': 2},
        'final':   {'particles': ['p', 'p', 'pi0'], 'B': 2, 'L': 0, 'Q': 2},
        'B_conserved': True, 'L_conserved': True, 'Q_conserved': True,
        'allowed': True,
    }
    # Forbidden: p -> e+ + pi0 (violates B)
    forbidden = {
        'reaction': 'p -> e+ + pi0',
        'initial_B': 1, 'final_B': 0,
        'B_conserved': False, 'allowed': False,
        'note': 'No observed proton decay -> proton lifetime > 10^34 years',
    }

    # Noether charges as integrals
    noether_table = [
        ('Time translation',     'H (Hamiltonian)',       'Energy E'),
        ('Space translation',    'P (momentum operator)', 'Momentum p'),
        ('Rotation',             'L (angular momentum)',  'Angular momentum J'),
        ('U(1) phase psi->e^jq*psi', 'charge operator', 'Electric charge Q'),
        ('SU(3) color rotation', 'color generators T^a', 'Color charge'),
        ('Lorentz boost',        'boost generator K',     'Center-of-energy'),
    ]

    # CP violation: delta_CP in CKM
    delta_CP_deg = 68.0
    CP_violation = {
        'CP_phase_deg': delta_CP_deg,
        'CP_violation_magnitude': 'epsilon_K ~ 2e-3 (kaon system)',
        'why_it_matters': (
            'CP violation + baryon number violation + departure from thermal equilibrium\n'
            '= Sakharov conditions (1967) for matter/antimatter asymmetry.\n'
            'Result: ~1 extra baryon per 10^9 photons in early universe.\n'
            'That 1 extra baryon per billion is ALL the matter around us.\n'
            'CP violation is the causal origin of matter dominance -> stars -> life.'
        ),
    }

    return {
        'noether_table': [{'symmetry': s, 'generator': g, 'conserved': c}
                          for s,g,c in noether_table],
        'allowed_reaction': reaction,
        'forbidden_reaction': forbidden,
        'CPT': 'EXACT in any Lorentz-invariant QFT: C*P*T = 1',
        'CP_violation': CP_violation,
        'proton_lifetime': '>10^34 years (no decay observed)',
        'photon_connection': (
            'U(1) gauge symmetry -> conservation of electric charge.\n'
            'Current conservation: div(J) + d(rho)/dt = 0  (continuity equation)\n'
            'This IS the same continuity equation in causality.py.\n'
            'Maxwell equations are the CONSEQUENCE of requiring U(1) gauge invariance.\n'
            'The photon EXISTS because of this symmetry.'
        ),
    }


# ============================================================
# Cosmology: Big Bang, CMB, Dark Matter, Dark Energy
# ============================================================

def cosmology():
    """
    Cosmology: Friedmann equation, Big Bang, inflation, CMB, dark matter, dark energy.

    Friedmann equation (from GR + homogeneous isotropic universe):
      (da/dt/a)^2 = H^2 = 8*pi*G*rho/3 - k*c^2/a^2 + Lambda*c^2/3
      a = scale factor, H = Hubble parameter, rho = energy density
      k = spatial curvature (+1 closed, 0 flat, -1 open)
      Lambda = cosmological constant (dark energy)

    Cosmic history (lookback time):
      t=0:            Big Bang (T -> infinity, all forces unified?)
      t=10^-43 s:     Planck time (quantum gravity important, E ~ 10^19 GeV)
      t=10^-35 s:     Inflation (exponential expansion, a ~ exp(H*t))
      t=10^-12 s:     Electroweak phase transition (W,Z get mass)
      t=10^-6 s:      QCD phase transition (quarks -> hadrons)
      t=3 min:        Big Bang nucleosynthesis (H, He, Li formed)
      t=380,000 yr:   Recombination: electrons + protons -> H atoms, universe TRANSPARENT
                      -> CMB (Cosmic Microwave Background) emitted
      t=100 Myr:      First stars (Population III: pure H+He, no metals)
      t=9 Gyr:        Solar system forms (4.6 Gyr ago)
      t=13.8 Gyr:     NOW

    Energy budget of universe (Planck 2018):
      Dark energy (Lambda): 68.3%
      Dark matter:          26.8%
      Baryonic matter:       4.9%  [stars, gas, YOU: only 4.9% of the universe!]

    CMB:
      Temperature: T_CMB = 2.725 K  [perfect blackbody!]
      Peak wavelength: lambda_max = 2.898e-3/2.725 = 1.063 mm  [microwave!]
      Anisotropies: Delta_T/T ~ 10^-5  [seeds of large-scale structure]
      Power spectrum: C_l ~ l*(l+1)*C_l -- same FT-based analysis as signal processing
    """
    # Hubble parameter and age of universe
    H0_km_s_Mpc = 67.4   # Planck 2018 [km/s/Mpc]
    H0_per_s = H0_km_s_Mpc * 1e3 / (3.086e22)   # convert to 1/s
    t_Hubble = 1/H0_per_s / (3.156e7 * 1e9)   # Gyr

    # CMB temperature and blackbody
    T_CMB = 2.725
    lambda_CMB_mm = 2.898e-3 / T_CMB * 1e3
    h = 6.626e-34
    nu_arr = np.linspace(1e9, 3e11, 500)   # GHz range
    I_CMB = (2*h*nu_arr**3/c**2) / (np.exp(h*nu_arr/(kB*T_CMB)) - 1)

    # Friedmann equation: flat universe (k=0)
    Omega_Lambda = 0.683; Omega_m = 0.317; Omega_r = 9e-5
    a_arr = np.linspace(0.001, 1, 400)   # scale factor from BBN to now
    H_over_H0 = np.sqrt(Omega_r/a_arr**4 + Omega_m/a_arr**3 + Omega_Lambda)

    # Dark matter evidence
    dark_matter_evidence = {
        'rotation_curves': 'Galaxy rotation: v(r) stays flat at large r -> invisible mass',
        'gravitational_lensing': 'Bullet cluster: mass != light (dark matter separated from gas)',
        'CMB_peaks': 'Baryon acoustic oscillations: dark matter affects oscillation amplitude',
        'structure_formation': 'Without DM, galaxies and large-scale structure cannot form',
        'candidates': ['WIMPs (weakly interacting massive particles)',
                       'Axions (light pseudo-scalar, motivated by strong CP problem)',
                       'Sterile neutrinos',
                       'Primordial black holes'],
    }

    # Nucleosynthesis: elements formed in Big Bang
    BBN = {
        'H':   0.75,   # mass fraction
        'He4': 0.25,
        'Li7': 3e-10,
        'D':   2.5e-5,
        'note': 'Everything heavier than Li was made in stars (stellar nucleosynthesis)',
    }

    return {
        'H0_km_s_Mpc': H0_km_s_Mpc,
        't_Hubble_Gyr': t_Hubble,
        'T_CMB_K': T_CMB,
        'lambda_CMB_peak_mm': lambda_CMB_mm,
        'nu_CMB_GHz': nu_arr/1e9,
        'I_CMB': I_CMB.tolist(),
        'energy_budget': {'dark_energy_pct': 68.3, 'dark_matter_pct': 26.8, 'baryons_pct': 4.9},
        'Friedmann': {
            'a_arr': a_arr.tolist(),
            'H_over_H0': H_over_H0.tolist(),
            'equation': 'H^2 = H0^2 * (Omega_r/a^4 + Omega_m/a^3 + Omega_Lambda)',
        },
        'dark_matter': dark_matter_evidence,
        'BBN': BBN,
        'photon_connection': (
            'CMB is a perfect blackbody (Planck distribution) at T=2.725 K.\n'
            'CMB photons: redshifted from T=3000 K at recombination -> microwave today.\n'
            'Same Planck distribution as quantum_theory_of_light().\n'
            'CMB anisotropy analysis: spherical harmonics Y_lm (from qm_3d_hydrogen).\n'
            'Power spectrum C_l = FT of temperature correlation function.\n'
            'Same math as spectral analysis in dispersion_gs.'
        ),
    }


# ============================================================
# Quantum Field Theory Preview
# ============================================================

def quantum_field_theory_preview():
    """
    QFT Level 5: creation/annihilation operators, path integrals, Lagrangian.

    QFT = QM + special relativity.
    Key insight: particles are EXCITATIONS of underlying quantum fields.
      Photon = excitation of the electromagnetic field A_mu(x,t)
      Electron = excitation of the Dirac field psi(x,t)
      Higgs boson = excitation of the Higgs field phi(x,t)

    Creation/annihilation operators:
      a^dag(k) creates a photon with momentum k
      a(k) destroys a photon with momentum k
      [a(k), a^dag(k')] = delta^3(k-k')  [bosons -- commutator]
      {c(k), c^dag(k')} = delta^3(k-k')  [fermions -- anticommutator -> Pauli exclusion]

    Klein-Gordon equation (spin-0 boson):
      (d^2/dt^2 - c^2*nabla^2 + (mc^2/hbar)^2) phi = 0
      Dispersion: omega^2 = c^2*k^2 + (mc^2/hbar)^2
      m=0: omega = c*k (massless photon)
      m>0: omega = sqrt(c^2*k^2 + m^2c^4/hbar^2)  (massive particle)
      Compare GVD: omega = c*k/n + beta2/2*(k-k0)^2  -- SAME DISPERSION FORM

    Dirac equation (spin-1/2 fermion):
      (i*gamma^mu*d_mu - mc/hbar) psi = 0
      gamma matrices: 4x4 analogues of Pauli matrices
      Prediction: positron (antiparticle) -- confirmed 1932.
      Spin emerges AUTOMATICALLY from combining QM + SR.

    Path integral (Feynman):
      Z = int D[phi] exp(i*S[phi]/hbar)
      S[phi] = int d^4x L(phi, d_mu phi)   [action]
      All paths contribute, weighted by exp(i*S/hbar)
      Classical limit: stationary phase -> delta S = 0 -> Euler-Lagrange equations
      Quantum corrections: fluctuations around classical path -> loop diagrams

    Connection to GS phase retrieval:
      GS update: phi_new = angle(F^{-1}[sqrt(I)*exp(j*phi_old)])
      = stationary phase of the action S[phi] = ||sqrt(I)-|F{A*e^{j*phi}}|||^2
      GS IS solving a path integral problem in function space.
    """
    # Klein-Gordon dispersion
    k_arr = np.linspace(0, 3e10, 400)   # wavenumber (1/m)
    mc2_hbar = me*c**2/hbar   # = m_e*c/hbar for electron (Compton wavenumber)
    omega_photon  = c * k_arr
    omega_electron = c * np.sqrt(k_arr**2 + mc2_hbar**2)
    omega_massive = c * np.sqrt(k_arr**2 + (10*mc2_hbar)**2)   # 10x heavier

    # Group velocity: v_gr = d(omega)/dk
    # Photon: v_gr = c (dispersionless)
    # Massive: v_gr = c^2*k/omega < c (always less than c, causality)
    v_gr_electron = c**2 * k_arr[1:] / omega_electron[1:]   # m/s

    # Creation/annihilation: coherent state (laser)
    # Coherent state |alpha> is eigenstate of a: a|alpha> = alpha|alpha>
    # |alpha> = exp(-|alpha|^2/2) * sum_n alpha^n/sqrt(n!) * |n>
    alpha_laser = 10.0 + 5j   # complex amplitude
    n_arr = np.arange(0, 100)
    P_n = np.exp(-abs(alpha_laser)**2) * abs(alpha_laser)**(2*n_arr) / np.array(
        [float(math.factorial(n)) for n in n_arr]
    )   # Poisson photon number distribution

    mean_n = float(abs(alpha_laser)**2)
    variance_n = float(abs(alpha_laser)**2)   # Poisson: mean = variance

    # Vacuum energy (zero-point energy of all fields)
    # Sum_k (1/2)*hbar*omega_k -> divergent!
    # This IS the cosmological constant problem.
    # Regularized: Lambda_obs < 10^-52 m^-2
    # Predicted by naive QFT: Lambda_QFT ~ M_Planck^4/hbar^3*c^3 ~ 10^71 GeV^4
    # Ratio: 10^120 off -- "worst prediction in physics"
    lambda_prediction_ratio = 1e120

    return {
        'Klein_Gordon': {
            'k_per_m': k_arr.tolist(),
            'omega_photon_rad_per_s': omega_photon.tolist(),
            'omega_electron': omega_electron.tolist(),
            'omega_massive': omega_massive.tolist(),
            'v_group_electron_frac_c': (v_gr_electron/c).tolist(),
            'note': 'v_gr < c for massive particle: causality is automatic in Klein-Gordon',
        },
        'coherent_state_photon': {
            'alpha': alpha_laser,
            'n_arr': n_arr.tolist(),
            'P_n': P_n.tolist(),
            'mean_n': mean_n,
            'variance_n': variance_n,
            'is_Poisson': True,
            'laser_is_coherent_state': True,
        },
        'vacuum_energy': {
            'cosmological_constant_problem': 'QFT prediction vs observation: off by 10^120',
            'ratio': lambda_prediction_ratio,
            'status': 'Largest unexplained discrepancy in physics',
        },
        'GS_path_integral': (
            'Feynman path integral: Z = int D[phi] exp(j*S/hbar)\n'
            'GS phase retrieval: phi* = argmin_{phi} ||sqrt(I) - |F{A*e^{j*phi}}|||^2\n'
            'Classical limit of path integral: stationary phase = least action.\n'
            'GS = finding the stationary phase of the optical coherence action.\n'
            'Each GS iteration = one step of steepest descent on the action landscape.\n'
            'Feedback coherence: same as imaginary-time path integral (= energy minimization).'
        ),
        'particles_as_field_excitations': {
            'photon': 'excitation of A_mu(x,t): the EM 4-potential you know from Maxwell',
            'electron': 'excitation of psi(x,t): Dirac spinor field',
            'Higgs': 'excitation of phi(x,t): scalar field; nonzero VEV = symmetry breaking',
            'phonon': 'excitation of atomic displacement field in solid (not fundamental, but same math)',
            'plasmon': 'excitation of electron density field in metal (same math -> plasmonic photonics)',
        },
    }


# ============================================================
# 7-Hour Theory-First Study Plan
# ============================================================

def seven_hour_study_plan():
    """
    7-hour reading + writing plan: theory before engineering.
    Sequence: curl/div -> particle physics -> QFT -> connections.
    Each block includes what to READ, what to WRITE (code/notes), and the key insight.
    """
    plan = [
        {
            'hour': '0:00-1:00',
            'topic': 'Vector Calculus + Field Theory (curl, div, grad)',
            'read': [
                'Griffiths E&M Ch 1-2 (vector calculus review)',
                'dgs/vector_calculus.py: curl_demo(), divergence_demo()',
                'Key: div(E) = rho/eps0  is a CAUSAL PDE -- charge causes field',
                'Key: curl(E) = -dB/dt  is a CAUSAL PDE -- changing B causes E',
            ],
            'write': [
                'dgs/vector_calculus.py: run demo(), observe complex field dark side',
                'Sketch: draw E field of a point charge using maxwell_vector_field()',
                'Notebook cell: show Re[E] and Im[E] of a plane wave side by side',
            ],
            'key_insight': 'The electromagnetic field has a real (measured) and imaginary (phase) part. GS recovers the imaginary part.',
        },
        {
            'hour': '1:00-2:00',
            'topic': 'Standard Model Particles (this file)',
            'read': [
                'THIS FILE: standard_model_particles() -- quarks, leptons, bosons',
                'Griffiths Introduction to Elementary Particles Ch 1-2',
                'Key: 3 generations of fermions, 4 forces, 17 particles',
                'Key: symmetry group SU(3)xSU(2)xU(1)',
            ],
            'write': [
                'Run standard_model_particles() and print forces table',
                'Draw the Standard Model table from memory (12 fermions + 5 bosons)',
                'Note: which particles interact via which force?',
            ],
            'key_insight': 'Forces ARISE from demanding local gauge symmetry. The photon exists because of U(1) symmetry.',
        },
        {
            'hour': '2:00-3:00',
            'topic': 'Conservation Laws + Feynman Diagrams',
            'read': [
                'THIS FILE: conservation_laws_noether() + feynman_diagrams()',
                'Griffiths Particles Ch 6-8 (Feynman rules for QED)',
                'Key: Noether theorem -- symmetry -> conserved current',
                'Key: each Feynman vertex = factor of coupling constant e (or g_s, g_w)',
            ],
            'write': [
                'Draw 3 Feynman diagrams: Compton, e+e- annihilation, beta decay',
                'Check conservation laws for each (B, L, Q, 4-momentum)',
                'Run feynman_diagrams() and plot Z resonance -- compare to resonator S21',
            ],
            'key_insight': 'Z boson resonance = Breit-Wigner = same formula as resonator S21. All resonances share the same math.',
        },
        {
            'hour': '3:00-4:00',
            'topic': 'Quantum Field Theory Preview + Path Integral',
            'read': [
                'THIS FILE: quantum_field_theory_preview()',
                'Feynman QED (popular book): chapters 1-3',
                'Key: particles = field excitations',
                'Key: path integral Z = int D[phi] exp(jS/hbar)',
                'Key: coherent state = laser = Poisson photon statistics',
            ],
            'write': [
                'Run quantum_field_theory_preview() -- plot Klein-Gordon dispersion',
                'Compare omega vs k: photon (linear) vs electron (hyperbolic)',
                'Note: same dispersion shape as optical fiber with GVD',
            ],
            'key_insight': 'GS phase retrieval is a path integral: minimize action over phase functions. Every GS iteration is a steepest-descent step.',
        },
        {
            'hour': '4:00-5:00',
            'topic': 'Cosmology + Gravity',
            'read': [
                'THIS FILE: cosmology()',
                'Ryden Introduction to Cosmology Ch 1-4',
                'Key: Friedmann equation = GR applied to homogeneous universe',
                'Key: CMB is Planck distribution at T=2.725K',
                'Key: dark matter = 27% of universe, still unknown',
            ],
            'write': [
                'Run cosmology() -- plot CMB blackbody spectrum',
                'Compare CMB to blackbody from quantum_theory_of_light()',
                'Calculate: at what frequency does CMB peak? What telescope sees it?',
            ],
            'key_insight': 'CMB power spectrum = FT of temperature correlations = same spectral analysis as GS algorithm. Cosmology uses signal processing.',
        },
        {
            'hour': '5:00-6:00',
            'topic': 'Connections: Particle Physics -> EE -> This Repo',
            'read': [
                'dgs/modern_physics.py: solid_state() -- semiconductor band gaps',
                'dgs/quantum_information.py: chern_number_2d() -- topological invariant',
                'Key: Standard Model gauge fields -> Berry phase -> Chern number -> photonic crystal',
                'Key: neutrino PMNS matrix = beam splitter = same unitary as qubit gate',
            ],
            'write': [
                'Run chern_number_2d() -- compute topological invariant for 2-band model',
                'Note: topological protection = non-local encoding = same as QEC surface code',
                'Write 1 paragraph connecting: photon -> gauge boson -> Maxwell -> this repo',
            ],
            'key_insight': 'The photon in this repo is a gauge boson. H(f)=exp(j*pi*D*f^2) describes how the gauge field propagates in a medium.',
        },
        {
            'hour': '6:00-7:00',
            'topic': 'Write / Build: graph network module OR NSF GRFP paragraph',
            'read': ['Review dgs/ module list -- what gaps remain?'],
            'write': [
                'OPTION A: Start dgs/graph_networks.py -- packet switching, each node = signal processor',
                '  Node: receives waveform, applies filter H(z), routes to neighbors',
                '  Graph Laplacian: L = D - A (diffusion on graph)',
                '  Spectral graph theory: eigenvalues of L = "frequencies" of the graph',
                '  Same FT structure as DFT but for irregular graphs',
                'OPTION B: NSF GRFP research statement paragraph:',
                '  "This work unifies H(f)=exp(j*pi*D*f^2) across photonic time-stretch,',
                '   SAR pulse compression, and InSAR phase retrieval using differentiable',
                '   optics (autograd through the forward model), enabling gradient-based',
                '   optimization of the GS algorithm -- a Feynman path integral minimization',
                '   on the space of optical phase functions."',
            ],
            'key_insight': 'Graph Laplacian eigenvalues = spectral content of a network = same FT math applied to topology.',
        },
    ]

    return {'plan': plan, 'total_hours': 7, 'theme': 'theory_first_then_engineering'}


def demo():
    print("=== PARTICLE PHYSICS + STANDARD MODEL ===\n")

    print("--- Particles ---")
    sm = standard_model_particles()
    print(f"  Total particles: {sm['total_particles']} ({sm['generations']} generations)")
    print(f"  Symmetry group: {sm['symmetry_group']}")
    for f, d in sm['forces'].items():
        print(f"  {f:20s}: {d['coupling_name']}, range={d['range']}")

    print("\n--- Field Equations ---")
    ff = four_forces_field_equations()
    print(f"  CKM unitary: {ff['CKM_unitary']}, PMNS unitary: {ff['PMNS_unitary']}")
    print(f"  {ff['PMNS_beam_splitter_analogy'][:80]}...")

    print("\n--- Feynman Diagrams ---")
    fd = feynman_diagrams()
    print(f"  Z resonance peak ~{max(fd['Z_resonance']['sigma_Z_nb']):.1f} nb at {fd['Z_resonance']['M_Z_GeV']} GeV")
    print(f"  Breit-Wigner = resonator S21: {fd['Breit_Wigner_resonator_analogy'][:70]}...")

    print("\n--- Conservation Laws ---")
    cl = conservation_laws_noether()
    print(f"  CPT: {cl['CPT']}")
    print(f"  CP violation: {cl['CP_violation']['CP_phase_deg']} degrees")

    print("\n--- Cosmology ---")
    cosm = cosmology()
    print(f"  H0 = {cosm['H0_km_s_Mpc']} km/s/Mpc -> t_Hubble = {cosm['t_Hubble_Gyr']:.2f} Gyr")
    print(f"  T_CMB = {cosm['T_CMB_K']} K -> lambda_peak = {cosm['lambda_peak_mm']:.3f} mm" if 'lambda_peak_mm' in cosm else f"  lambda_CMB = {cosm['lambda_CMB_peak_mm']:.3f} mm")
    eb = cosm['energy_budget']
    print(f"  Universe: {eb['dark_energy_pct']}% DE + {eb['dark_matter_pct']}% DM + {eb['baryons_pct']}% baryons")

    print("\n--- QFT Preview ---")
    qft = quantum_field_theory_preview()
    cs = qft['coherent_state_photon']
    print(f"  Laser coherent state: mean_n = {cs['mean_n']:.0f} photons, var = {cs['variance_n']:.0f} (Poisson)")
    print(f"  Cosmological constant problem: off by {qft['vacuum_energy']['ratio']:.0e}")
    print(f"  GS = path integral: {qft['GS_path_integral'][:80]}...")

    print("\n--- 7-Hour Study Plan ---")
    plan = seven_hour_study_plan()
    for block in plan['plan']:
        print(f"  {block['hour']}: {block['topic']}")
        print(f"    KEY: {block['key_insight'][:70]}...")

    print("\n=== PARTICLE PHYSICS COMPLETE ===")


if __name__ == '__main__':
    demo()
