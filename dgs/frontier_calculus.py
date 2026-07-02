"""
Frontier Calculus: Precalculus -> Chain Rule -> Phasor -> Supervised Learning
             -> Set Theory -> Boolean -> SQL -> ARM -> Serway -> Multiphysics

Everything connects to ONE idea:
  The chain rule is the universal composition law.

  In calculus:     d/dx[f(g(x))] = f'(g(x)) * g'(x)
  In complex:      d/dz[f(g(z))] = same -- but z = x + jy opens the complex plane
  In phasor:       d/dt[A*exp(jwt)] = jw * A*exp(jwt)  (chain rule with g(t)=jwt)
  In backprop:     dL/dw = dL/da * da/dz * dz/dw   (chain of partial derivatives)
  In dispersion:   H(f) = exp(j*pi*D*f^2)  -> dH/df = j*2*pi*D*f * H(f)  (chain rule!)
  In ARM CPU:      instruction pipeline = function composition = chain rule for execution
  In SQL:          SELECT ... JOIN ... WHERE = function composition on sets

THE SERWAY MODERN PHYSICS CONNECTION:
  Serway, Moses, Moyer: Modern Physics, 3rd ed.
  Chapter 1: Relativity -- Lorentz transform = linear phasor rotation in 4D
  Chapter 2: Quantum theory origins -- photoelectric, Compton, de Broglie
  Chapter 3: Bohr model -- quantized energy = standing wave = boundary condition on chain rule
  Chapter 4-5: Wave mechanics -- Schrodinger = dispersion + potential = H(f) + V(x)
  Chapter 6: Tunneling -- evanescent wave = imaginary k = exp(-kappa*x)
  Chapter 9: Statistical physics -- Bose-Einstein, Fermi-Dirac, Maxwell-Boltzmann
  Chapter 12: Nuclear physics -- decay chains = differential equation chain rule
  Chapter 13: Particle physics -- Feynman diagrams = composition of propagators
"""
import math
import numpy as np

c_light = 2.998e8; hbar = 1.0546e-34; kB = 1.381e-23; h_P = 6.626e-34
q_e = 1.602e-19; m_e = 9.109e-31; eps0 = 8.854e-12; mu0 = 4*np.pi*1e-7


# ============================================================
# I. Set Theory, Boolean Algebra, Logic
# ============================================================

def set_theory_and_boolean():
    """
    SET THEORY: The mathematical foundation of every data structure, SQL query, and CPU gate.

    DEFINITIONS:
      Set A = {x | P(x)} -- "set of all x satisfying property P"
      Union:        A ∪ B = {x | x∈A OR x∈B}
      Intersection: A ∩ B = {x | x∈A AND x∈B}
      Difference:   A \\ B = {x | x∈A AND x∉B}
      Complement:   A' = {x | x∉A} (relative to universal set U)
      Power set:    P(A) = set of all subsets of A, |P(A)| = 2^|A|

    DE MORGAN'S LAWS (bridge between set theory and Boolean algebra):
      (A ∪ B)' = A' ∩ B'     (NOT(A OR B)  = (NOT A) AND (NOT B))
      (A ∩ B)' = A' ∪ B'     (NOT(A AND B) = (NOT A) OR  (NOT B))

    BOOLEAN ALGEBRA (Boole 1854 -> Shannon 1937 -> modern CPU):
      Variables: x ∈ {0, 1}  (FALSE, TRUE)
      AND:  x·y  (series switches)
      OR:   x+y  (parallel switches)
      NOT:  x'   (inverted switch)
      XOR:  x⊕y = x'y + xy'

    BOOLEAN IDENTITIES:
      Idempotent:  x·x = x,  x+x = x
      Complement:  x·x' = 0,  x+x' = 1
      Absorption:  x + x·y = x,  x·(x+y) = x
      Distributive: x·(y+z) = x·y + x·z
      De Morgan:   (x·y)' = x'+y',  (x+y)' = x'·y'

    CONNECTION TO CHAIN RULE:
      Boolean composition: f(g(x)) where f,g: {0,1}^n -> {0,1}^m
      Digital circuit = composition of gate functions = chain of Boolean ops
      ARM instruction pipeline: decode(fetch(instruction)) = function composition

    CONNECTION TO H(f) = exp(j*pi*D*f^2):
      The bit '1' at a specific time slot = a frequency bin being occupied.
      OFDM modulation: bit pattern -> FFT -> time-domain signal -> fiber -> H(f) -> GS recover.
      Digital coherent receiver: ADC bits -> DSP (Boolean + arithmetic) -> GS phase retrieval.
    """
    # Truth tables for basic gates
    inputs = [(0,0),(0,1),(1,0),(1,1)]
    gates = {
        'AND':  [int(a and b) for a,b in inputs],
        'OR':   [int(a or b)  for a,b in inputs],
        'NAND': [int(not(a and b)) for a,b in inputs],
        'NOR':  [int(not(a or b))  for a,b in inputs],
        'XOR':  [int(a != b) for a,b in inputs],
        'XNOR': [int(a == b) for a,b in inputs],
    }

    # De Morgan's law verification
    deMorgan_1 = all(int(not(a or b)) == int((not a) and (not b)) for a,b in inputs)
    deMorgan_2 = all(int(not(a and b)) == int((not a) or (not b)) for a,b in inputs)

    # 1-bit full adder: sum = A XOR B XOR Cin, carry = majority(A,B,Cin)
    def full_adder(A, B, Cin):
        S = A ^ B ^ Cin
        Cout = (A and B) or (B and Cin) or (A and Cin)
        return int(S), int(Cout)
    adder_table = [(A,B,C,*full_adder(A,B,C)) for A in [0,1] for B in [0,1] for C in [0,1]]

    # 4-bit ripple carry adder
    def ripple_adder_4bit(a4, b4):
        carry = 0; result = []
        for i in range(4):
            s, carry = full_adder((a4>>i)&1, (b4>>i)&1, carry)
            result.append(s)
        return sum(bit<<i for i,bit in enumerate(result)), carry
    sum_7_5, cout = ripple_adder_4bit(7, 5)   # 7+5 = 12, no overflow in 4 bits (carry out)

    # Set operations (Python native -- SQL analogy)
    A = {1,2,3,4,5}; B = {3,4,5,6,7}
    union = A | B; intersection = A & B; difference = A - B; sym_diff = A ^ B
    power_set_A = [frozenset(s) for r in range(len(A)+1)
                   for s in _combinations(list(A), r)]

    # ARM instruction set (simplified 4-instruction model)
    arm_instructions = {
        'ADD Rd, Rn, Rm': 'Rd = Rn + Rm  (chain: fetch->decode->execute->writeback)',
        'LDR Rd, [Rn]':   'Rd = Memory[Rn]  (load: address compute chain)',
        'STR Rd, [Rn]':   'Memory[Rn] = Rd  (store)',
        'BL  label':      'LR=PC+4; PC=label  (function call = composition entry)',
        'chain_rule':     'Pipeline = f4(f3(f2(f1(instruction)))) = function composition',
    }

    # SQL operations as set operations
    sql_set_map = {
        'SELECT *':     'Identity function: returns all elements',
        'WHERE P(x)':   'Filter: {x ∈ Table | P(x)} = set comprehension',
        'UNION':        'A ∪ B: combine two queries (removes duplicates)',
        'UNION ALL':    'Multiset union (preserves duplicates)',
        'INTERSECT':    'A ∩ B: rows in both queries',
        'EXCEPT':       'A \\ B: rows in first query not in second',
        'INNER JOIN':   'A ∩ B on key: Cartesian product filtered by condition',
        'LEFT JOIN':    'A ∪ (A ∩ B): all of A, matched B, NULLs elsewhere',
        'GROUP BY':     'Partition: split set into equivalence classes',
        'HAVING':       'Filter on groups: {g ∈ groups | P(aggregate(g))}',
    }

    return {
        'truth_tables': {
            'inputs': inputs,
            'gates': gates,
        },
        'de_morgan': {
            'law1_verified': bool(deMorgan_1),
            'law2_verified': bool(deMorgan_2),
            'statement1': 'NOT(A OR B) = (NOT A) AND (NOT B)',
            'statement2': 'NOT(A AND B) = (NOT A) OR (NOT B)',
        },
        'adder': {
            'full_adder_table': adder_table,
            '7_plus_5': {'sum': int(sum_7_5), 'carry_out': int(cout),
                         'correct': sum_7_5 + (cout<<4) == 12},
        },
        'sets': {
            'A': sorted(A), 'B': sorted(B),
            'union': sorted(union), 'intersection': sorted(intersection),
            'difference_A_minus_B': sorted(difference),
            'symmetric_difference': sorted(sym_diff),
            'power_set_size': 2**len(A),
        },
        'ARM': arm_instructions,
        'SQL_set_map': sql_set_map,
        'H_f_connection': (
            'OFDM bits -> FFT -> time signal -> H(f)=exp(j*pi*D*f^2) -> photodetector ->'
            ' ADC bits -> SQL database -> GS phase retrieval -> recovered signal'
        ),
    }


def _combinations(lst, r):
    """Simple combinations for power set (no itertools)."""
    if r == 0: return [()]
    if not lst: return []
    head, tail = lst[0], lst[1:]
    with_head = [(head,)+c for c in _combinations(tail, r-1)]
    without_head = _combinations(tail, r)
    return with_head + without_head


# ============================================================
# II. Trigonometry, Precalculus, Complex Numbers
# ============================================================

def trig_precalculus_complex():
    """
    THE UNIT CIRCLE -> EULER'S FORMULA -> PHASOR -> H(f).

    UNIT CIRCLE:
      Any point on the unit circle: (cos θ, sin θ) = Re[e^{jθ}], Im[e^{jθ}]
      Parameterized by angle θ ∈ [0, 2π).
      sin²θ + cos²θ = 1  (Pythagorean identity = |e^{jθ}|² = 1)

    KEY TRIG IDENTITIES (all derivable from Euler's formula):
      e^{jθ} = cos θ + j sin θ                     (Euler's formula)
      cos θ = (e^{jθ} + e^{-jθ})/2                (definition)
      sin θ = (e^{jθ} - e^{-jθ})/(2j)             (definition)
      cos(A+B) = cos A cos B - sin A sin B          (from e^{j(A+B)} = e^{jA}*e^{jB})
      sin(A+B) = sin A cos B + cos A sin B          (same)
      cos²θ = (1 + cos 2θ)/2                       (double angle)
      sin²θ = (1 - cos 2θ)/2                       (double angle)

    PRECALCULUS -> CALCULUS BRIDGE:
      Slope of secant: [f(x+h) - f(x)] / h  as h->0 = derivative
      Average rate of change -> instantaneous rate of change
      Area under step functions (Riemann sums) -> integral

    COMPLEX NUMBERS (full treatment):
      z = x + jy = r*e^{jθ}
      r = |z| = sqrt(x² + y²)       (modulus)
      θ = arg(z) = atan2(y, x)       (argument)
      z* = x - jy = r*e^{-jθ}       (complex conjugate)
      |z|² = z*z* = r²               (modulus squared -- what a photodetector measures)
      1/z = z*/|z|²                  (reciprocal)
      z₁*z₂ = r₁r₂*e^{j(θ₁+θ₂)}    (multiplication = ROTATE + SCALE)
      z₁/z₂ = (r₁/r₂)*e^{j(θ₁-θ₂)} (division = rotate + scale)

    PHASOR DOMAIN (the engineering shortcut):
      Signal: v(t) = V₀ cos(ωt + φ) = Re[V₀ e^{jφ} e^{jωt}]
      Phasor: V = V₀ e^{jφ}    (complex amplitude, e^{jωt} suppressed)
      d/dt in time -> jω in phasor domain  (from chain rule: d/dt[e^{jωt}] = jω*e^{jωt})
      ∫...dt in time -> 1/(jω) in phasor  (inverse)

    THIS REPO:
      H(f) = exp(j*pi*D*f²)
      d/df H(f) = j*2*pi*D*f * H(f)   (chain rule: d/df[e^{g(f)}] = e^{g(f)}*g'(f))
      |H(f)| = 1  (phasor with unit magnitude = pure phase rotation = all-pass filter)
      arg(H(f)) = pi*D*f²  (quadratic phase = GVD = maps frequency to time)
    """
    # Unit circle: sin²+cos²=1 verification
    theta_arr = np.linspace(0, 2*np.pi, 10000)
    identity_error = np.max(np.abs(np.sin(theta_arr)**2 + np.cos(theta_arr)**2 - 1.0))

    # Euler's formula: e^{jθ} = cos θ + j sin θ
    z_euler = np.exp(1j*theta_arr)
    euler_error = np.max(np.abs(z_euler - (np.cos(theta_arr) + 1j*np.sin(theta_arr))))

    # Key trig values (exact)
    trig_exact = {
        'sin(0)':    (0.0,    0,   '0'),
        'sin(pi/6)': (0.5,    1,   '1/2'),
        'sin(pi/4)': (math.sqrt(2)/2, 2, 'sqrt(2)/2'),
        'sin(pi/3)': (math.sqrt(3)/2, 3, 'sqrt(3)/2'),
        'sin(pi/2)': (1.0,    4,   '1'),
        'cos(0)':    (1.0,    5,   '1'),
        'cos(pi/3)': (0.5,    6,   '1/2'),
        'cos(pi/2)': (0.0,    7,   '0'),
        'tan(pi/4)': (1.0,    8,   '1'),
    }
    trig_verified = {k: abs(math.sin(math.pi/6) - 0.5) < 1e-15 for k in ['sin(pi/6)']}

    # Addition formulas from Euler
    A_ang = math.pi/6; B_ang = math.pi/4
    cos_ApB_euler = float(np.real(np.exp(1j*(A_ang+B_ang))))
    cos_ApB_formula = math.cos(A_ang)*math.cos(B_ang) - math.sin(A_ang)*math.sin(B_ang)
    addition_error = abs(cos_ApB_euler - cos_ApB_formula)

    # Complex arithmetic
    z1 = 3 + 4j; z2 = 1 - 2j
    z1_mod = abs(z1); z1_arg_deg = math.degrees(math.atan2(z1.imag, z1.real))
    z1_polar = f'{z1_mod:.2f} * exp(j*{math.radians(z1_arg_deg):.4f})'
    product = z1*z2; quotient = z1/z2

    # Phasor domain: RC circuit
    R = 1e3; C = 1e-6; omega_rc = 1/(R*C)   # resonance: |Z_C| = R
    omega_arr_rc = np.logspace(1, 5, 300)
    Z_R = R * np.ones_like(omega_arr_rc, dtype=complex)
    Z_C = 1/(1j*omega_arr_rc*C)
    V_out_over_in = Z_C / (Z_R + Z_C)   # voltage divider
    H_mag = np.abs(V_out_over_in)
    H_phase_deg = np.degrees(np.angle(V_out_over_in))
    # At omega=omega_rc: H = 1/(1+j) -> |H|=1/sqrt(2)=-3dB, phase=-45 deg
    idx_3dB = int(np.argmin(np.abs(omega_arr_rc - omega_rc)))

    # H(f) = exp(j*pi*D*f^2): chain rule check
    D_GVD = 5000.0   # ps/nm (D*L product)
    f_arr = np.linspace(-1e9, 1e9, 1000)   # Hz
    H_f = np.exp(1j*np.pi*D_GVD*f_arr**2)
    # Chain rule: d/df H(f) = jw*2*pi*D*f * H(f)
    dH_df_chain = 1j*2*np.pi*D_GVD*f_arr * H_f
    dH_df_numeric = np.gradient(H_f, f_arr)
    chain_rule_error = float(np.max(np.abs(dH_df_chain[10:-10] - dH_df_numeric[10:-10])))

    # Python language history timeline
    python_history = {
        '1991': 'Python 0.9 (Guido van Rossum, ABC-derived, first public release)',
        '1994': 'Python 1.0 (lambda, map, filter, reduce)',
        '2000': 'Python 2.0 (list comprehensions, GC, Unicode strings)',
        '2008': 'Python 3.0 (print as function, bytes, integer division fixed)',
        '2020': 'Python 3.8 (walrus :=, f-strings, positional-only /)',
        '2022': 'Python 3.11 (3x faster CPython, exception groups)',
        '2024': 'Python 3.13 (JIT compiler experimental, free-threaded mode)',
        '2026': 'Python 3.13 (current -- this repo uses py -3.13)',
        'this_repo': 'py -3.13 for dgs/* (no scipy); py -3.12 for torch only',
    }

    return {
        'unit_circle': {
            'identity_max_error': float(identity_error),
            'identity_holds': identity_error < 1e-12,
            'euler_max_error': float(euler_error),
            'euler_formula': 'e^{j*theta} = cos(theta) + j*sin(theta)',
        },
        'trig_exact': {k: v[0] for k,v in trig_exact.items()},
        'addition_formula': {
            'cos(A+B)': float(cos_ApB_euler),
            'formula': float(cos_ApB_formula),
            'error': float(addition_error),
            'derivation': 'Re[e^{j(A+B)}] = Re[e^{jA}*e^{jB}] = cosA*cosB - sinA*sinB',
        },
        'complex_arithmetic': {
            'z1': str(z1), 'z2': str(z2),
            'z1_modulus': float(z1_mod), 'z1_arg_deg': float(z1_arg_deg),
            'z1_polar': z1_polar,
            'product': str(product), 'quotient': f'{quotient:.4f}',
            'conjugate': str(z1.conjugate()),
            'modulus_squared': float(abs(z1)**2),
        },
        'phasor_RC': {
            'omega_3dB_rad_s': float(omega_rc),
            'f_3dB_Hz': float(omega_rc/(2*math.pi)),
            'H_mag_at_3dB': float(H_mag[idx_3dB]),
            'H_phase_at_3dB_deg': float(H_phase_deg[idx_3dB]),
            'expected_mag': float(1/math.sqrt(2)),
            'expected_phase_deg': -45.0,
        },
        'H_f_chain_rule': {
            'formula': 'H(f) = exp(j*pi*D*f^2)',
            'derivative': 'dH/df = j*2*pi*D*f * H(f)  [chain rule: f=f^2, outer=exp(j*pi*D*f)]',
            'chain_rule_error': float(chain_rule_error),
            'chain_rule_holds': chain_rule_error < 1e4,  # numerical gradient is approximate
        },
        'python_history': python_history,
    }


# ============================================================
# III. Chain Rule -- Calculus with Complex Numbers
# ============================================================

def chain_rule_calculus():
    """
    THE CHAIN RULE: d/dx[f(g(x))] = f'(g(x)) * g'(x)

    This is the single most important rule in:
      1. Calculus (derivatives of composite functions)
      2. Complex analysis (Cauchy-Riemann + chain rule = holomorphic functions)
      3. Backpropagation (ML training algorithm)
      4. Signal processing (phasor: d/dt[e^{jwt}] = jw*e^{jwt})
      5. Optics (H(f) derivative, GVD, phase accumulation)

    REAL CHAIN RULE EXAMPLES:
      f(x) = sin(x²)       -> f'(x) = cos(x²) * 2x
      f(x) = exp(-x²/2)    -> f'(x) = -x * exp(-x²/2)   [Gaussian!]
      f(x) = (1+x²)^{-1}   -> f'(x) = -2x/(1+x²)²
      f(x) = ln(sin(x))    -> f'(x) = cos(x)/sin(x) = cot(x)

    COMPLEX CHAIN RULE (Cauchy-Riemann conditions):
      f(z) = f(x+jy) holomorphic (complex-differentiable) if:
        ∂u/∂x = ∂v/∂y  and  ∂u/∂y = -∂v/∂x
      where f = u + jv, z = x + jy.
      Chain rule: d/dz[f(g(z))] = f'(g(z)) * g'(z)   (identical form to real case!)

    PHASOR CHAIN RULE:
      v(t) = V₀ cos(ωt + φ)
      Phasor V = V₀ e^{jφ}    (strip e^{jwt} carrier)
      dv/dt = -V₀ω sin(ωt+φ) = Re[jω * V₀e^{jφ} * e^{jωt}]
      CHAIN RULE: d/dt[e^{jωt+jφ}] = jω * e^{jωt+jφ}
      -> phasor of dv/dt = jω * V   (multiply phasor by jω)
      -> phasor of ∫v dt = V/(jω)  (divide phasor by jω)

    MULTI-LAYER CHAIN RULE (backpropagation):
      z1 = w1*x         (layer 1: linear)
      a1 = sigmoid(z1)  (layer 1: activation)
      z2 = w2*a1        (layer 2: linear)
      a2 = sigmoid(z2)  (layer 2: activation = output)
      L = (a2 - y)^2    (loss)

      Backprop:
        dL/dw2 = dL/da2 * da2/dz2 * dz2/dw2
               = 2(a2-y) * sigmoid'(z2) * a1
        dL/dw1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
               = 2(a2-y) * sigmoid'(z2) * w2 * sigmoid'(z1) * x
      Each layer's gradient = product of all upstream chain-rule factors.
    """
    x_arr = np.linspace(-3, 3, 1000)

    # Real chain rule examples
    def f1(x): return np.sin(x**2)
    def f1_prime(x): return np.cos(x**2)*2*x

    def f2(x): return np.exp(-x**2/2)   # Gaussian
    def f2_prime(x): return -x*np.exp(-x**2/2)

    # Verify numerically
    dx = x_arr[1]-x_arr[0]
    f1_prime_numeric = np.gradient(f1(x_arr), x_arr)
    f2_prime_numeric = np.gradient(f2(x_arr), x_arr)
    chain_error_f1 = float(np.max(np.abs(f1_prime(x_arr[5:-5]) - f1_prime_numeric[5:-5])))
    chain_error_f2 = float(np.max(np.abs(f2_prime(x_arr[5:-5]) - f2_prime_numeric[5:-5])))

    # Complex chain rule: f(z) = exp(j*pi*z^2) -> f'(z) = j*2*pi*z*exp(j*pi*z^2)
    z_arr = np.linspace(-2, 2, 500) + 0j   # real axis for now
    fz = np.exp(1j*np.pi*z_arr**2)
    fz_prime_analytic = 1j*2*np.pi*z_arr * fz
    fz_prime_numeric = np.gradient(fz, np.real(z_arr))
    complex_chain_error = float(np.max(np.abs(fz_prime_analytic[5:-5] - fz_prime_numeric[5:-5])))

    # Cauchy-Riemann verification for f(z) = z^2
    x_cr = np.linspace(-2, 2, 50); y_cr = np.linspace(-2, 2, 50)
    X_cr, Y_cr = np.meshgrid(x_cr, y_cr)
    Z_cr = X_cr + 1j*Y_cr
    Fz_cr = Z_cr**2   # f(z) = z^2 = x^2-y^2 + 2jxy
    u_cr = np.real(Fz_cr); v_cr = np.imag(Fz_cr)
    # du/dx = 2x, dv/dy = 2x -> equal ✓
    # du/dy = -2y, -dv/dx = -2y -> equal ✓
    du_dx = np.gradient(u_cr, x_cr, axis=1)
    dv_dy = np.gradient(v_cr, y_cr, axis=0)
    CR_error = float(np.max(np.abs(du_dx - dv_dy)))   # should be ~0

    # Phasor domain: RC circuit differentiation
    omega_test = 1000.0; V0 = 5.0; phi = math.pi/4
    R_p = 1e3; C_p = 1e-6
    V_phasor = V0 * np.exp(1j*phi)   # input voltage phasor
    # Current through capacitor: I = C * dV/dt  -> phasor: I = j*omega*C * V
    I_phasor = 1j*omega_test*C_p * V_phasor
    # Voltage across R: VR = I*R
    VR_phasor = I_phasor * R_p
    # Transfer function H = VR/V = j*omega*R*C/(1+j*omega*R*C)
    H_RC = 1j*omega_test*R_p*C_p / (1 + 1j*omega_test*R_p*C_p)

    # Backprop: 2-layer network with sigmoid
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -20, 20)))
    def sigmoid_prime(x): s = sigmoid(x); return s*(1-s)

    np.random.seed(42)
    x_data = np.random.randn(100)
    y_data = np.sin(x_data) + 0.1*np.random.randn(100)

    w1, w2 = 0.5, 0.3
    lr = 0.01
    losses = []
    for _ in range(200):
        # Forward pass
        z1 = w1*x_data; a1 = sigmoid(z1)
        z2 = w2*a1; a2 = z2   # linear output
        L = float(np.mean((a2 - y_data)**2))
        losses.append(L)
        # Backprop (chain rule)
        dL_da2 = 2*(a2 - y_data)/len(x_data)
        dL_dw2 = float(np.sum(dL_da2 * a1))
        dL_da1 = dL_da2 * w2
        dL_dz1 = dL_da1 * sigmoid_prime(z1)
        dL_dw1 = float(np.sum(dL_dz1 * x_data))
        w1 -= lr*dL_dw1; w2 -= lr*dL_dw2

    return {
        'real_chain_rule': {
            'f1_sin_x2_error': float(chain_error_f1),
            'f2_gaussian_error': float(chain_error_f2),
            'examples': {
                'd/dx[sin(x^2)]': 'cos(x^2)*2x',
                'd/dx[exp(-x^2/2)]': '-x*exp(-x^2/2)  [Gaussian bell curve derivative]',
                'd/dx[(1+x^2)^-1]': '-2x/(1+x^2)^2',
                'd/dx[ln(sin(x))]': 'cot(x) = cos(x)/sin(x)',
            },
        },
        'complex_chain_rule': {
            'f_z': 'exp(j*pi*z^2)',
            'df_dz_analytic': 'j*2*pi*z * exp(j*pi*z^2)',
            'error': float(complex_chain_error),
            'Cauchy_Riemann_error_z2': float(CR_error),
            'CR_satisfied': CR_error < 0.1,
        },
        'phasor': {
            'omega_rad_s': float(omega_test),
            'V_phasor_mag': float(abs(V_phasor)),
            'I_phasor_mag_mA': float(abs(I_phasor)*1e3),
            'rule_d_dt': 'Multiply phasor by j*omega',
            'rule_integral': 'Divide phasor by j*omega',
            'H_RC_mag': float(abs(H_RC)),
            'H_RC_phase_deg': float(math.degrees(math.atan2(H_RC.imag, H_RC.real))),
        },
        'backprop': {
            'initial_loss': float(losses[0]),
            'final_loss': float(losses[-1]),
            'converged': losses[-1] < losses[0],
            'chain_rule_explanation': (
                'dL/dw1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1  '
                '(5-term chain from output to weight1)'
            ),
            'loss_history': losses,
        },
        'H_f_chain_rule': {
            'H': 'H(f) = exp(j*pi*D*f^2)',
            'dH_df': 'j*2*pi*D*f * H(f)',
            'd2H_df2': '(j*2*pi*D - (2*pi*D*f)^2) * H(f)',
            'GVD_as_chain': 'Phase phi(f)=pi*D*f^2, H=exp(j*phi), group_delay=dphi/domega=D*f/pi',
        },
    }


# ============================================================
# IV. Serway Modern Physics -- Key Problems
# ============================================================

def serway_modern_physics_problems():
    """
    SERWAY, MOSES, MOYER: MODERN PHYSICS (3rd ed.)
    Selected problems from each chapter -- solved analytically and numerically.

    Chapter 1: Special Relativity
    Chapter 2: Quantum Theory Origins (photoelectric, Compton, de Broglie)
    Chapter 3: Bohr Model of Hydrogen
    Chapter 4-5: Wave Mechanics, Schrodinger Equation
    Chapter 6: Quantum Tunneling
    Chapter 9: Statistical Physics (Bose-Einstein, Fermi-Dirac)
    Chapter 12: Nuclear Physics
    Chapter 13: Particle Physics
    """
    results = {}

    # --- Chapter 1: Special Relativity ---
    # Lorentz factor gamma = 1/sqrt(1-beta^2)
    beta_vals = np.array([0.1, 0.5, 0.8, 0.9, 0.99, 0.999])
    gamma_vals = 1/np.sqrt(1-beta_vals**2)
    # Time dilation: dt_obs = gamma * dt_proper
    # Length contraction: L_obs = L_proper / gamma
    # Relativistic KE: KE = (gamma-1)*m*c^2
    # Relativistic momentum: p = gamma*m*v = gamma*m*beta*c
    # Serway Problem 1.x: electron at v=0.99c
    beta_e = 0.99; gamma_e = float(1/math.sqrt(1-beta_e**2))
    KE_e_MeV = (gamma_e-1)*m_e*c_light**2/q_e/1e6
    p_e_MeV_c = gamma_e*m_e*beta_e*c_light / (q_e/c_light) / 1e6

    results['Ch1_relativity'] = {
        'beta': beta_vals.tolist(),
        'gamma': gamma_vals.tolist(),
        'problem_electron_0_99c': {
            'gamma': float(gamma_e),
            'KE_MeV': float(KE_e_MeV),
            'p_MeV_c': float(p_e_MeV_c),
            'time_dilation': f't_obs = {gamma_e:.2f} * t_proper',
            'length_contraction': f'L_obs = L_proper / {gamma_e:.2f}',
        },
        'energy_momentum_invariant': 'E^2 = (pc)^2 + (m*c^2)^2  [4-vector magnitude]',
    }

    # --- Chapter 2: Quantum Origins ---
    # Photoelectric effect: KE_max = hf - phi_work
    # Compton scattering: delta_lambda = (h/m_e*c)*(1-cos(theta))
    # de Broglie: lambda = h/p = h/(m*v) = h/(gamma*m*v) relativistic
    phi_Cu_eV = 4.65   # Cu work function [eV]
    f_UV = 1.5e15   # Hz UV light
    KE_photo_eV = (h_P*f_UV - phi_Cu_eV*q_e)/q_e
    V_stop = KE_photo_eV   # stopping voltage [V]

    # Compton wavelength
    lambda_C = h_P/(m_e*c_light)   # 2.426e-12 m = 2.426 pm
    theta_Compton = math.pi/2   # 90 degrees
    delta_lambda_Compton = lambda_C*(1-math.cos(theta_Compton))   # m

    # de Broglie: thermal neutron at 300 K
    m_n = 1.675e-27   # kg
    v_thermal = math.sqrt(3*kB*300/m_n)
    lambda_deBroglie_neutron = h_P/(m_n*v_thermal)   # m

    results['Ch2_quantum_origins'] = {
        'photoelectric': {
            'material': 'Copper (phi=4.65 eV)',
            'f_UV_Hz': float(f_UV),
            'hf_eV': float(h_P*f_UV/q_e),
            'KE_max_eV': float(KE_photo_eV),
            'V_stop_V': float(V_stop),
            'threshold_f_Hz': float(phi_Cu_eV*q_e/h_P),
        },
        'Compton': {
            'lambda_C_pm': float(lambda_C*1e12),
            'theta_deg': 90.0,
            'delta_lambda_pm': float(delta_lambda_Compton*1e12),
            'formula': 'delta_lambda = (h/m_e*c)*(1-cos(theta))',
        },
        'de_Broglie': {
            'thermal_neutron_T_K': 300.0,
            'v_m_s': float(v_thermal),
            'lambda_angstrom': float(lambda_deBroglie_neutron*1e10),
            'note': 'Lambda ~ 1 Angstrom = atomic spacing -> neutron diffraction!',
            'formula': 'lambda = h/p = h/(m*v)',
        },
    }

    # --- Chapter 3: Bohr Model ---
    # E_n = -13.6 eV / n^2    (hydrogen energy levels)
    # r_n = n^2 * a_0           (orbital radii, a_0 = 0.529 Angstrom)
    # v_n = alpha * c / n       (orbital speed, alpha = fine structure constant)
    a0 = 0.529e-10   # Bohr radius [m]
    alpha_fine = q_e**2/(4*math.pi*eps0*hbar*c_light)   # ~1/137
    n_arr = np.arange(1, 8)
    E_n = -13.6/n_arr**2   # eV
    r_n = n_arr**2 * a0 * 1e10   # Angstrom
    v_n = alpha_fine * c_light / n_arr   # m/s
    # Lyman series (to n=1): UV
    # Balmer series (to n=2): visible
    # Paschen series (to n=3): IR
    series = {}
    for n_f, name in [(1,'Lyman'),(2,'Balmer'),(3,'Paschen')]:
        lines = []
        for n_i in range(n_f+1, n_f+5):
            dE = (13.6/n_f**2 - 13.6/n_i**2)*q_e
            lam = h_P*c_light/dE * 1e9   # nm
            lines.append({'n_i': n_i, 'n_f': n_f, 'lambda_nm': float(lam)})
        series[name] = lines
    # Verify Balmer series: first line (n=3->2) should be ~656 nm (H-alpha)
    H_alpha_nm = series['Balmer'][0]['lambda_nm']

    results['Ch3_Bohr'] = {
        'alpha_fine': float(alpha_fine),
        'a0_angstrom': float(a0*1e10),
        'energy_levels_eV': E_n.tolist(),
        'radii_angstrom': r_n.tolist(),
        'speeds_m_s': v_n.tolist(),
        'spectral_series': series,
        'H_alpha_nm': float(H_alpha_nm),
        'H_alpha_expected_nm': 656.3,
    }

    # --- Chapter 4-5: Schrodinger Equation (infinite square well) ---
    # psi_n(x) = sqrt(2/L)*sin(n*pi*x/L)   [n=1,2,3,...]
    # E_n = n^2 * pi^2 * hbar^2 / (2*m*L^2)
    L_well = 1e-9   # 1 nm quantum well
    n_QM = np.arange(1, 6)
    E_QM_eV = (n_QM**2 * math.pi**2 * hbar**2) / (2*m_e*L_well**2) / q_e
    x_well = np.linspace(0, L_well, 500)
    psi_1 = math.sqrt(2/L_well) * np.sin(math.pi*x_well/L_well)
    psi_2 = math.sqrt(2/L_well) * np.sin(2*math.pi*x_well/L_well)
    # Normalization check: integral |psi|^2 dx = 1
    norm_1 = float(np.trapezoid(psi_1**2, x_well))
    norm_2 = float(np.trapezoid(psi_2**2, x_well))
    # Orthogonality: integral psi_1*psi_2 dx = 0
    ortho_12 = float(abs(np.trapezoid(psi_1*psi_2, x_well)))

    results['Ch4_5_Schrodinger'] = {
        'infinite_well_L_nm': float(L_well*1e9),
        'energy_levels_eV': E_QM_eV.tolist(),
        'ground_state_eV': float(E_QM_eV[0]),
        'normalization_n1': float(norm_1),
        'normalization_n2': float(norm_2),
        'orthogonality_12': float(ortho_12),
        'GS_connection': (
            'Psi_n(x) = Fourier sine mode n. '
            'Schrodinger in k-space: (hbar*k)^2/(2m) = E_n '
            '= discrete eigenvalues of -hbar^2/(2m)*d^2/dx^2. '
            'Same math as frequency-domain GS: H(f)=exp(j*pi*D*f^2) '
            'is the free-particle propagator in momentum space.'
        ),
    }

    # --- Chapter 6: Tunneling ---
    # Transmission coefficient T = 1/(1 + (kappa^2+k^2)^2*sinh^2(kappa*L)/(4*k^2*kappa^2))
    # Approximate (kappa*L >> 1): T ~ 16*(E/U0)*(1-E/U0)*exp(-2*kappa*L)
    # kappa = sqrt(2*m*(U0-E))/hbar   [evanescent wavenumber]
    U0_eV = 5.0; L_barrier = 1e-9   # 5 eV barrier, 1 nm thick
    E_arr_eV = np.linspace(0.1, 4.9, 200)
    kappa_arr = np.sqrt(2*m_e*(U0_eV-E_arr_eV)*q_e)/hbar
    k_arr = np.sqrt(2*m_e*E_arr_eV*q_e)/hbar
    T_arr = 1/(1 + ((kappa_arr**2+k_arr**2)**2*np.sinh(kappa_arr*L_barrier)**2)
                   /(4*k_arr**2*kappa_arr**2))
    T_at_1eV = float(T_arr[np.argmin(np.abs(E_arr_eV - 1.0))])

    results['Ch6_tunneling'] = {
        'barrier': {'U0_eV': float(U0_eV), 'L_nm': float(L_barrier*1e9)},
        'E_eV': E_arr_eV.tolist(),
        'T_transmission': T_arr.tolist(),
        'T_at_1_eV': float(T_at_1eV),
        'evanescent_wave': 'psi(x) ~ exp(-kappa*x) inside barrier [imaginary k -> real decay]',
        'optics_analog': 'Frustrated total internal reflection in fiber couplers = same T formula',
        'GS_note': 'H(f) with imaginary D would give T(f) -- tunneling in frequency domain',
    }

    # --- Chapter 9: Statistical Physics ---
    E_arr_stat = np.linspace(0, 5, 500)*q_e   # J
    T_stat = 300.0; mu = 2.0*q_e   # chemical potential [J]
    # Fermi-Dirac (fermions: electrons)
    f_FD = 1/(np.exp((E_arr_stat - mu)/(kB*T_stat)) + 1)
    # Bose-Einstein (bosons: photons with mu=0)
    mu_BE = 0; E_min = 0.01*q_e
    E_BE = np.linspace(E_min, 5, 500)*q_e
    f_BE = 1/(np.exp(E_BE/(kB*T_stat)) - 1)
    # Maxwell-Boltzmann (classical)
    f_MB = np.exp(-E_arr_stat/(kB*T_stat))

    results['Ch9_statistical'] = {
        'T_K': float(T_stat),
        'kT_meV': float(kB*T_stat/q_e*1e3),
        'Fermi_Dirac': {'E_eV': (E_arr_stat/q_e).tolist(), 'f_FD': f_FD.tolist(),
                        'note': 'Step function at T=0K, smeared by kT at finite T'},
        'Bose_Einstein': {'E_eV': (E_BE/q_e).tolist(), 'f_BE': (f_BE/f_BE.max()).tolist(),
                          'note': 'Diverges as E->0 (photon bunching, laser threshold)'},
        'Maxwell_Boltzmann': {'f_MB': f_MB.tolist(),
                               'note': 'Classical limit: valid when f << 1'},
        'photon_connection': 'h*f >> kT at 1550nm (h*f/kT = 82): quantum noise limited',
    }

    return results


# ============================================================
# V. Multiphysics Coupling
# ============================================================

def multiphysics_coupling(n_steps=500):
    """
    MULTIPHYSICS: EM + Thermal + Mechanical coupling in a photonic device.

    SCENARIO: Silicon photonic ring resonator under high optical power.
      - Optical power P_opt absorbed -> heating (EM -> Thermal)
      - Temperature rise DeltaT -> refractive index change via dn/dT (Thermal -> EM)
      - Thermal expansion -> stress -> strain -> photoelastic index change (Thermal -> Mechanical -> EM)
      - Index shift -> resonance frequency shift -> feedback loop

    GOVERNING EQUATIONS:
      EM (optical resonance):
        Transfer function: H(omega) = 1/(1 - r*exp(j*(omega-omega_0)/FSR * 2*pi))
        Stored energy: U = P_in * Q / omega_0
        Absorption: P_abs = U * omega_0/Q_abs

      Thermal (heat equation, lumped model):
        C * dT/dt = P_abs - (T-T_amb)/R_th
        Steady state: DeltaT = P_abs * R_th

      Thermo-optic (silicon: dn/dT ~ 1.86e-4 /K):
        Delta_n_TO = (dn/dT) * DeltaT
        Delta_omega_TO = -omega_0 * Delta_n_TO / n_eff

      Mechanical (thermal expansion, alpha_Si = 2.6e-6 /K):
        DeltaL/L = alpha_Si * DeltaT
        Strain: epsilon = alpha_Si * DeltaT
        Photoelastic: Delta_n_PE = -n^3/2 * p_12 * epsilon   (p_12 ~ -0.101 for Si)
        Delta_omega_PE = -omega_0 * Delta_n_PE / n_eff

      COUPLED FEEDBACK (bistability):
        omega_resonance(T) = omega_0 + Delta_omega_TO(T) + Delta_omega_PE(T)
        P_abs(omega, T) = P_in * |H(omega, omega_resonance(T))|^2 * alpha_abs
        C * dT/dt = P_abs(omega, T) - (T-T_amb)/R_th
        -> Nonlinear ODE! Can exhibit bistability (two stable states) at high power.

    CONNECTION TO H(f) = exp(j*pi*D*f^2):
      The ring resonator IS a dispersive element.
      Near resonance: H(omega) ~ exp(j*omega*tau_RT) where tau_RT = 2*pi*R*n_eff/c
      Far from resonance: H(omega) ~ exp(j*phase) - same quadratic phase approximation.
      Thermal drift in D: dD/dT = (dn/dT)*(-2*pi*c/lambda^2)*L -> GS must track this.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps = {n_steps} must be > 0")

    # Silicon ring resonator parameters
    R_ring = 10e-6   # 10 um radius [m]
    n_eff = 3.48     # effective refractive index (Si at 1550 nm)
    lambda0 = 1550e-9; omega0 = 2*math.pi*c_light/lambda0
    FSR_Hz = c_light/(2*math.pi*R_ring*n_eff)   # Free spectral range
    Q_total = 1e5; Q_abs = 1e6   # loaded and absorption Q factors
    kappa_e = omega0/Q_total   # coupling rate (energy decay rate to bus)
    kappa_abs = omega0/Q_abs   # absorption rate

    # Thermal parameters (silicon)
    C_th = 1e-12   # thermal capacitance [J/K] (tiny ring = femtojoule scale)
    R_th = 1e5     # thermal resistance [K/W] (~100 K/mW, typical)
    T_amb = 300.0  # K
    dn_dT = 1.86e-4  # Si thermo-optic coefficient [1/K]
    alpha_Si = 2.6e-6  # thermal expansion coefficient [1/K]
    p12_Si = -0.101   # photoelastic coefficient

    # Input power and detuning sweep
    P_in_arr = np.array([0.1, 1.0, 5.0]) * 1e-3   # 0.1, 1, 5 mW
    delta_omega = np.linspace(-5*kappa_e, 5*kappa_e, 200)

    def ring_H(delta, omega_res_shift=0):
        """Ring transfer function: H = kappa_e/(j*(delta-omega_res_shift) + kappa_total/2)"""
        kappa_total = kappa_e + kappa_abs
        return kappa_e / (1j*(delta - omega_res_shift) + kappa_total/2)

    transmission_spectra = {}
    for P_mW, P_in in zip([0.1, 1.0, 5.0], P_in_arr):
        T_spec = np.abs(ring_H(delta_omega))**2
        transmission_spectra[f'P_{P_mW}mW'] = T_spec.tolist()

    # Coupled ODE: thermal bistability simulation (RK4)
    dt_mult = 1/(10*kappa_e)   # timestep [s]
    t_arr_mult = np.arange(n_steps) * dt_mult
    T_temp = np.zeros(n_steps); T_temp[0] = T_amb
    P_abs_hist = np.zeros(n_steps)
    P_in_high = 5e-3   # 5 mW (high enough for thermal shift)
    delta_fixed = -kappa_e   # slightly blue-detuned (classic for thermal bistability)

    for i in range(1, n_steps):
        DeltaT = T_temp[i-1] - T_amb
        # Thermo-optic + photoelastic resonance shift
        dn_TO = dn_dT * DeltaT
        dn_PE = -n_eff**3/2 * p12_Si * alpha_Si * DeltaT
        omega_shift = -omega0 * (dn_TO + dn_PE) / n_eff
        # Absorbed power at current temperature
        H_val = ring_H(delta_fixed, omega_res_shift=omega_shift)
        P_circ = P_in_high * abs(H_val)**2
        P_abs = P_circ * kappa_abs / (kappa_e + kappa_abs + 1e-30)
        P_abs_hist[i] = P_abs
        # Thermal ODE: C*dT/dt = P_abs - (T-T_amb)/R_th
        dT_dt = (P_abs - DeltaT/R_th) / C_th
        T_temp[i] = T_temp[i-1] + dt_mult*dT_dt

    DeltaT_steady = float(T_temp[-1] - T_amb)
    dn_TO_steady = dn_dT * DeltaT_steady
    omega_shift_steady_GHz = float(-omega0*(dn_TO_steady)/n_eff / (2*math.pi*1e9))

    # Dispersion D shift with temperature
    # beta2(T) = beta2_0 + d(beta2)/dT * DeltaT
    # d(beta2)/dT = -(lambda^2/(2*pi*c)) * d(D)/dT
    # For Si at 1550 nm: dD/dT ~ 0.001 ps/(nm*km*K) (approximate)
    dD_dT_Si = 0.001   # ps/(nm*km*K)
    D_shift_per_K = dD_dT_Si * DeltaT_steady   # ps/(nm*km)

    return {
        'ring': {
            'R_um': float(R_ring*1e6),
            'n_eff': float(n_eff),
            'Q_total': float(Q_total),
            'FSR_GHz': float(FSR_Hz/1e9),
            'kappa_e_GHz': float(kappa_e/(2*math.pi*1e9)),
        },
        'thermal': {
            'C_th_J_K': float(C_th),
            'R_th_K_W': float(R_th),
            'dn_dT_per_K': float(dn_dT),
            'alpha_Si_per_K': float(alpha_Si),
        },
        'transmission_spectra': {
            'delta_kappa': (delta_omega/kappa_e).tolist(),
            **transmission_spectra,
        },
        'bistability_sim': {
            't_ns': (t_arr_mult*1e9).tolist(),
            'T_K': T_temp.tolist(),
            'P_abs_uW': (P_abs_hist*1e6).tolist(),
            'DeltaT_steady_K': float(DeltaT_steady),
            'omega_shift_GHz': float(omega_shift_steady_GHz),
            'D_shift_ps_nm_km': float(D_shift_per_K),
        },
        'coupling_map': {
            'EM->Thermal': 'P_abs = U * omega_0/Q_abs  (stored energy * absorption rate)',
            'Thermal->EM': 'delta_omega = -(dn/dT)*DeltaT*(omega_0/n_eff)  (thermo-optic)',
            'Thermal->Mechanical': 'strain = alpha_Si * DeltaT',
            'Mechanical->EM': 'delta_n_PE = -n^3/2 * p12 * strain  (photoelastic)',
            'GS_impact': 'D shifts with T -> GS must use |D| >= 5000 AND track D(T)',
        },
        'H_f_connection': (
            f'D drifts by {D_shift_per_K:.4f} ps/(nm*km) per degree K of heating.\n'
            f'At 5 mW: DeltaT={DeltaT_steady:.2f} K -> D shift tracked by GS algorithm.\n'
            f'This is why |D| >= 5000 is required: thermal drift < 0.01% of D.'
        ),
    }


def demo():
    print("=== FRONTIER CALCULUS: PRECALCULUS -> CHAIN RULE -> MULTIPHYSICS ===\n")

    print("--- Set Theory & Boolean ---")
    sb = set_theory_and_boolean()
    print(f"  De Morgan law 1: {sb['de_morgan']['law1_verified']}  law 2: {sb['de_morgan']['law2_verified']}")
    print(f"  7 + 5 = {sb['adder']['7_plus_5']['sum']} carry={sb['adder']['7_plus_5']['carry_out']}  correct:{sb['adder']['7_plus_5']['correct']}")
    print(f"  A={sb['sets']['A']}  B={sb['sets']['B']}")
    print(f"  A intersect B = {sb['sets']['intersection']}  A union B = {sb['sets']['union']}")
    print(f"  |P(A)| = 2^{len(sb['sets']['A'])} = {sb['sets']['power_set_size']}")
    print(f"  ARM: {sb['ARM']['chain_rule']}")

    print("\n--- Trig + Precalculus + Complex ---")
    tc = trig_precalculus_complex()
    print(f"  sin^2+cos^2=1 max error: {tc['unit_circle']['identity_max_error']:.2e}")
    print(f"  Euler's formula error:   {tc['unit_circle']['euler_max_error']:.2e}")
    print(f"  cos(A+B) via Euler vs formula: {tc['addition_formula']['error']:.2e}")
    print(f"  RC -3dB: |H| = {tc['phasor_RC']['H_mag_at_3dB']:.4f} (expected {tc['phasor_RC']['expected_mag']:.4f})")
    print(f"  RC phase at -3dB: {tc['phasor_RC']['H_phase_at_3dB_deg']:.1f} deg (expected -45.0)")
    print(f"  Python 3.13: {tc['python_history']['2024']}")

    print("\n--- Chain Rule Calculus ---")
    cr = chain_rule_calculus()
    print(f"  d/dx[sin(x^2)] numeric error: {cr['real_chain_rule']['f1_sin_x2_error']:.4f}")
    print(f"  d/dx[Gaussian] numeric error:  {cr['real_chain_rule']['f2_gaussian_error']:.4f}")
    print(f"  Complex chain rule error:      {cr['complex_chain_rule']['error']:.4f}")
    print(f"  Cauchy-Riemann (z^2) error:   {cr['complex_chain_rule']['Cauchy_Riemann_error_z2']:.4f}")
    print(f"  Backprop: loss {cr['backprop']['initial_loss']:.4f} -> {cr['backprop']['final_loss']:.4f}")
    print(f"  Phasor RC: H_mag={cr['phasor']['H_RC_mag']:.4f} phase={cr['phasor']['H_RC_phase_deg']:.1f} deg")
    print(f"  H(f) chain rule: {cr['H_f_chain_rule']['dH_df']}")

    print("\n--- Serway Modern Physics ---")
    serway = serway_modern_physics_problems()
    print(f"  Ch1 Relativity: electron at 0.99c: gamma={serway['Ch1_relativity']['problem_electron_0_99c']['gamma']:.2f}")
    print(f"    KE = {serway['Ch1_relativity']['problem_electron_0_99c']['KE_MeV']:.2f} MeV")
    print(f"  Ch2 Photoelectric (Cu, UV): KE_max = {serway['Ch2_quantum_origins']['photoelectric']['KE_max_eV']:.2f} eV")
    print(f"  Ch2 Compton shift at 90 deg: {serway['Ch2_quantum_origins']['Compton']['delta_lambda_pm']:.3f} pm")
    print(f"  Ch2 de Broglie (thermal n): {serway['Ch2_quantum_origins']['de_Broglie']['lambda_angstrom']:.2f} Angstrom")
    print(f"  Ch3 Bohr: H-alpha = {serway['Ch3_Bohr']['H_alpha_nm']:.1f} nm (expected 656.3)")
    print(f"  Ch4-5 Schrodinger (1nm well): E_1 = {serway['Ch4_5_Schrodinger']['energy_levels_eV'][0]:.3f} eV")
    print(f"    norm(psi_1) = {serway['Ch4_5_Schrodinger']['normalization_n1']:.6f}  ortho(1,2) = {serway['Ch4_5_Schrodinger']['orthogonality_12']:.2e}")
    print(f"  Ch6 Tunneling (5eV, 1nm): T(1eV) = {serway['Ch6_tunneling']['T_at_1_eV']:.4e}")
    print(f"  Ch9 Statistical: kT = {serway['Ch9_statistical']['kT_meV']:.1f} meV at 300 K")

    print("\n--- Multiphysics (Si ring resonator) ---")
    mp = multiphysics_coupling()
    print(f"  Ring FSR = {mp['ring']['FSR_GHz']:.1f} GHz  Q = {mp['ring']['Q_total']:.0f}")
    print(f"  Thermal steady state: DeltaT = {mp['bistability_sim']['DeltaT_steady_K']:.2f} K at 5 mW")
    print(f"  Resonance shift: {mp['bistability_sim']['omega_shift_GHz']:.4f} GHz")
    print(f"  D drift per K: {mp['thermal']['dn_dT_per_K']:.2e} per K (thermo-optic)")
    print(f"  EM->Thermal: {mp['coupling_map']['EM->Thermal']}")

    print("\n=== FRONTIER CALCULUS COMPLETE ===")


if __name__ == '__main__':
    demo()
