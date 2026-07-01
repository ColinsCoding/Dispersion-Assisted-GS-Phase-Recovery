"""
Transistor Technology: From Silicon to Software
CSUS CE curriculum bridge

The abstraction stack (bottom to top):
  Silicon atoms      -> MOSFET physics (quantum tunneling, band gap)
  MOSFET             -> CMOS logic gate (inverter, NAND, NOR)
  Logic gates        -> Boolean algebra (AND/OR/XOR/NOT)
  Boolean algebra    -> Combinational circuits (adder, multiplexer)
  Combinational      -> Sequential circuits (flip-flop, register, counter)
  Sequential         -> CPU datapath (ALU, register file, PC)
  CPU datapath       -> ISA (x86, ARM, RISC-V instruction set)
  ISA                -> Compiler output (assembly)
  Compiler           -> High-level language (C, Python)
  Python             -> You typing `import dgs`

One MOSFET at the bottom. Everything else is abstraction on top.
This is what CE teaches. Physics tells you WHY the MOSFET works.
"""
import numpy as np
import sympy as sp

# Physical constants
kB  = 1.380649e-23
e   = 1.602176634e-19
h   = 6.62607015e-34
eps_si = 11.7 * 8.854e-12   # Si permittivity

# ===========================================================================
# MOSFET Physics (CE 110 / EE 112 level)
# ===========================================================================

MOSFET_TECHNOLOGY_NODES = {
    '1971_intel_4004':  {'node_nm': 10000, 'transistors': 2300,       'year': 1971, 'vdd_V': 15.0},
    '1978_intel_8086':  {'node_nm': 3000,  'transistors': 29000,      'year': 1978, 'vdd_V': 5.0},
    '1993_pentium':     {'node_nm': 800,   'transistors': 3.1e6,      'year': 1993, 'vdd_V': 5.0},
    '2000_pentium4':    {'node_nm': 180,   'transistors': 42e6,       'year': 2000, 'vdd_V': 1.8},
    '2006_core2':       {'node_nm': 65,    'transistors': 291e6,      'year': 2006, 'vdd_V': 1.2},
    '2012_ivy_bridge':  {'node_nm': 22,    'transistors': 1.4e9,      'year': 2012, 'vdd_V': 0.9},
    '2017_kaby_lake':   {'node_nm': 14,    'transistors': 3.0e9,      'year': 2017, 'vdd_V': 0.8},
    '2020_apple_m1':    {'node_nm': 5,     'transistors': 16e9,       'year': 2020, 'vdd_V': 0.75},
    '2022_apple_m2':    {'node_nm': 3,     'transistors': 20e9,       'year': 2022, 'vdd_V': 0.7},
    '2024_apple_m4':    {'node_nm': 3,     'transistors': 28e9,       'year': 2024, 'vdd_V': 0.65},
}


def moores_law_check():
    """
    Moore's Law: transistor count doubles every ~2 years.
    Gordon Moore, 1965: observation, NOT a law of physics.
    It held for 50 years because engineers kept finding new tricks.
    At 3nm: gate oxide is 1-2 atoms thick. Quantum tunneling IS the device now.
    """
    nodes = MOSFET_TECHNOLOGY_NODES
    years = [v['year'] for v in nodes.values()]
    counts = [v['transistors'] for v in nodes.values()]
    names = list(nodes.keys())

    # Log-linear fit: log2(N) = a*year + b
    log2_counts = np.log2(counts)
    coeffs = np.polyfit(years, log2_counts, 1)
    doubling_time = np.log2(np.e) / (coeffs[0] * np.log(2))
    # simpler: slope in log2 units per year
    doublings_per_year = coeffs[0]
    doubling_years = 1.0 / doublings_per_year

    return {
        'nodes': names,
        'years': years,
        'transistor_counts': counts,
        'doubling_time_years': doubling_years,
        'moores_prediction': 2.0,
        'current_node_nm': 3,
        'gate_oxide_atoms': 5,
        'tunneling_current': 'SIGNIFICANT at <5nm: quantum mechanics is the device',
        'next_frontier': 'GAA (Gate-All-Around) FET, 2D materials (MoS2), carbon nanotubes',
    }


def mosfet_iv_curves(Vgs_values=None, Vt=0.5, k=2e-3, lambda_ch=0.1, N=200):
    """
    NMOS MOSFET I-V characteristics (SPICE Level 1 model):

    Cutoff (Vgs < Vt):
      Id = 0

    Linear/Triode (Vgs > Vt, Vds < Vgs-Vt):
      Id = k * [(Vgs-Vt)*Vds - Vds^2/2] * (1 + lambda*Vds)

    Saturation (Vgs > Vt, Vds > Vgs-Vt):
      Id = k/2 * (Vgs-Vt)^2 * (1 + lambda*Vds)

    k = mu_n * Cox * W/L  [A/V^2]  -- transconductance parameter
    lambda = channel-length modulation [1/V]

    Software connection:
      When Vgs=Vdd (logic HIGH): NMOS ON, current flows, output pulled LOW
      When Vgs=0   (logic LOW):  NMOS OFF, no current, output = high-Z
      Two states = binary = 1 bit
    """
    if Vgs_values is None:
        Vgs_values = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8]

    Vds = np.linspace(0, 2.0, N)
    curves = {}
    for Vgs in Vgs_values:
        Vov = Vgs - Vt
        if Vov <= 0:
            Id = np.zeros(N)
        else:
            Vds_sat = Vov
            Id = np.where(
                Vds < Vds_sat,
                k * ((Vov)*Vds - 0.5*Vds**2) * (1 + lambda_ch*Vds),
                0.5 * k * Vov**2 * (1 + lambda_ch*Vds)
            )
            Id = np.maximum(Id, 0)
        curves[Vgs] = Id
    return {'Vds': Vds, 'Id_curves': curves, 'Vt': Vt, 'k_A_per_V2': k}


def cmos_inverter(Vin_V=None, Vdd=1.8, Vtn=0.4, Vtp=-0.4, kn=2e-3, kp=1e-3, N=300):
    """
    CMOS Inverter: 1 NMOS + 1 PMOS in series between Vdd and GND.
    Vout = NOT(Vin)

    When Vin = LOW (0V):
      NMOS OFF (Vgs_n < Vtn)
      PMOS ON  (Vgs_p = 0-Vdd = -Vdd < Vtp -> ON)
      Vout = Vdd (HIGH)

    When Vin = HIGH (Vdd):
      NMOS ON
      PMOS OFF
      Vout = 0 (LOW)

    The voltage transfer characteristic (VTC) shows the switching threshold.
    Logic threshold Vth ~ Vdd/2 for balanced NMOS/PMOS (kn=kp).

    Power: P = alpha * C * Vdd^2 * f  (dynamic, no static power in CMOS!)
      alpha = activity factor, C = load capacitance, f = clock frequency
      This is why Vdd scaled: Intel 1993 = 5V, Apple M4 = 0.65V
      P scales as Vdd^2 -- 8x power reduction from 5V to 1.8V

    Key insight: CMOS consumes power ONLY during switching, not when idle.
    Bipolar (BJT) consumes static power. CMOS won.
    """
    if Vin_V is None:
        Vin_V = np.linspace(0, Vdd, N)

    Vin = np.array(Vin_V)
    Vout = np.zeros_like(Vin)

    for i, vin in enumerate(Vin):
        # Simplified: find Vout where Id_n = |Id_p| (equilibrium)
        # Iterative Newton's method on current balance
        vout = Vdd / 2
        for _ in range(50):
            Vgs_n = vin; Vds_n = vout
            Vov_n = Vgs_n - Vtn
            if Vov_n <= 0:
                In = 0.0
            elif Vds_n >= Vov_n:
                In = 0.5 * kn * Vov_n**2
            else:
                In = kn * (Vov_n*Vds_n - 0.5*Vds_n**2)

            Vgs_p = vin - Vdd; Vds_p = vout - Vdd
            Vov_p = Vgs_p - Vtp
            if Vov_p >= 0:
                Ip = 0.0
            elif Vds_p <= Vov_p:
                Ip = 0.5 * kp * Vov_p**2
            else:
                Ip = kp * (Vov_p*Vds_p - 0.5*Vds_p**2)

            err = In - abs(Ip)
            vout -= err * 1e2
            vout = np.clip(vout, 0, Vdd)
        Vout[i] = vout

    noise_margin_H = Vdd - 0.7*Vdd
    noise_margin_L = 0.3*Vdd

    return {
        'Vin': Vin,
        'Vout': Vout,
        'Vdd': Vdd,
        'logic_threshold_V': Vdd/2,
        'noise_margin_H': noise_margin_H,
        'noise_margin_L': noise_margin_L,
        'static_power_W': 0.0,
        'dynamic_power_formula': 'P = alpha * C * Vdd^2 * f',
        'lesson': 'CMOS: zero static power. Vdd scaling is the key to mobile computing.',
    }


# ===========================================================================
# Logic Gates (gates -> software)
# ===========================================================================

def logic_gate_truth_tables():
    """
    CMOS implementation of basic gates.
    Each gate = 2-6 transistors.

    NAND gate: 2 NMOS series + 2 PMOS parallel (4 transistors total)
    Universal gate: NAND alone can implement ANY logic function.
    This is why CPU designers love NAND.
    """
    A = [0, 0, 1, 1]
    B = [0, 1, 0, 1]

    gates = {
        'AND':  [int(a and b) for a, b in zip(A, B)],
        'OR':   [int(a or  b) for a, b in zip(A, B)],
        'NAND': [int(not (a and b)) for a, b in zip(A, B)],
        'NOR':  [int(not (a or  b)) for a, b in zip(A, B)],
        'XOR':  [int(a ^ b)  for a, b in zip(A, B)],
        'XNOR': [int(not (a ^ b)) for a, b in zip(A, B)],
    }

    # NAND universality: implement NOT, AND, OR from NAND only
    NOT_from_NAND  = lambda a: int(not (a and a))   # NAND(A,A) = NOT(A)
    AND_from_NAND  = lambda a,b: NOT_from_NAND(int(not (a and b)))  # NOT(NAND(A,B))
    OR_from_NAND   = lambda a,b: int(not (not a and not b))         # De Morgan

    return {
        'inputs': list(zip(A, B)),
        'gates': gates,
        'transistors_per_gate': {'NOT':2, 'NAND':4, 'NOR':4, 'AND':6, 'OR':6, 'XOR':12},
        'nand_universality': 'NAND(A,A)=NOT(A). Any function implementable with NAND only.',
        'lesson': '2 transistors -> 1 bit of logic. 28 billion transistors -> Apple M4.',
    }


def ripple_carry_adder(A_bits, B_bits):
    """
    N-bit ripple carry adder: chain of full adders.
    Full adder: Sum = A XOR B XOR Cin, Cout = (A AND B) OR (Cin AND (A XOR B))

    Gate delay: 2*N gate delays (critical path through carry chain)
    This is WHY carry-lookahead adder (CLA) was invented: O(log N) delay.

    Connection to software: every + operation in Python eventually becomes
    this circuit running at 3-4 GHz on your CPU.
    """
    n = len(A_bits)
    assert len(B_bits) == n, "A and B must have same length"

    Sum_bits = []
    Cout = 0

    for i in range(n-1, -1, -1):  # LSB first
        a = A_bits[i]
        b = B_bits[i]
        cin = Cout
        s   = a ^ b ^ cin
        cout= (a & b) | (cin & (a ^ b))
        Sum_bits.insert(0, s)
        Cout = cout

    return {
        'A_bits': A_bits,
        'B_bits': B_bits,
        'Sum_bits': Sum_bits,
        'Carry_out': Cout,
        'A_decimal': int(''.join(str(b) for b in A_bits), 2),
        'B_decimal': int(''.join(str(b) for b in B_bits), 2),
        'Sum_decimal': int(''.join(str(b) for b in Sum_bits), 2) + Cout * 2**n,
        'gate_delay_fullpath': 2*n,
        'lesson': 'Every integer add in Python -> this circuit at 3GHz.',
    }


# ===========================================================================
# Abstraction Stack: Transistor to Python
# ===========================================================================

ABSTRACTION_STACK = [
    {
        'layer': 0, 'name': 'Silicon / Quantum Physics',
        'what': 'Si crystal lattice, band gap 1.12eV, p-n junction, inversion layer',
        'math': 'Schrodinger eq, Fermi-Dirac distribution, Poisson eq (electrostatics)',
        'who_cares': 'Device physicist, fab engineer',
        'example': 'Electron density in channel: n(x) = Ni*exp((Phi_F-Phi(x))/VT)',
        'course': 'EE 115 Device Physics',
    },
    {
        'layer': 1, 'name': 'MOSFET Device',
        'what': 'Vt, Id-Vds curves, gm, Cgs, saturation, subthreshold slope',
        'math': 'SPICE Level 1-3 models, small-signal equivalent circuit',
        'who_cares': 'Analog/mixed-signal IC designer',
        'example': 'Id = k/2*(Vgs-Vt)^2*(1+lambda*Vds)',
        'course': 'EE 112 Electronics',
    },
    {
        'layer': 2, 'name': 'CMOS Logic Gate',
        'what': 'Inverter, NAND, NOR, XOR. Voltage transfer characteristic.',
        'math': 'Boolean algebra, noise margins, propagation delay, power',
        'who_cares': 'Digital IC designer (standard cell library)',
        'example': 'NAND: 2 NMOS series + 2 PMOS parallel. 4 transistors.',
        'course': 'EE/CE 142 Digital Logic Design',
    },
    {
        'layer': 3, 'name': 'Combinational / Sequential Logic',
        'what': 'Adder, multiplexer, flip-flop, register, counter, state machine',
        'math': 'Karnaugh maps, FSM, timing analysis (setup/hold), Verilog',
        'who_cares': 'RTL designer, FPGA engineer',
        'example': 'D flip-flop: Q(t+1) = D(t) on rising clock edge',
        'course': 'CE 141 Logic Design with HDL',
    },
    {
        'layer': 4, 'name': 'CPU Datapath / Microarchitecture',
        'what': 'ALU, register file, PC, instruction fetch/decode/execute/writeback',
        'math': 'Pipeline hazards, branch prediction, cache miss rate',
        'who_cares': 'CPU architect (Intel, AMD, Apple, ARM)',
        'example': 'MIPS: add $t0, $t1, $t2  -> 5 pipeline stages -> 1 cycle at 3GHz',
        'course': 'CE 151 Computer Organization',
    },
    {
        'layer': 5, 'name': 'Instruction Set Architecture (ISA)',
        'what': 'x86-64, ARM64, RISC-V: instruction encoding, addressing modes, ABI',
        'math': 'Integer/float formats (IEEE 754), memory model, calling convention',
        'who_cares': 'Compiler writer, OS developer, embedded programmer',
        'example': 'x86: ADD RAX, RBX  (3 bytes) -> micro-ops in out-of-order engine',
        'course': 'CE 152 Assembly Language',
    },
    {
        'layer': 6, 'name': 'Operating System / Runtime',
        'what': 'Process/thread scheduler, virtual memory, syscall, drivers',
        'math': 'Scheduling algorithms, page table walks, interrupt latency',
        'who_cares': 'OS developer, systems programmer',
        'example': 'Python GIL: one thread at a time in CPython. Use multiprocessing for parallelism.',
        'course': 'CE 155 Operating Systems',
    },
    {
        'layer': 7, 'name': 'High-Level Language (Python)',
        'what': 'CPython bytecode -> interpreter -> C extension -> syscall -> hardware',
        'math': 'Big-O algorithm complexity, memory allocator, GC',
        'who_cares': 'Software engineer, data scientist, you',
        'example': 'x = a + b  -> BINARY_ADD bytecode -> PyNumber_Add C func -> ADDQ x86',
        'course': 'CE 150 Data Structures (Python)',
    },
]


def explain_stack(start_layer=0, end_layer=7):
    """Print the abstraction stack from transistor to Python."""
    for entry in ABSTRACTION_STACK:
        if start_layer <= entry['layer'] <= end_layer:
            print(f"Layer {entry['layer']}: {entry['name']}")
            print(f"  What:    {entry['what']}")
            print(f"  Example: {entry['example']}")
            print(f"  Course:  {entry['course']}")
            print()


def python_to_transistor(expression='x = 2 + 3'):
    """
    Trace one Python expression down to the transistor level.
    x = 2 + 3:
      Python:    LOAD_CONST 2, LOAD_CONST 3, BINARY_ADD, STORE_NAME x
      CPython:   PyNumber_Add(a, b) in ceval.c
      C:         long_add() -> C integer add
      Compiler:  ADD instruction in x86/ARM
      Microcode: ALU operation (ripple_carry_adder with N=64)
      CMOS:      N=64 full adders = 64*28 = ~1800 transistors for one add
      Silicon:   ~1800 MOSFETs switching in ~0.3ns at 3.3GHz
    """
    transistors_per_full_adder = 28
    n_bits = 64
    transistors_for_one_add = n_bits * transistors_per_full_adder

    return {
        'expression': expression,
        'python_bytecode': 'LOAD_CONST, BINARY_ADD, STORE_NAME',
        'cpython_function': 'PyNumber_Add -> long_add in Objects/longobject.c',
        'compiler_output': 'ADD RAX, RBX  (x86-64)',
        'microarchitecture': f'{n_bits}-bit ripple carry adder ({transistors_per_full_adder} transistors/stage)',
        'transistors_activated': transistors_for_one_add,
        'time_ns': 1000/3300,
        'power_pJ': 0.5 * 1e-15 * 1.8**2 * 3.3e9,
        'lesson': (
            f'One Python `+` = {transistors_for_one_add} transistors switching '
            f'in {1000/3300:.2f}ns. Apple M4 has 28 BILLION transistors.'
        ),
    }


def software_rate_of_change():
    """
    Technology doubles every 2 years (Moore). Software abstraction rate:
    - 1950: program in machine code (0s and 1s)
    - 1955: assembly language (mov, add, jmp)
    - 1960: FORTRAN, COBOL (first compilers)
    - 1972: C (systems programming)
    - 1991: Python (interpreted, everything is an object)
    - 2015: PyTorch (autograd -- differentiate through GPU code)
    - 2020: LLM code generation (describe intent, get code)

    Each layer adds abstraction, hides complexity, enables more people.
    CE degree = understand ALL layers. Everyone else uses the top 2.
    """
    return {
        'stack_height': 8,
        'abstraction_gain_per_decade': '10-100x programmer productivity',
        'ce_advantage': 'Understand all 8 layers. Debug at any level. No black boxes.',
        'moore_law_status': 'Slowing at 3nm. Software innovation > hardware from here.',
        'next_platforms': ['Quantum computing (this repo)', 'Neuromorphic', 'Photonic computing'],
        'photonic_connection': (
            'Photonic time-stretch (this repo) computes H(f)=exp(j*pi*D*f^2) '
            'at the SPEED OF LIGHT in analog domain. '
            'No transistors needed for the FFT -- physics does it for free.'
        ),
    }


def demo():
    print("=== TRANSISTOR TECHNOLOGY STACK ===\n")

    print("Moore's Law:")
    ml = moores_law_check()
    print(f"  Doubling time: {ml['doubling_time_years']:.1f} years")
    print(f"  Doubling time from data: {ml['doubling_time_years']:.1f} years (Moore predicted 2.0)")
    for name, data in list(MOSFET_TECHNOLOGY_NODES.items())[-3:]:
        print(f"  {data['year']} {name.split('_',1)[1]}: {data['transistors']:.2e} transistors, {data['node_nm']}nm, Vdd={data['vdd_V']}V")

    print("\nCMOS Inverter:")
    inv = cmos_inverter()
    print(f"  Logic threshold: {inv['logic_threshold_V']:.2f}V")
    print(f"  Static power: {inv['static_power_W']} W  <- CMOS advantage")
    print(f"  Dynamic power: {inv['dynamic_power_formula']}")

    print("\nLogic Gates:")
    lg = logic_gate_truth_tables()
    print(f"  NAND truth table: {lg['gates']['NAND']}")
    print(f"  Transistors: {lg['transistors_per_gate']}")
    print(f"  {lg['nand_universality']}")

    print("\n4-bit Addition (1011 + 0101):")
    add = ripple_carry_adder([1,0,1,1], [0,1,0,1])
    print(f"  {add['A_decimal']} + {add['B_decimal']} = {add['Sum_decimal']}")
    print(f"  Binary: {add['Sum_bits']} (carry={add['Carry_out']})")

    print("\nPython x=2+3 -> transistors:")
    pt = python_to_transistor()
    print(f"  {pt['lesson']}")

    print("\nAbstraction Stack (top 3 layers):")
    for entry in ABSTRACTION_STACK[-3:]:
        print(f"  L{entry['layer']}: {entry['name']}")
        print(f"    {entry['example']}")

    print("\nSoftware rate of change:")
    sr = software_rate_of_change()
    print(f"  CE advantage: {sr['ce_advantage']}")
    print(f"  Photonic: {sr['photonic_connection'][:80]}...")


if __name__ == '__main__':
    demo()
