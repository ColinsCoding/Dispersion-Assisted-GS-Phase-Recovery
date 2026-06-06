"""
_repl_bool_circuits_creep.py

S1: Experimental design -- independent/dependent/control variables, alpha=0.05
S2: Boolean algebra -- &&, ||, ~, XOR, De Morgan, truth tables
S3: Logic circuits -- gates, Karnaugh map, combinational vs sequential
S4: Fried circuits -- thermal failure, I^2*R, fusing current, derating
S5: Material creep -- primary/secondary/tertiary, Norton, Arrhenius
S6: Applied math connections -- all of the above in one framework
"""

import numpy as np
import sympy as sp

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: EXPERIMENTAL DESIGN
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 1: VARIABLES AND SIGNIFICANCE -- alpha = 0.05")
print(SEP)

print("""
  THREE VARIABLE TYPES:
  ---------------------------------------------------------------
  INDEPENDENT   What you deliberately CHANGE.
                "The cause."  You control it.
                Example: laser power (W), temperature (C), drug dose (mg)

  DEPENDENT     What you MEASURE as a result.
                "The effect."  It responds to the independent variable.
                Example: phase error (rad), cell survival fraction, yield

  CONTROL       Everything you HOLD CONSTANT to isolate the effect.
                "Held fixed."  Eliminates confounds.
                Example: fiber length, room temp, batch of cells, time of day

  GOOD EXPERIMENT: change ONE independent variable at a time.
  FACTORIAL DESIGN: change multiple simultaneously, test interactions.
    2^k full factorial: k factors at 2 levels each = 2^k runs
    Interaction term: effect of A depends on level of B

  NULL HYPOTHESIS (H0):
    "No effect."  IV has no influence on DV.
    e.g. H0: mean_group_A == mean_group_B

  ALPHA = 0.05  (significance level):
    Probability of rejecting H0 when it is TRUE (Type I error / false positive).
    If p-value < 0.05 -> reject H0 -> "statistically significant"
    If p-value >= 0.05 -> fail to reject H0

  p-VALUE:
    Probability of observing data THIS extreme or more if H0 were true.
    p < 0.05 does NOT mean "95% chance the effect is real."
    It means "if H0 true, chance of this result = p."

  TYPE I  error (alpha): reject H0 when it is true    (false positive)
  TYPE II error (beta):  fail to reject H0 when false (false negative)
  POWER = 1 - beta:      probability of detecting real effect

  EFFECT SIZE:
    Cohen's d = (mean_A - mean_B) / pooled_SD
    d=0.2 small, d=0.5 medium, d=0.8 large

  SCIENCE vs ENGINEERING vs APPLIED MATH:
    Science:       discover what IS TRUE about nature (falsifiable claims)
    Engineering:   use truth to BUILD things that WORK under constraints
    Applied math:  provide TOOLS (models, proofs) for both
    Connection:    science gives models -> math formalizes -> engineering implements
""")

# live t-test
from scipy import stats

np.random.seed(42)
group_a = np.random.normal(loc=10.0, scale=2.0, size=30)
group_b = np.random.normal(loc=11.2, scale=2.0, size=30)
t_stat, p_val = stats.ttest_ind(group_a, group_b)
d_cohen = (group_b.mean() - group_a.mean()) / np.sqrt(
    ((group_a.std()**2 + group_b.std()**2) / 2))

print(f"  Example t-test:")
print(f"  Group A: mean={group_a.mean():.3f}, std={group_a.std():.3f}, n=30")
print(f"  Group B: mean={group_b.mean():.3f}, std={group_b.std():.3f}, n=30")
print(f"  t = {t_stat:.3f},  p = {p_val:.4f}")
print(f"  Cohen d = {d_cohen:.3f}")
print(f"  Decision (alpha=0.05): {'REJECT H0 -- significant' if p_val < 0.05 else 'FAIL TO REJECT H0'}")

# ------------------------------------------------------------------ #
# S2: BOOLEAN ALGEBRA
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 2: BOOLEAN ALGEBRA  &&  ||  ~  XOR")
print(SEP)

print("""
  OPERATOR MAPPING:
  Math symbol   C / Python    Name
  ----------------------------------
  A AND B       A && B        Conjunction  (both true)
  A OR  B       A || B        Disjunction  (at least one true)
  NOT A         !A  / ~A      Complement
  A XOR B       A ^ B         Exclusive or (exactly one true)
  A XNOR B      !(A^B)        Equivalence  (same value)
  A NAND B      !(A&&B)       Functionally complete alone
  A NOR  B      !(A||B)       Also functionally complete alone

  NOTE: in Python,  & | ~ ^ are BITWISE on integers.
        and or not    are LOGICAL (short-circuit, return object).
        For numpy arrays: use & | ~ ^ (not and/or/not).

  LAWS OF BOOLEAN ALGEBRA:
  Identity:     A && 1 = A           A || 0 = A
  Null:         A && 0 = 0           A || 1 = 1
  Idempotent:   A && A = A           A || A = A
  Complement:   A && ~A = 0          A || ~A = 1
  Double neg:   ~~A = A
  Commutative:  A&&B = B&&A          A||B = B||A
  Associative:  (A&&B)&&C = A&&(B&&C)
  Distributive: A&&(B||C) = (A&&B)||(A&&C)
  Absorption:   A&&(A||B) = A        A||(A&&B) = A

  DE MORGAN'S LAWS (most important):
    ~(A && B) = ~A || ~B     NOT(AND) = OR(NOTs)
    ~(A || B) = ~A && ~B     NOT(OR)  = AND(NOTs)

  Used everywhere:
    NAND gate = NOT-AND -> ~(A&&B) = ~A||~B  -> flip inputs, flip gate
    NOR  gate = NOT-OR  -> ~(A||B) = ~A&&~B
    Simplifying logic expressions, timing optimization, FPGA synthesis
""")

# truth tables
print("  TRUTH TABLE (2 variables):")
print(f"  {'A':>3} {'B':>3} {'A&&B':>6} {'A||B':>6} {'~A':>4} {'A^B':>5} {'~(A&&B)':>9} {'~A||~B':>8}")
print("  " + "-" * 50)
for A in [0, 1]:
    for B in [0, 1]:
        AND  = A & B
        OR   = A | B
        NOTA = 1 - A
        XOR  = A ^ B
        NAND = 1 - AND
        DEM  = (1-A) | (1-B)
        eq   = "==" if NAND == DEM else "!="
        print(f"  {A:>3} {B:>3} {AND:>6} {OR:>6} {NOTA:>4} {XOR:>5} "
              f"{NAND:>9} {DEM:>8}  DeMorgan {eq}")

print("""
  BITWISE vs LOGICAL in Python:
    x = 6  (binary 110)
    y = 3  (binary 011)
    x & y  = 2  (binary 010)   -- bitwise AND, bit by bit
    x | y  = 7  (binary 111)   -- bitwise OR
    x ^ y  = 5  (binary 101)   -- bitwise XOR
    ~x     = -7               -- bitwise NOT (two's complement: ~x = -x-1)
    x && y (C) / (x and y) Python: returns y if x truthy, else x  (short-circuit)

  SHIFT OPERATORS:
    x << n = x * 2^n    (left shift, fast multiply by power of 2)
    x >> n = x // 2^n   (right shift, fast integer divide)
    6 << 2 = 24          6 >> 1 = 3
""")

# bitwise demo
x, y = 6, 3
print(f"  x={x} ({x:04b}),  y={y} ({y:04b})")
print(f"  x & y  = {x&y} ({x&y:04b})  AND")
print(f"  x | y  = {x|y} ({x|y:04b})  OR")
print(f"  x ^ y  = {x^y} ({x^y:04b})  XOR")
print(f"  ~x     = {~x}  (two's complement)")
print(f"  x << 2 = {x<<2}   x >> 1 = {x>>1}")

# ------------------------------------------------------------------ #
# S3: LOGIC CIRCUITS
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 3: LOGIC CIRCUITS")
print(SEP)

print("""
  BASIC GATES:
  Gate    Symbol  Output
  ---------------------------------
  NOT     bubble  Y = ~A
  AND     D-shape Y = A & B
  OR      curve   Y = A | B
  NAND    AND+bubble  Y = ~(A&B)   <- FUNCTIONALLY COMPLETE
  NOR     OR+bubble   Y = ~(A|B)   <- FUNCTIONALLY COMPLETE
  XOR     !=      Y = A ^ B
  XNOR    =       Y = ~(A^B)

  FUNCTIONAL COMPLETENESS:
    Any Boolean function can be built from NAND gates alone.
    Any Boolean function can be built from NOR gates alone.
    CMOS: NAND and NOR are cheapest (fewer transistors than AND/OR).

  COMBINATIONAL vs SEQUENTIAL:
    Combinational: output depends ONLY on current inputs (no memory).
      Examples: adder, mux, decoder, priority encoder
    Sequential:   output depends on inputs AND past state (has memory).
      Examples: flip-flop, register, counter, FSM, SRAM

  SOP (Sum of Products) -- canonical form from truth table:
    Write a minterm (AND of all variables) for each row where Y=1.
    OR all minterms together.
    Example: Y = ~A&~B&C | ~A&B&C | A&~B&C | A&B&~C
             simplified by Karnaugh map.

  KARNAUGH MAP (2-variable example):
    Cells adjacent (including wrap) -> factor out differing variable.

       B=0  B=1
  A=0 |  0 |  1 |    -> B (A=0, B=1 and A=1, B=1) -> group {A=0,B=1},{A=1,B=1} = B
  A=1 |  0 |  1 |       Y = B

  3-variable K-map groups:
    1 cell  = 3 literals
    2 cells = 2 literals (1 variable eliminated)
    4 cells = 1 literal  (2 variables eliminated)
    8 cells = 1          (tautology, Y=1 always)
    Group sizes must be powers of 2. Wrap around edges counts.

  FLIP-FLOP (D type):
    Q(t+1) = D(t)  on rising clock edge
    Stores 1 bit.  SR, JK, T types derived from D.

  PROPAGATION DELAY:
    t_pd = time from input change to output stable
    Setup time t_su: D must be stable BEFORE clock edge
    Hold time t_h:   D must stay stable AFTER  clock edge
    Maximum clock freq: f_max = 1 / (t_pd_combinational + t_su + t_clk_skew)

  CMOS GATE TRANSISTOR COUNT:
    NOT:  2 transistors (1 PMOS + 1 NMOS)
    NAND2: 4  (2P series pull-up, 2N parallel pull-down... wait:
               2P parallel + 2N series = 4 total)
    NOR2:  4  (2P series + 2N parallel)
    AND2:  NAND2 + NOT = 6
    OR2:   NOR2  + NOT = 6
    XOR2:  8-12 depending on implementation

  2-BIT ADDER truth table (A1A0 + B1B0 -> S2S1S0):
""")

print(f"  {'A':>4} {'B':>4} {'Sum':>5} {'C_out':>6}")
print("  " + "-" * 22)
for A in range(4):
    for B in range(4):
        S = A + B
        print(f"  {A:>4} {B:>4} {S%4:>5} {S>>2:>6}")

# ------------------------------------------------------------------ #
# S4: FRIED CIRCUITS -- THERMAL FAILURE
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: FRIED CIRCUITS -- THERMAL FAILURE")
print(SEP)

print("""
  WHY CIRCUITS FRY:
    1. I^2 * R heating -- Joule heating in resistive elements
    2. Avalanche breakdown -- reverse voltage exceeds V_BR in diodes/BJT
    3. Latch-up -- CMOS SCR structure triggers, low-impedance path
    4. ESD (electrostatic discharge) -- >1000V spike, ns duration
    5. Thermal runaway -- Ic increases -> Tj increases -> Ic increases
    6. Overcurrent (fusing current) -- trace melts at I_fuse

  JOULE HEATING:
    P = I^2 * R = V^2 / R = V * I      [Watts]
    Temperature rise:  delta_T = P * R_theta_JA
    R_theta_JA = junction-to-ambient thermal resistance [C/W]
    Tj = Ta + P * R_theta_JA

  DERATING RULE:
    Never operate at 100% of rated values.
    Military standard (MIL-HDBK-217): derate to 50-70% of rated I, V, P.
    Consumer: 80% typical.
    Capacitors: derate voltage to 50% for long life (electrolytic).

  FUSING CURRENT (PCB trace):
    Onderdonk's equation:  I = A * (T_m - T_a)^0.5 / (sqrt(t) * 33)
    A = cross-section area (mils^2),  T_m = melting point (C), t = time (s)
    For copper 1oz (35um), 1mm wide trace:
      A = 1mm * 35um = 35e-6 m^2... (use mils: 39.4 mils wide, 1.37 mils thick)
    IPC-2221 current capacity: I = k * dT^0.44 * A^0.725
      k=0.048 (internal), k=0.048 (external differs slightly)

  THERMAL RUNAWAY in BJT:
    Ic increases with temperature (minority carrier concentration ~ exp(-Eg/kT))
    -> more power dissipation -> higher Tc -> more Ic -> unstable
    Fix: emitter degeneration resistor Re -> negative feedback
         separate base bias (not collector-feedback bias alone)

  PCB TRACE RESISTANCE:
    R = rho * L / (w * t)
    rho_Cu = 1.72e-8 ohm*m at 25C
    1 oz copper = 35 um thick
    Temperature coefficient: R(T) = R0 * (1 + alpha*(T-T0))
    alpha_Cu = 0.00393 /C
""")

# thermal calculations
print("  PCB trace heating examples (copper, 35um thick):")
rho_Cu = 1.72e-8   # ohm*m
t_Cu   = 35e-6     # 35 um = 1 oz copper
alpha  = 0.00393   # /C

print(f"  {'Width (mm)':>12} {'Length (mm)':>13} {'R (mohm)':>10} {'I_max@20C rise':>16}")
print("  " + "-" * 52)
for w_mm in [0.1, 0.25, 0.5, 1.0, 2.0]:
    L_mm = 50.0
    w_m = w_mm * 1e-3
    L_m = L_mm * 1e-3
    R_ohm = rho_Cu * L_m / (w_m * t_Cu)
    # IPC-2221 external trace: I = 0.048 * dT^0.44 * A_mils^0.725
    w_mils = w_mm * 39.37
    t_mils = t_Cu * 39370  # 1.378 mils
    A_mils2 = w_mils * t_mils
    dT = 20.0  # C rise
    I_max = 0.048 * (dT**0.44) * (A_mils2**0.725)
    print(f"  {w_mm:>12.2f} {L_mm:>13.1f} {R_ohm*1e3:>10.2f} {I_max:>16.2f} A")

print("""
  ESD PROTECTION:
    Human body model (HBM): 100 pF cap + 1.5 kohm series -> 100-2000V pulse
    Machine model (MM): 200 pF, no series R -> more destructive
    CDM (charged device model): fastest, most common in manufacturing
    Protection: TVS diode clamps to V_clamp in ns; input series resistor limits I

  LATCH-UP:
    CMOS has parasitic PNPN (SCR) structure between supply rails.
    If triggered (ESD, overvoltage): low-impedance path, chips burn.
    Fix: guard rings, well ties, careful layout, bulk CMOS process options.
""")

# ------------------------------------------------------------------ #
# S5: MATERIAL CREEP
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 5: MATERIAL CREEP -- TIME-DEPENDENT DEFORMATION")
print(SEP)

print("""
  CREEP: slow, permanent deformation under CONSTANT stress, over time.
  Occurs when T > 0.3 * T_melting (homologous temperature T/T_m > 0.3).
  Metals: above ~200-400C.  Polymers: at room temperature.
  Solder joints in electronics: creep is the PRIMARY failure mode.

  THREE STAGES:
    Primary (transient):  strain rate DECREASES -- work hardening wins
    Secondary (steady):   constant strain rate dE/dt = eps_dot
                          <- this is what engineering equations model
    Tertiary:             strain rate INCREASES -> necking -> fracture

  NORTON POWER LAW (steady-state creep):
    eps_dot = A * sigma^n * exp(-Q_c / (R * T))

    eps_dot = steady-state strain rate [1/s]
    A       = material constant [depends on units]
    sigma   = applied stress [MPa]
    n       = creep exponent (typically 3-8 for metals)
    Q_c     = activation energy for creep [J/mol]
    R       = 8.314 J/(mol*K)
    T       = absolute temperature [K]

  PHYSICAL MEANING OF n:
    n=1: diffusion creep (Nabarro-Herring, Coble) -- grain boundary diffusion
    n=3: viscous glide of dislocations
    n=5: climb-controlled dislocation creep (most metals at moderate stress)
    n>8: power-law breakdown -- very high stress, solute drag

  ARRHENIUS TEMPERATURE DEPENDENCE:
    exp(-Q_c / RT) -- same form as chemical kinetics, semiconductor intrinsic
    carrier concentration, diffusion coefficients, oxidation rates.
    Doubling temperature (in K) massively accelerates creep.
    Rule of thumb: +10C roughly DOUBLES creep rate for many materials.

  LARSON-MILLER PARAMETER (life prediction):
    LMP = T * (log10(t_r) + C)
    t_r = time to rupture [hr],  T [R or K],  C ~ 20 (material constant)
    Isoparametric: higher T compensated by shorter t for same LMP.
    Used to extrapolate short-time lab tests to long service life.

  CREEP IN SOLDER (Pb-Sn, SAC305):
    T_m (63/37 SnPb) = 183 C = 456 K
    Room temp (25 C = 298 K): T/T_m = 0.65 -> WELL in creep regime
    Electronic solder is ALWAYS creeping at room temperature.
    Thermal cycling (-40 to 125C) alternately compresses/stretches joint.
    Coffin-Manson fatigue: N_f ~ (delta_gamma)^(-c)  where c ~ 2 for solder

  CREEP IN FIBER OPTIC CABLES:
    Silica fiber coated in polymer jacket.
    Polymer creeps -> changes fiber bend radius over years.
    Causes slow drift in coupling efficiency -> signal loss.
    Relevant for RogueGuard long-term deployment stability.
""")

# Norton creep: Arrhenius plot
print("  Norton creep rate vs temperature (A=1e-10, sigma=50MPa, n=5, Q=150kJ/mol):")
A_cr  = 1e-10
sigma = 50.0       # MPa
n_cr  = 5
Q_cr  = 150e3      # J/mol
R_gas = 8.314

print(f"  {'T (C)':>8} {'T (K)':>8} {'T/Tm_steel':>12} {'eps_dot (1/s)':>16}")
print("  " + "-" * 46)
Tm_steel_K = 1811.0
for T_C in [200, 400, 600, 800, 1000, 1200]:
    T_K = T_C + 273.15
    eps_dot = A_cr * (sigma**n_cr) * np.exp(-Q_cr / (R_gas * T_K))
    hom = T_K / Tm_steel_K
    print(f"  {T_C:>8} {T_K:>8.0f} {hom:>12.3f} {eps_dot:>16.3e}")

# ------------------------------------------------------------------ #
# S6: CONNECTED FRAMEWORK
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 6: THE CONNECTED FRAMEWORK")
print(SEP)

print("""
  EVERYTHING ABOVE IS THE SAME MATH:

  EXPONENTIAL:
    Creep:      eps_dot ~ exp(-Q/RT)    Arrhenius
    Transistor: Ic ~ exp(Vbe/Vt)       Shockley diode equation
    Chemistry:  k = A*exp(-Ea/RT)      rate constant
    Stats:      likelihood ~ exp(-chi^2/2) normal distribution
    GS phase:   H(nu) = exp(i*pi*D*nu^2)  dispersion transfer function

  POWER LAW:
    Creep:      eps_dot ~ sigma^n
    Diffraction: I(theta) ~ sinc^2(...)
    Noise:      PSD ~ 1/f^alpha  (flicker noise, alpha~1)
    Turbulence: E(k) ~ k^(-5/3)  Kolmogorov

  BOOLEAN + PHYSICAL:
    &&  = series resistors (both must conduct = AND)
    ||  = parallel resistors (either conducts = OR)
    ~   = complement = short to ground (NOT)
    XOR = differential pair: output high if inputs DIFFER
    NAND= universal gate = cheapest CMOS topology
    De Morgan = parallel <-> series swap when you invert

  EXPERIMENTAL DESIGN + CIRCUITS:
    Independent variable: input voltage, gate input
    Dependent variable:   output voltage, propagation delay
    Control variable:     temperature, supply voltage, load
    alpha=0.05:           5% chance of calling a design "different"
                          when it isn't -- same criterion for
                          process variation in semiconductor yield

  APPLIED MATH TOOLS USED HERE:
    Probability:   t-test, p-value, Type I/II error
    Linear algebra: Boolean as GF(2) vector space; truth table = matrix
    Calculus:       dE/dt for creep; dVbe/dT for BJT temperature
    Differential eq: thermal R-C circuit: C*dT/dt = P - (T-Ta)/Rth
    Exponentials:   all rate processes share the same kernel

  THERMAL R-C CIRCUIT (transient heating):
    C_th * dTj/dt = P_in - (Tj - Ta) / R_th_JA
    Time constant:  tau = C_th * R_th_JA
    Steady state:   Tj_ss = Ta + P_in * R_th_JA
    Same equation as RC low-pass filter:  tau = R*C, Vout_ss = Vin
    -> Thermal impedance Z_th(f) = R_th / (1 + j*omega*tau)
    -> Just a low-pass filter in the thermal domain
""")

# thermal RC transient
print("  Thermal RC step response (P=1W, Rth=50 C/W, Cth=0.01 J/C, Ta=25C):")
P_in  = 1.0    # W
Rth   = 50.0   # C/W
Cth   = 0.01   # J/C
Ta    = 25.0   # C
tau_th = Cth * Rth
Tj_ss  = Ta + P_in * Rth
t_arr  = np.array([0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]) * tau_th
print(f"  tau = {tau_th:.4f} s,   Tj_steady = {Tj_ss:.1f} C")
print(f"  {'t (ms)':>10} {'Tj (C)':>10} {'% of final':>12}")
for t in t_arr:
    Tj = Ta + P_in * Rth * (1 - np.exp(-t / tau_th))
    print(f"  {t*1e3:>10.2f} {Tj:>10.2f} {100*(Tj-Ta)/(Tj_ss-Ta):>12.1f}")

print()
print(SEP)
print("Done.")
print(SEP)
