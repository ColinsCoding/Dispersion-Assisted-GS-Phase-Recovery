"""Computer engineering: carriers -> diode -> transistor -> gates -> adder -> waves.

THE FULL CHAIN (one file, bottom to top):

  1. SEMICONDUCTOR PHYSICS
     - Majority carriers: electrons in n-type, holes in p-type
     - Minority carriers: holes in n-type, electrons in p-type
     - Carrier concentration: n*p = ni^2  (mass action law)

  2. P-N DIODE
     - Shockley equation: I = Is*(exp(V/VT) - 1)
     - Forward bias: majority carriers injected -> minority carriers on other side
     - Depletion region: E field opposes diffusion

  3. DIODE-RESISTOR OR GATE (DRL)
     - Two diodes + one resistor -> OR gate
     - Physics: forward-biased diode conducts; highest input wins

  4. BJT / MOSFET SWITCH  (see transistor_logic.py)
     - Base current controls collector (BJT minority carrier injection)
     - CMOS NAND (4 transistors) -> see transistor_logic.py

  5. ADDER (half -> full -> ripple carry)
     - Half adder:  S = XOR(A,B),  C = AND(A,B)  [2 gates, 9 transistors CMOS]
     - Full adder:  S = XOR(XOR(A,B),Cin),  Cout = majority(A,B,Cin)
     - MAJORITY GATE: Cout = 1 when 2 or more of {A,B,Cin} are 1
     - MINORITY GATE: output = 1 when fewer than half inputs are 1 (inverse)

  6. TRANSMISSION LINE WAVE PHYSICS
     - PCB trace = distributed L and C -> wave equation
     - Characteristic impedance Z0 = sqrt(L'/C')  [50 ohm standard]
     - Reflection coefficient: Gamma = (Z_L - Z0)/(Z_L + Z0)
     - Signal integrity: reflections cause false switching -> terminate Z_L = Z0

  7. ELECTRODYNAMICS ON A PCB
     - Return current path follows Faraday's law (least inductance)
     - Ground bounce: L * dI/dt spike when many outputs switch simultaneously
     - Decoupling capacitor: C * dV/dt = I_spike  -> C >= I*dt/dV

GRIFFITHS CONNECTION:
  Transmission line wave equation = 1D Maxwell's equations (Ch 9)
  d^2V/dx^2 = L'C' * d^2V/dt^2  identical to  d^2E/dz^2 = (1/c^2)*d^2E/dt^2
  Z0 = sqrt(L'/C')  analogous to  eta = sqrt(mu/epsilon) = 377 ohm (free space)

Run: py -3.13 -c "from dgs.computer_engineering import demo; demo()"
"""
import numpy as np
import sympy as sp

# ── Constants ─────────────────────────────────────────────────────────────────
KB   = 1.380649e-23
Q_E  = 1.602176634e-19
VT   = KB * 300.0 / Q_E   # ~0.02585 V at 300K

# ── 1. Minority / majority carriers ──────────────────────────────────────────

SI_NI = 1.5e10   # intrinsic carrier concentration, cm^-3 at 300K

def carrier_concentrations(N_D=0.0, N_A=0.0, ni=SI_NI):
    """Majority and minority carrier concentrations via mass action law.

    Mass action law: n * p = ni^2  (always true at thermal equilibrium)

    n-type (N_D > 0, donors, extra electrons):
      majority: n ~ N_D   (electrons)
      minority: p = ni^2 / N_D  (holes)

    p-type (N_A > 0, acceptors, extra holes):
      majority: p ~ N_A   (holes)
      minority: n = ni^2 / N_A  (electrons)

    WHY MINORITY CARRIERS MATTER:
      BJT action: emitter injects minority carriers into base.
      They diffuse across the thin base -> collected at collector.
      The base is p-type; the minority carriers (electrons) do the work.
      Shorter base -> less recombination -> higher beta (gain).
    """
    if N_D == 0 and N_A == 0:
        return {"type": "intrinsic", "n": ni, "p": ni, "ni": ni}
    if N_D > 0 and N_A == 0:
        n_maj = N_D
        p_min = ni**2 / N_D
        return {"type": "n-type", "majority": "electrons", "minority": "holes",
                "n_cm3": n_maj, "p_cm3": p_min, "ni": ni,
                "n_times_p": n_maj * p_min,
                "ni_squared": ni**2}
    if N_A > 0 and N_D == 0:
        p_maj = N_A
        n_min = ni**2 / N_A
        return {"type": "p-type", "majority": "holes", "minority": "electrons",
                "p_cm3": p_maj, "n_cm3": n_min, "ni": ni,
                "n_times_p": p_maj * n_min,
                "ni_squared": ni**2}
    # compensated
    net = abs(N_D - N_A)
    n = 0.5*(N_D - N_A) + np.sqrt((0.5*(N_D-N_A))**2 + ni**2)
    p = ni**2 / n
    return {"type": "compensated", "n_cm3": n, "p_cm3": p,
            "n_times_p": n*p, "ni_squared": ni**2}


# ── 2. Diode ─────────────────────────────────────────────────────────────────

def diode_iv(V_arr, I_s=1e-12, n=1.0, T=300.0):
    """Shockley diode equation over an array of voltages.

    I(V) = I_s * (exp(V / n*VT) - 1)

    Key points:
      V = 0:      I = 0          (no current at zero bias)
      V = 0.6V:   I >> I_s       (Si diode turns on)
      V = -inf:   I = -I_s       (reverse saturation current)
      V = VT*ln2: I = I_s        (crossover from reverse to forward)
    """
    vt = KB * T / Q_E
    V = np.asarray(V_arr, float)
    I = I_s * (np.exp(np.clip(V / (n * vt), -500, 500)) - 1.0)
    return I


def diode_small_signal(V_bias, I_s=1e-12, T=300.0):
    """Small-signal conductance gd = dI/dV|_{V_bias} = I_s * exp(V/VT) / VT.

    At forward bias V_bias >> VT:  gd ~ I_DC / VT
    This is the linearized SPICE model (Taylor expansion around operating point).
    """
    vt = KB * T / Q_E
    I_dc = diode_iv(V_bias, I_s=I_s, T=T)
    gd = (I_dc + I_s) / vt
    r_d = 1.0 / (gd + 1e-300)
    return {"gd_S": float(gd), "r_d_ohm": float(r_d),
            "I_dc_A": float(I_dc), "V_bias": V_bias}


# ── 3. Diode-resistor OR gate (DRL) ──────────────────────────────────────────

def diode_or_gate(A, B, V_high=5.0, V_low=0.0, V_D=0.7, R=1e3):
    """Diode-resistor logic OR gate.

    Circuit:
      A ---[D1]---+--- OUT
      B ---[D2]---+
                  |
                 [R]
                  |
                 GND

    Logic: OUT = A OR B
    Physics: whichever input is HIGH forward-biases its diode;
             current flows through R; V_out = V_high - V_D
    Limitation of DRL: V_out = V_high - 0.7V (voltage degradation).
    CMOS OR = NAND(NAND(A,A), NAND(B,B)) -- no voltage drop.
    """
    va = V_high if A else V_low
    vb = V_high if B else V_low
    v_in_max = max(va, vb)
    if v_in_max > V_D:
        v_out = v_in_max - V_D
        conducting = []
        if va > V_D: conducting.append("D1")
        if vb > V_D: conducting.append("D2")
        i_out = v_out / R
    else:
        v_out = 0.0
        conducting = []
        i_out = 0.0
    logic_out = 1 if v_out > (V_high / 2) else 0
    return {"A": A, "B": B, "V_out": round(v_out, 3),
            "logic_out": logic_out, "conducting": conducting,
            "I_mA": round(i_out * 1e3, 3)}


def diode_or_truth_table(V_high=5.0):
    return [diode_or_gate(a, b, V_high=V_high)
            for a in (0, 1) for b in (0, 1)]


# ── 4. Majority / minority gate (adder logic) ─────────────────────────────────

def majority_gate(A, B, C):
    """Majority gate: output 1 when 2 or more of {A,B,C} are 1.

    This IS the carry-out of a full adder:
      Cout = majority(A, B, Cin) = AB + BCin + ACin

    In CMOS: implemented as AOI (AND-OR-INVERT) + inverter = 6 transistors.
    In transmission gate logic: 4 transistors.
    In threshold logic: a single threshold gate with weights [1,1,1], threshold=2.
    """
    total = int(bool(A)) + int(bool(B)) + int(bool(C))
    return 1 if total >= 2 else 0


def minority_gate(A, B, C):
    """Minority gate: output 1 when fewer than 2 of {A,B,C} are 1.

    minority(A,B,C) = NOT majority(A,B,C)

    Used in: fault-tolerant logic (TMR), quantum-dot cellular automata (QCA),
    spintronic logic where NAND is not the primitive.
    """
    return 1 - majority_gate(A, B, C)


def half_adder(A, B):
    """Half adder: S = XOR(A,B), C = AND(A,B).

    CMOS transistor count: XOR=8 + AND=6 = 14T  (or XOR+NAND = 10T optimized)
    No carry-in; cascades into full adder.
    """
    s = int(bool(A)) ^ int(bool(B))
    c = int(bool(A)) & int(bool(B))
    return {"S": s, "Cout": c, "A": int(bool(A)), "B": int(bool(B))}


def full_adder(A, B, Cin):
    """Full adder: S = XOR(XOR(A,B),Cin), Cout = majority(A,B,Cin).

    CMOS transistor count: ~28T (two XOR + majority)
    Optimized: 20T with transmission gate XOR.

    KEY INSIGHT: Cout = majority(A,B,Cin)
    This is why majority gate IS the adder carry.
    """
    ab = int(bool(A)) ^ int(bool(B))
    s  = ab ^ int(bool(Cin))
    cout = majority_gate(A, B, Cin)
    return {"S": s, "Cout": cout,
            "A": int(bool(A)), "B": int(bool(B)), "Cin": int(bool(Cin))}


def ripple_carry_adder(a_bits, b_bits):
    """N-bit ripple carry adder using full_adder cells.

    Delay: O(N) -- each carry must ripple through all N stages.
    Delay_total = N * t_FA  where t_FA ~ 2 gate delays per full adder.
    For 64-bit: 128 gate delays ~ 1.3ns at 100ps/gate -> limits clock to 750MHz.
    Carry-lookahead (CLA) reduces to O(log N).
    """
    if len(a_bits) != len(b_bits):
        raise ValueError("a_bits and b_bits must be same length")
    n = len(a_bits)
    result = []
    carry = 0
    for i in range(n - 1, -1, -1):
        fa = full_adder(a_bits[i], b_bits[i], carry)
        result.append(fa["S"])
        carry = fa["Cout"]
    result.reverse()
    return {"sum_bits": result, "carry_out": carry,
            "sum_int": int("".join(str(b) for b in result), 2) + carry * (2**n)}


def adder_truth_table_4bit():
    """All entries of half adder and full adder truth tables."""
    ha = [half_adder(a, b) for a in (0,1) for b in (0,1)]
    fa = [full_adder(a, b, c) for a in (0,1) for b in (0,1) for c in (0,1)]
    return {"half_adder": ha, "full_adder": fa}


# ── 5. Transmission line wave physics ────────────────────────────────────────

def transmission_line(Z0=50.0, Z_L=50.0, length_m=0.1,
                      freq_Hz=1e9, v_prop=2e8):
    """Transmission line analysis: reflection, standing waves, signal integrity.

    Wave equation on a PCB trace:
      d^2V/dx^2 = L'C' * d^2V/dt^2

    This is IDENTICAL to Maxwell's 1D wave equation (Griffiths Ch 9):
      d^2E/dz^2 = (1/c^2) * d^2E/dt^2

    with  c -> v_prop = 1/sqrt(L'C'),  eta -> Z0 = sqrt(L'/C')

    Z0 = 50 ohm: industry standard (compromise between power capacity and loss)
    Z0 = 75 ohm: cable TV, video
    Z0 = 100 ohm: differential pairs (USB, HDMI, PCIe)
    """
    if Z0 <= 0:
        raise ValueError("Z0 must be positive")
    # reflection coefficient at load
    if Z_L == float('inf'):
        Gamma = 1.0    # open circuit: total reflection, V doubles
    elif Z_L == 0:
        Gamma = -1.0   # short circuit: total reflection, V inverts
    else:
        Gamma = (Z_L - Z0) / (Z_L + Z0)

    # electrical length
    wavelength = v_prop / freq_Hz
    elec_length_deg = (length_m / wavelength) * 360.0

    # VSWR (Voltage Standing Wave Ratio)
    vswr = (1 + abs(Gamma)) / (1 - abs(Gamma) + 1e-12)

    # return loss in dB
    if abs(Gamma) > 1e-10:
        return_loss_dB = -20.0 * np.log10(abs(Gamma))
    else:
        return_loss_dB = float('inf')

    return {
        "Z0_ohm": Z0,
        "Z_L_ohm": Z_L,
        "Gamma": Gamma,               # reflection coefficient
        "VSWR": round(vswr, 3),
        "return_loss_dB": round(return_loss_dB, 2),
        "elec_length_deg": round(elec_length_deg, 2),
        "wavelength_m": round(wavelength, 4),
        "matched": abs(Gamma) < 0.05,
        "signal_integrity": "OK" if abs(Gamma) < 0.1 else "REFLECTIONS -- terminate!",
    }


def pcb_trace_impedance(w_mm, h_mm=1.6, er=4.5):
    """Microstrip characteristic impedance (IPC-2141 approximation).

    w: trace width (mm)
    h: dielectric thickness to ground plane (mm)
    er: relative permittivity (FR4 = 4.5 typical)

    Z0 ~ (87 / sqrt(er + 1.41)) * ln(5.98*h / (0.8*w + t))
    Simplified (t << w):
    """
    if w_mm <= 0 or h_mm <= 0 or er <= 0:
        raise ValueError("w, h, er must be positive")
    # Wadell approximation
    w_h = w_mm / h_mm
    if w_h <= 1:
        Z0 = (60.0 / np.sqrt(er)) * np.log(8.0 / w_h + 0.25 * w_h)
    else:
        Z0 = (120.0 * np.pi / np.sqrt(er)) / (w_h + 1.393 + 0.667*np.log(w_h + 1.444))
    v_prop = 3e8 / np.sqrt(er)
    return {"Z0_ohm": round(Z0, 1), "w_mm": w_mm, "h_mm": h_mm,
            "er": er, "v_prop_m_s": round(v_prop, 3),
            "tip": "widen trace to lower Z0; narrow to raise Z0"}


def ground_bounce(n_outputs, I_per_output_A=0.02, L_pkg_nH=3.0, t_rise_ns=0.5):
    """Ground bounce (simultaneous switching noise) from package inductance.

    V_bounce = L * dI/dt = L_pkg * (n * I_per_output) / t_rise

    This is Faraday's law (Griffiths Ch 7) applied to the bond wire inductance.
    When n outputs switch 0->1 simultaneously, a spike appears on the ground pin.
    Fix: decoupling capacitor C >= I_total * t_rise / V_allowed_bounce.
    """
    if t_rise_ns <= 0:
        raise ValueError("t_rise must be positive")
    L = L_pkg_nH * 1e-9
    dt = t_rise_ns * 1e-9
    dI = n_outputs * I_per_output_A
    V_bounce = L * dI / dt
    C_needed_nF = (dI * dt / 0.1) * 1e9   # allow 100mV bounce
    return {
        "V_bounce_mV": round(V_bounce * 1e3, 1),
        "n_outputs": n_outputs,
        "L_pkg_nH": L_pkg_nH,
        "t_rise_ns": t_rise_ns,
        "C_decoupling_nF": round(C_needed_nF, 2),
        "faraday": "V = L * dI/dt  (Griffiths eq 7.17)",
        "fix": f"Place {C_needed_nF:.1f} nF decoupling cap within 2mm of power pin",
    }


# ── 6. EM wave - digital logic connection (sympy) ────────────────────────────

def wave_equation_pcb_sympy():
    """Show PCB transmission line = 1D Maxwell wave equation.

    Telegrapher's equations:
      -dV/dx = L' * dI/dt
      -dI/dx = C' * dV/dt

    Combine -> wave equation:
      d^2V/dx^2 = L'C' * d^2V/dt^2

    Solution: V(x,t) = V+ * exp(i*(omega*t - k*x)) + V- * exp(i*(omega*t + k*x))
    k = omega * sqrt(L'C') = omega / v_prop
    Z0 = V+ / I+ = sqrt(L'/C')
    """
    x, t = sp.symbols("x t", real=True)
    omega, k = sp.symbols("omega k", positive=True)
    Lp, Cp = sp.symbols("L_prime C_prime", positive=True)  # per-unit-length
    V0 = sp.Symbol("V_0")

    # forward traveling wave
    V_plus = V0 * sp.exp(sp.I * (omega * t - k * x))

    # wave equation residual
    lhs = sp.diff(V_plus, x, 2)
    rhs = Lp * Cp * sp.diff(V_plus, t, 2)
    residual = sp.simplify(lhs - rhs)

    # dispersion relation k = omega*sqrt(L'C')
    k_val = omega * sp.sqrt(Lp * Cp)
    residual_at_k = sp.simplify(residual.subs(k, k_val))

    Z0_sym = sp.sqrt(Lp / Cp)

    return {
        "V_forward": V_plus,
        "wave_eq_residual": residual,
        "dispersion_relation": sp.Eq(k, k_val),
        "residual_at_dispersion": residual_at_k,
        "Z0": Z0_sym,
        "griffiths_analogy": "Same as d^2E/dz^2 = (1/c^2)*d^2E/dt^2 with c->v_prop, eta->Z0",
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  dgs/computer_engineering.py  --  demo")
    print("=" * 65)

    print("\n--- 1. Minority / Majority Carriers (Si, 300K) ---")
    for ND, NA, label in [(1e16, 0, "n-type N_D=1e16"), (0, 1e17, "p-type N_A=1e17")]:
        r = carrier_concentrations(N_D=ND, N_A=NA)
        t = r["type"]
        n = r.get("n_cm3", r.get("n", 0))
        p = r.get("p_cm3", r.get("p", 0))
        print(f"  {label:22s}: n={n:.2e}, p={p:.2e}  n*p={n*p:.2e} (ni^2={SI_NI**2:.2e})")
    print("  Majority carriers conduct. Minority carriers control BJT gain.")

    print("\n--- 2. Diode IV (Si, Is=1pA) ---")
    for V in [-1.0, 0.0, 0.3, 0.6, 0.7, 1.0]:
        I = diode_iv(V, I_s=1e-12)
        print(f"  V={V:+.1f}V  I={I:.3e} A  "
              f"({'forward' if V > 0.5 else 'reverse' if V < 0 else 'near-zero'})")
    ss = diode_small_signal(0.65)
    print(f"  Small-signal at 0.65V: r_d={ss['r_d_ohm']:.1f} ohm, gm={ss['gd_S']*1e3:.2f} mS")

    print("\n--- 3. Diode-Resistor OR gate ---")
    print("  A  B  V_out    logic  conducting")
    for r in diode_or_truth_table():
        print(f"  {r['A']}  {r['B']}  {r['V_out']:.2f}V    "
              f"  {r['logic_out']}    {r['conducting']}")
    print("  Note: V_out=4.3V not 5.0V -- 0.7V diode drop (DRL degrades voltage)")

    print("\n--- 4. Majority / Minority gate + Full Adder ---")
    print("  Majority gate (= carry-out of full adder):")
    print("  A  B  Cin | Majority(=Cout) | Minority")
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                maj = majority_gate(a,b,c)
                mn  = minority_gate(a,b,c)
                print(f"  {a}  {b}   {c}  |       {maj}        |    {mn}")

    print("\n  Full adder truth table (S, Cout):")
    print("  A  B  Cin | S  Cout")
    for r in adder_truth_table_4bit()["full_adder"]:
        print(f"  {r['A']}  {r['B']}   {r['Cin']}  | {r['S']}   {r['Cout']}")

    print("\n  4-bit ripple carry: 0110 + 0111 =")
    result = ripple_carry_adder([0,1,1,0], [0,1,1,1])
    print(f"  sum={result['sum_bits']}  carry_out={result['carry_out']}  "
          f"decimal={result['sum_int']}  (6+7={6+7})")

    print("\n--- 5. Transmission line / PCB trace ---")
    cases = [("matched Z_L=50",50), ("open Z_L=inf",1e9), ("short Z_L=0",0.001), ("mismatch Z_L=100",100)]
    for label, ZL in cases:
        r = transmission_line(Z0=50, Z_L=ZL, freq_Hz=1e9)
        print(f"  {label:22s}: Gamma={r['Gamma']:+.2f}  VSWR={r['VSWR']:.2f}  "
              f"RL={r['return_loss_dB']:.1f}dB  {r['signal_integrity']}")

    print("\n  PCB trace width for 50 ohm (FR4, h=1.6mm):")
    for w in [0.5, 1.0, 2.0, 3.0]:
        r = pcb_trace_impedance(w, h_mm=1.6)
        print(f"  w={w}mm -> Z0={r['Z0_ohm']:.1f} ohm")

    print("\n--- 6. Ground bounce (Faraday's law on chip) ---")
    gb = ground_bounce(n_outputs=8, I_per_output_A=0.02, L_pkg_nH=3, t_rise_ns=0.5)
    print(f"  8 outputs switch simultaneously:")
    print(f"  V_bounce = L*dI/dt = {gb['V_bounce_mV']:.0f} mV  ({gb['faraday']})")
    print(f"  Fix: {gb['fix']}")

    print("\n--- 7. PCB wave equation = Maxwell (SymPy) ---")
    r = wave_equation_pcb_sympy()
    print(f"  V(x,t) =", end=" "); sp.pprint(r["V_forward"])
    print(f"  Dispersion: ", end=""); sp.pprint(r["dispersion_relation"])
    print(f"  Residual at k=omega*sqrt(LC): {r['residual_at_dispersion']}  (0=correct)")
    print(f"  Z0 = ", end=""); sp.pprint(r["Z0"])
    print(f"  {r['griffiths_analogy']}")


if __name__ == "__main__":
    demo()
