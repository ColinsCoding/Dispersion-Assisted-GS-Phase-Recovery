"""Transistor physics -> NAND gate -> all digital logic.

THE CHAIN:
  Physics (doping, band gap)
    -> BJT / MOSFET as a voltage-controlled switch
      -> CMOS NAND (2 PMOS pull-up + 2 NMOS pull-down)
        -> NAND is universal: NOT, AND, OR, XOR all from NAND only
          -> half adder, full adder, ripple carry (dgs/digital_logic.py)
            -> ALU, CPU, every computer ever built

GRIFFITHS CONNECTION:
  The p-n junction (diode) lives in Griffiths Ch 4 (polarization/dielectrics)
  and the Kronig-Penney model (Ch 2 / quantum). The band gap E_g separates
  the valence band (filled) from the conduction band (empty).
  For Si: E_g = 1.12 eV at 300K.
  Doping shifts the Fermi level: n-type (donor, extra e-) raises E_F toward
  conduction band; p-type (acceptor, hole) lowers E_F toward valence band.

JACKSON CONNECTION:
  Jackson Ch 3 (boundary value problems) -> depletion region E field
  Jackson Ch 5 (magnetostatics) -> current flow in the channel
  Jackson Ch 7 (plane waves) -> signal propagation on the PCB trace

Run: py -3.13 -c "from dgs.transistor_logic import demo; demo()"
"""
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────

KB   = 1.380649e-23   # J/K
Q_E  = 1.602176634e-19  # C (electron charge)
T300 = 300.0          # K (room temperature)
VT   = KB * T300 / Q_E  # thermal voltage ~25.85 mV

# ── Semiconductor physics ─────────────────────────────────────────────────────

MATERIALS = {
    "Si":  {"E_g_eV": 1.12,  "mu_n": 1400, "mu_p": 450,  "ni_cm3": 1.5e10},
    "Ge":  {"E_g_eV": 0.66,  "mu_n": 3900, "mu_p": 1900, "ni_cm3": 2.4e13},
    "GaAs":{"E_g_eV": 1.42,  "mu_n": 8500, "mu_p": 400,  "ni_cm3": 2.0e6},
    "GaN": {"E_g_eV": 3.4,   "mu_n": 1000, "mu_p": 30,   "ni_cm3": 1e-10},
}


def fermi_dirac(E_eV, E_F_eV, T=300.0):
    """Fermi-Dirac distribution: probability that state at energy E is occupied.

    f(E) = 1 / (1 + exp((E - E_F) / kT))

    At T=0: step function (all below E_F filled, all above empty).
    At T=300K: thermal smearing of ~kT = 25.85 meV around E_F.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    kT = KB * T / Q_E  # eV
    x = (E_eV - E_F_eV) / kT
    return 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))


def pn_junction_voltage(N_D, N_A, material="Si", T=300.0):
    """Built-in voltage of a p-n junction (contact potential).

    V_bi = (kT/q) * ln(N_D * N_A / ni^2)

    This is the barrier the transistor switch must overcome.
    For Si with N_D=N_A=1e16 cm^-3: V_bi ~ 0.72 V.
    """
    if N_D <= 0 or N_A <= 0:
        raise ValueError("doping concentrations must be positive")
    if material not in MATERIALS:
        raise ValueError(f"material must be one of {list(MATERIALS)}")
    ni = MATERIALS[material]["ni_cm3"]
    E_g = MATERIALS[material]["E_g_eV"]
    kT_over_q = KB * T / Q_E
    V_bi = kT_over_q * np.log(N_D * N_A / ni**2)
    return {"V_bi_V": float(V_bi), "kT_eV": float(kT_over_q),
            "E_g_eV": E_g, "ni_cm3": ni, "material": material}


def diode_current(V, I_s=1e-12, T=300.0, n=1.0):
    """Shockley diode equation: I = I_s * (exp(V / n*VT) - 1).

    n=1: ideal (diffusion current dominates)
    n=2: recombination current dominates (low forward bias)
    Forward bias V > 0.6V (Si): exponential on, transistor base-emitter junction.
    Reverse bias V < 0: I ~ -I_s (leakage, very small).
    """
    if n <= 0:
        raise ValueError("ideality factor n must be > 0")
    vt = KB * T / Q_E
    return float(I_s * (np.exp(np.clip(V / (n * vt), -500, 500)) - 1.0))


# ── BJT (Bipolar Junction Transistor) ────────────────────────────────────────

def bjt_switch(V_in, V_CC=5.0, R_B=10e3, R_C=1e3,
               beta=100, V_BE_on=0.7, V_CE_sat=0.2):
    """NPN BJT as a digital switch (common-emitter configuration).

    V_in -> R_B -> Base -> Emitter (ground)
                   |
                  R_C -> Collector -> V_CC

    Cutoff  (V_in < V_BE_on):  I_B=0, I_C=0,  V_out = V_CC  (logic HIGH)
    Active  (linear region):   I_C = beta * I_B
    Saturation (I_C limited):  V_out = V_CE_sat              (logic LOW)

    PHYSICS: the base-emitter junction is a forward-biased p-n diode.
    A small base current controls a large collector current (current amplifier).
    In digital use, we drive it hard into saturation (on) or cutoff (off).
    """
    if V_in < V_BE_on:
        # cutoff: transistor OFF
        mode = "cutoff"
        I_B = 0.0
        I_C = 0.0
        V_out = V_CC
    else:
        I_B = (V_in - V_BE_on) / R_B
        I_C_active = beta * I_B
        I_C_sat = (V_CC - V_CE_sat) / R_C
        if I_C_active >= I_C_sat:
            # saturation: transistor ON (switch closed)
            mode = "saturation"
            I_C = I_C_sat
            V_out = V_CE_sat
        else:
            mode = "active"
            I_C = I_C_active
            V_out = V_CC - I_C * R_C
    return {"mode": mode, "V_in": V_in, "V_out": float(V_out),
            "I_B_uA": float(I_B * 1e6), "I_C_mA": float(I_C * 1e3),
            "logic_out": 1 if V_out > V_CC / 2 else 0}


# ── MOSFET switch ─────────────────────────────────────────────────────────────

def nmos_switch(V_GS, V_DS=5.0, V_th=1.0, k=2e-3):
    """NMOS transistor as a switch (enhancement mode).

    V_GS < V_th:  cutoff  (OFF)  -- gate voltage below threshold
    V_GS > V_th AND V_DS < V_GS-V_th: linear (triode, switch ON)
    V_GS > V_th AND V_DS > V_GS-V_th: saturation (current source, amplifier)

    PHYSICS: V_GS creates an electric field (E = V_GS / t_ox) across the
    gate oxide, attracting electrons to form a conductive channel.
    This is a CAPACITOR effect -- no DC gate current flows.
    Why MOSFET wins in digital vs BJT: zero gate current = zero static power.

    k = mu_n * C_ox * (W/L)  -- process transconductance * geometry
    """
    if V_GS < V_th:
        return {"mode": "cutoff", "I_D_mA": 0.0, "V_GS": V_GS,
                "logic_out": 1}
    V_ov = V_GS - V_th  # overdrive voltage
    if V_DS < V_ov:
        mode = "linear"
        I_D = k * (V_ov * V_DS - 0.5 * V_DS**2)
    else:
        mode = "saturation"
        I_D = 0.5 * k * V_ov**2
    return {"mode": mode, "I_D_mA": float(I_D * 1e3),
            "V_GS": V_GS, "V_ov": float(V_ov),
            "logic_out": 0}  # drain pulled low when ON


# ── CMOS NAND gate ────────────────────────────────────────────────────────────

def cmos_nand_transistors():
    """Describe the transistor-level circuit of a 2-input CMOS NAND gate.

    CMOS NAND (4 transistors total):
      Pull-up network  (PMOS, parallel):  MP1 (gate=A), MP2 (gate=B)
      Pull-down network (NMOS, series):   MN1 (gate=A) -> MN2 (gate=B)

    Truth table:
      A=0, B=0: MP1 ON, MP2 ON  -> output HIGH  (NAND=1)
      A=0, B=1: MP1 ON, MP2 OFF -> output HIGH  (NAND=1)
      A=1, B=0: MP1 OFF,MP2 ON  -> output HIGH  (NAND=1)
      A=1, B=1: MP1 OFF,MP2 OFF -> output LOW   (NAND=0)  [only case with low]

    WHY NAND IS UNIVERSAL:
      NOT A      = NAND(A, A)
      AND(A,B)   = NOT(NAND(A,B))       = NAND(NAND(A,B), NAND(A,B))
      OR(A,B)    = NAND(NOT A, NOT B)   = NAND(NAND(A,A), NAND(B,B))
      XOR(A,B)   = OR(AND(A,NOT B), AND(NOT A,B))  [4 NANDs]
    """
    return {
        "gate": "CMOS NAND2",
        "transistor_count": 4,
        "pmos_pull_up": ["MP1(gate=A)", "MP2(gate=B)"],
        "nmos_pull_down": ["MN1(gate=A) series MN2(gate=B)"],
        "topology": "pull-up PARALLEL, pull-down SERIES",
        "truth_table": [
            {"A": 0, "B": 0, "MP1": "ON",  "MP2": "ON",  "MN_path": "open", "out": 1},
            {"A": 0, "B": 1, "MP1": "ON",  "MP2": "OFF", "MN_path": "open", "out": 1},
            {"A": 1, "B": 0, "MP1": "OFF", "MP2": "ON",  "MN_path": "open", "out": 1},
            {"A": 1, "B": 1, "MP1": "OFF", "MP2": "OFF", "MN_path": "closed","out": 0},
        ],
        "universality": "NAND alone can implement NOT, AND, OR, XOR",
    }


def nand_universality_table():
    """Verify that NOT/AND/OR/XOR can all be built from NAND gates only."""
    def NAND(a, b): return 1 - (a & b)
    def NOT_from_nand(a):       return NAND(a, a)
    def AND_from_nand(a, b):    return NOT_from_nand(NAND(a, b))
    def OR_from_nand(a, b):     return NAND(NOT_from_nand(a), NOT_from_nand(b))
    def XOR_from_nand(a, b):
        n = NAND(a, b)
        return NAND(NAND(a, n), NAND(b, n))

    rows = []
    for a in (0, 1):
        for b in (0, 1):
            rows.append({
                "A": a, "B": b,
                "NOT_A":  NOT_from_nand(a),
                "AND":    AND_from_nand(a, b),
                "OR":     OR_from_nand(a, b),
                "XOR":    XOR_from_nand(a, b),
                "NAND":   NAND(a, b),
                "verified_AND":  AND_from_nand(a, b) == (a & b),
                "verified_OR":   OR_from_nand(a, b)  == (a | b),
                "verified_XOR":  XOR_from_nand(a, b) == (a ^ b),
            })
    return rows


# ── Energy per switching event ────────────────────────────────────────────────

def cmos_switching_energy(C_load_fF=10.0, V_DD=1.0, f_GHz=3.0, alpha=0.1):
    """Dynamic power dissipation in CMOS: P = alpha * C * V^2 * f.

    C_load: load capacitance (fF)  -- mostly gate capacitance of next stage
    V_DD:   supply voltage (V)
    f_GHz:  clock frequency (GHz)
    alpha:  activity factor (fraction of cycles with a 0->1 transition)

    Each 0->1 transition charges C to V_DD: energy = 0.5*C*V^2 from supply.
    Each 1->0 transition discharges C through NMOS to ground: 0.5*C*V^2 wasted.
    Total per cycle: C*V^2. Average power: P = alpha*C*V^2*f.

    PHOTONIC AI COMPARISON (photonic_ai.py):
      CMOS NAND at 3GHz, 10fF, 1V:  P ~ 3e-18 J/op (3 aJ)
      But an ADC at 20GSa/s costs ~500 fJ/sample = 1e5x more.
      The ADC is the bottleneck, not the logic -- exactly why photonic
      time-stretch ADC is the innovation: it does analog preprocessing
      (the dispersion H(f)) BEFORE the ADC, reducing its effective rate.
    """
    if C_load_fF <= 0 or V_DD <= 0 or f_GHz <= 0:
        raise ValueError("C, V, f must be positive")
    C = C_load_fF * 1e-15
    f = f_GHz * 1e9
    E_per_transition_J = C * V_DD**2
    P_dynamic_W = alpha * C * V_DD**2 * f
    return {
        "E_per_transition_fJ": float(E_per_transition_J * 1e15),
        "P_dynamic_mW": float(P_dynamic_W * 1e3),
        "C_load_fF": C_load_fF,
        "V_DD_V": V_DD,
        "f_GHz": f_GHz,
        "alpha": alpha,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  dgs/transistor_logic.py  --  demo")
    print("=" * 65)

    print("\n--- Fermi-Dirac at E_F (should be 0.5) ---")
    f = fermi_dirac(0.0, E_F_eV=0.0, T=300)
    print(f"  f(E=E_F, T=300K) = {f:.4f}  (theory: 0.5000)")

    print("\n--- p-n junction built-in voltage (Si, N_D=N_A=1e16) ---")
    pn = pn_junction_voltage(1e16, 1e16, "Si")
    print(f"  V_bi = {pn['V_bi_V']:.3f} V  (typical Si: ~0.72 V)")
    print(f"  kT   = {pn['kT_eV']*1000:.2f} meV  (thermal voltage at 300K)")

    print("\n--- BJT switch (NPN, V_CC=5V) ---")
    for v in [0.0, 0.7, 1.0, 2.0]:
        r = bjt_switch(v)
        print(f"  V_in={v:.1f}V  mode={r['mode']:12s}  "
              f"V_out={r['V_out']:.2f}V  logic={r['logic_out']}")

    print("\n--- NMOS switch ---")
    for vgs in [0.0, 0.5, 1.0, 2.0, 3.0]:
        r = nmos_switch(vgs)
        print(f"  V_GS={vgs:.1f}V  mode={r['mode']:12s}  "
              f"I_D={r['I_D_mA']:.2f} mA  logic_out={r['logic_out']}")

    print("\n--- CMOS NAND truth table ---")
    info = cmos_nand_transistors()
    print(f"  {info['gate']}  ({info['transistor_count']} transistors)")
    print(f"  Pull-up: {info['topology'].split(',')[0]}")
    print(f"  Pull-down: {info['topology'].split(',')[1]}")
    print(f"  A  B  MP1   MP2   out")
    for row in info["truth_table"]:
        print(f"  {row['A']}  {row['B']}  {row['MP1']:<5} {row['MP2']:<5} {row['out']}")

    print("\n--- NAND universality (all gates from NAND only) ---")
    rows = nand_universality_table()
    print(f"  A  B  NAND  NOT_A  AND  OR  XOR  verified")
    for r in rows:
        ok = all([r["verified_AND"], r["verified_OR"], r["verified_XOR"]])
        print(f"  {r['A']}  {r['B']}   {r['NAND']}     {r['NOT_A']}     "
              f"  {r['AND']}   {r['OR']}   {r['XOR']}    {'OK' if ok else 'FAIL'}")

    print("\n--- CMOS switching energy ---")
    e = cmos_switching_energy(C_load_fF=10, V_DD=1.0, f_GHz=3.0, alpha=0.1)
    print(f"  E/transition = {e['E_per_transition_fJ']:.1f} fJ")
    print(f"  P_dynamic    = {e['P_dynamic_mW']:.4f} mW")
    print(f"  ADC at 20 GSa/s ~ 500 fJ/sample = {500/e['E_per_transition_fJ']:.0f}x more")
    print(f"  -> fiber dispersion (free) reduces ADC load: that's the project")


if __name__ == "__main__":
    demo()
