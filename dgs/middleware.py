"""Middleware: dB, SPICE, Quartus, trig substitution, Taylor, Feynman-Griffiths-AI.

THE UNIFYING IDEA (Feynman's way of teaching):
  Every hard integral, every circuit measurement, every FPGA synthesis step
  and every layer of a neural network is the SAME mathematical object seen
  from a different angle. This module makes those connections explicit.

  dB          = log scale for ratios  ->  SNR = 10*log10(S/N)
  SPICE       = numerical ODE solver on a circuit graph (Euler/RK4)
  Quartus     = synthesis: boolean algebra -> logic cells -> bitstream
  Trig sub    = change of variable that un-hides the Pythagorean identity
  Taylor      = any smooth function IS a polynomial near a point
  Feynman     = path integral = sum over all histories = DFT on a graph
  Griffiths   = wave function |psi|^2 = intensity = |E|^2 in phase retrieval
  AI / GS     = H(f) = exp(i*pi*D*f^2) is the kernel of the path integral

Run: py -3.13 -c "from dgs.middleware import demo; demo()"
"""
import numpy as np
import sympy as sp

# ── dB toolkit ────────────────────────────────────────────────────────────────

def db(ratio, power=True):
    """Convert a linear ratio to dB.

    power=True  (default): dB = 10*log10(ratio)  -- for power, intensity, |H|^2
    power=False:           dB = 20*log10(ratio)  -- for voltage, current, |E|

    Common values to memorize:
      +3 dB  = 2x power    (doubling)
      +6 dB  = 2x voltage  (4x power)
      +10 dB = 10x power
      -3 dB  = half power  (3 dB bandwidth definition)
      0 dB   = ratio of 1  (no change)
    """
    if np.any(np.asarray(ratio) <= 0):
        raise ValueError("ratio must be positive for dB conversion")
    factor = 10.0 if power else 20.0
    return factor * np.log10(np.asarray(ratio, float))


def db_to_linear(db_val, power=True):
    """Convert dB back to linear ratio."""
    factor = 10.0 if power else 20.0
    return 10.0 ** (np.asarray(db_val, float) / factor)


def db_table():
    """Reference table: linear ratio -> dB (power scale)."""
    ratios = [0.001, 0.01, 0.1, 0.5, 1, 2, 4, 10, 100, 1000, 1e6]
    return [{"ratio": r, "dB_power": round(db(r, power=True), 2),
             "dB_voltage": round(db(r, power=False), 2)} for r in ratios]


def snr_budget(components):
    """Cascade SNR budget: sum of dB gains/losses along a signal chain.

    components: list of dicts with keys 'name', 'gain_dB'
    Positive gain_dB = amplifier. Negative = loss/attenuation.

    Example chain: laser(+20dB) -> fiber(-3dB) -> detector(-6dB) -> ADC(-3dB)
    """
    total_dB = 0.0
    chain = []
    for c in components:
        total_dB += c["gain_dB"]
        chain.append({"name": c["name"], "gain_dB": c["gain_dB"],
                      "cumulative_dB": round(total_dB, 2)})
    return {"chain": chain, "total_gain_dB": round(total_dB, 2),
            "total_linear": round(db_to_linear(total_dB), 4)}


# ── SPICE annotations ─────────────────────────────────────────────────────────

SPICE_ELEMENT_MAP = {
    "R": {"full": "Resistor",    "unit": "Ohm",   "SPICE": "R1 N+ N- 1k",
          "physics": "V=IR (Ohm's law)",
          "dB_use": "voltage divider attenuation = 20*log10(R2/(R1+R2))"},
    "C": {"full": "Capacitor",   "unit": "Farad", "SPICE": "C1 N+ N- 1n",
          "physics": "I = C dV/dt  (derivative relationship)",
          "dB_use": "RC low-pass: -3dB at f=1/(2*pi*R*C), -20dB/decade rolloff"},
    "L": {"full": "Inductor",    "unit": "Henry", "SPICE": "L1 N+ N- 1u",
          "physics": "V = L dI/dt  (integral relationship, dual of C)",
          "dB_use": "RL high-pass: +20dB/decade above f=R/(2*pi*L)"},
    "V": {"full": "Voltage src", "unit": "Volt",  "SPICE": "V1 N+ N- DC 5",
          "physics": "ideal: maintains V regardless of I",
          "dB_use": "source strength often quoted in dBm (ref 1mW into 50 ohm)"},
    "I": {"full": "Current src", "unit": "Amp",   "SPICE": "I1 N+ N- AC 1m",
          "physics": "ideal: maintains I regardless of V",
          "dB_use": "Norton equivalent of voltage source"},
    "D": {"full": "Diode",       "unit": "-",     "SPICE": "D1 A K 1N4148",
          "physics": "I = I_s*(exp(V/nVT)-1)  Shockley equation",
          "dB_use": "forward drop ~0.7V (Si); used in RF mixers, detectors"},
    "Q": {"full": "BJT",         "unit": "-",     "SPICE": "Q1 C B E 2N2222",
          "physics": "I_C = beta*I_B  (current-controlled current source)",
          "dB_use": "gain = 20*log10(beta) dB in voltage amplifier config"},
    "M": {"full": "MOSFET",      "unit": "-",     "SPICE": "M1 D G S B NMOS W=1u L=180n",
          "physics": "I_D = k*(V_GS-V_th)^2/2  (square law in saturation)",
          "dB_use": "transconductance gm in dB: gain = gm*R_D"},
}


def spice_netlist_rc_lowpass(R=1e3, C=1e-9, V_in=1.0, f_start=1e3, f_stop=1e9):
    """Generate a SPICE-compatible netlist for an RC low-pass filter.

    Also returns the analytical transfer function H(f) and -3dB frequency.
    This is the SIMPLEST dispersive element: H(f) = 1/(1 + j*2*pi*f*R*C)
    Compare to GS dispersion: H(f) = exp(i*pi*D*f^2)  -- same idea, different physics.
    """
    f3dB = 1.0 / (2 * np.pi * R * C)
    netlist = f"""* RC Low-pass filter -- generated by dgs/middleware.py
* Analytical: H(f) = 1 / (1 + j*2*pi*f*R*C)
* -3dB at f = {f3dB:.2e} Hz

V1 IN 0 AC {V_in}
R1 IN OUT {R:.0f}
C1 OUT 0 {C:.2e}

.AC DEC 20 {f_start:.0f} {f_stop:.0f}
.PROBE V(OUT) V(IN)
.END
"""
    return {"netlist": netlist, "f3dB_Hz": f3dB,
            "R": R, "C": C, "tau_s": R * C,
            "attenuation_at_f3dB_dB": db(0.5, power=True)}  # -3.01 dB


def spice_transfer_function_db(f, R, C):
    """RC low-pass |H(f)| in dB -- what SPICE .AC analysis plots."""
    H_mag2 = 1.0 / (1.0 + (2 * np.pi * f * R * C)**2)
    return db(H_mag2, power=True)


# ── Quartus / FPGA workflow ───────────────────────────────────────────────────

QUARTUS_WORKFLOW = {
    "1_design_entry": {
        "tool": "Quartus Prime (Intel/Altera) or Vivado (AMD/Xilinx)",
        "input": "VHDL or Verilog source files",
        "example_vhdl": """\
-- 2-input NAND gate in VHDL
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity nand2 is
    port (A, B : in  STD_LOGIC;
          Y    : out STD_LOGIC);
end nand2;

architecture rtl of nand2 is
begin
    Y <= A NAND B;
end rtl;""",
        "example_verilog": """\
// 2-input NAND gate in Verilog
module nand2 (
    input  A, B,
    output Y
);
    assign Y = ~(A & B);
endmodule""",
    },
    "2_synthesis": {
        "tool": "Quartus synthesis engine",
        "what_happens": [
            "Parse HDL -> abstract syntax tree",
            "Boolean minimization (Quine-McCluskey or ESPRESSO)",
            "Technology mapping -> LUT4 / LUT6 cells",
            "Output: synthesized netlist (.vqm or .edf)",
        ],
        "key_concept": "LUT (Look-Up Table): a 16-bit RAM that implements any 4-input function",
        "dB_connection": "Timing report shows slack in ns; -slack = timing violation",
    },
    "3_fitting": {
        "tool": "Quartus Fitter (place and route)",
        "what_happens": [
            "Place: assign each LUT to a physical location on the FPGA fabric",
            "Route: connect LUTs via programmable interconnect",
            "Output: .sof bitstream file",
        ],
        "key_concept": "Critical path delay -> max clock frequency f_max",
    },
    "4_timing_analysis": {
        "tool": "TimeQuest Timing Analyzer",
        "metrics": {
            "setup slack": "time available before clock edge (positive = OK)",
            "hold slack":  "minimum time data must be stable after clock (positive = OK)",
            "f_max":       "1 / (critical_path_delay + setup_time)",
        },
        "dB_connection": "Signal integrity: eye diagram margin often quoted in dBm",
    },
    "5_programming": {
        "tool": "Quartus Programmer",
        "output": "JTAG download of .sof to FPGA SRAM, or .pof to flash",
    },
    "physics_connection": {
        "LUT as truth table":    "Same truth table as transistor_logic.py nand_universality_table()",
        "FPGA flip-flop":        "D flip-flop: state machine, same math as Markov chain",
        "DSP block on FPGA":     "18x18 multiplier -> FIR filter -> same as DFT convolution",
        "FPGA for GS algorithm": "FFT IP core + custom phase-apply logic -> real-time GS at 250MHz",
    },
}


def lut_truth_table(func_bits):
    """Represent an arbitrary 4-input function as a 16-bit LUT.

    func_bits: integer 0-65535, each bit is the output for inputs (A3,A2,A1,A0) = row index.

    Example: NAND of A0 and A1 (ignoring A2,A3):
      row 0: A1=0,A0=0 -> NAND=1
      row 1: A1=0,A0=1 -> NAND=1
      row 2: A1=1,A0=0 -> NAND=1
      row 3: A1=1,A0=1 -> NAND=0
      repeats for A2,A3 combinations -> 0111 0111 0111 0111 = 0x7777
    """
    if not (0 <= func_bits <= 0xFFFF):
        raise ValueError("func_bits must be 0-65535 (16-bit)")
    rows = []
    for i in range(16):
        a = [(i >> k) & 1 for k in range(4)]
        out = (func_bits >> i) & 1
        rows.append({"A3": a[3], "A2": a[2], "A1": a[1], "A0": a[0], "out": out})
    return rows


def nand2_lut_value():
    """Return the 16-bit LUT value that implements A0 NAND A1 (A2,A3 ignored)."""
    val = 0
    for i in range(16):
        a0 = (i >> 0) & 1
        a1 = (i >> 1) & 1
        out = 1 - (a0 & a1)
        val |= (out << i)
    return val  # 0x7777


# ── Trig substitution ─────────────────────────────────────────────────────────

def trig_substitution_guide():
    """Reference table for trig substitution integrals.

    The three canonical forms and their substitutions:

    Form 1: sqrt(a^2 - x^2)  -> x = a*sin(theta)  [circle: x^2+y^2=a^2]
    Form 2: sqrt(a^2 + x^2)  -> x = a*tan(theta)  [hyperbola]
    Form 3: sqrt(x^2 - a^2)  -> x = a*sec(theta)  [outside circle]

    WHY IT WORKS: Pythagorean identity hides inside the radical.
      sin^2 + cos^2 = 1  ->  a^2 - a^2*sin^2 = a^2*cos^2  (Form 1)
      1 + tan^2 = sec^2  ->  a^2 + a^2*tan^2 = a^2*sec^2  (Form 2)
      sec^2 - 1 = tan^2  ->  a^2*sec^2 - a^2 = a^2*tan^2 (Form 3)

    PHYSICS USE: Coulomb integral for continuous charge distributions.
      E = integral of dq / r^2  often leads to sqrt(R^2 + z^2) -> Form 2.
    """
    return [
        {"form": "sqrt(a^2 - x^2)", "sub": "x = a*sin(t)",
         "dx": "a*cos(t) dt", "simplifies_to": "a*cos(t)",
         "back_sub": "t = arcsin(x/a)",
         "example": "integral dx/sqrt(1-x^2) = arcsin(x) + C",
         "griffiths": "E field on axis of uniform disk (Griffiths Ex 2.6)"},
        {"form": "sqrt(a^2 + x^2)", "sub": "x = a*tan(t)",
         "dx": "a*sec^2(t) dt", "simplifies_to": "a*sec(t)",
         "back_sub": "t = arctan(x/a)",
         "example": "integral dx/sqrt(1+x^2) = ln|x + sqrt(1+x^2)| + C",
         "griffiths": "B field on axis of solenoid (Griffiths Ex 5.6)"},
        {"form": "sqrt(x^2 - a^2)", "sub": "x = a*sec(t)",
         "dx": "a*sec(t)*tan(t) dt", "simplifies_to": "a*tan(t)",
         "back_sub": "t = arcsec(x/a)",
         "example": "integral dx/(x*sqrt(x^2-1)) = arcsec(x) + C",
         "griffiths": "Potential outside a uniformly charged sphere"},
    ]


def trig_sub_sympy(form=1, a=1):
    """Verify trig substitution symbolically for each form.

    form=1: integral of 1/sqrt(a^2-x^2) dx = arcsin(x/a)/a ... wait = arcsin(x/a) + C
    form=2: integral of 1/sqrt(a^2+x^2) dx = ln(x+sqrt(x^2+a^2)) + C
    form=3: integral of 1/sqrt(x^2-a^2) dx = ln|x+sqrt(x^2-a^2)| + C
    """
    x = sp.Symbol("x", real=True)
    a_sym = sp.Symbol("a", positive=True)
    forms = {
        1: (1 / sp.sqrt(a_sym**2 - x**2),   sp.asin(x / a_sym)),
        2: (1 / sp.sqrt(a_sym**2 + x**2),   sp.log(x + sp.sqrt(x**2 + a_sym**2))),
        3: (1 / sp.sqrt(x**2 - a_sym**2),   sp.log(sp.Abs(x + sp.sqrt(x**2 - a_sym**2)))),
    }
    if form not in forms:
        raise ValueError("form must be 1, 2, or 3")
    integrand, expected = forms[form]
    computed = sp.integrate(integrand, x)
    diff = sp.simplify(sp.diff(computed, x) - integrand)
    return {"form": form, "integrand": integrand,
            "computed": computed, "expected": expected,
            "derivative_check_residual": diff}


# ── Feynman-Griffiths-AI middleware ───────────────────────────────────────────

FEYNMAN_GRIFFITHS_AI = {
    "path_integral": {
        "Feynman (QM)":   "Z = sum over all paths exp(i*S/hbar)  [S=action]",
        "Griffiths (QM)": "|psi(x,t)|^2 = probability density",
        "Fourier":        "FT[psi] = sum over exp(i*k*x) paths in k-space",
        "GS algorithm":   "E_out = IFFT[exp(i*pi*D*f^2) * FFT[E_in]]  = one path",
        "AI (attention)": "Attention(Q,K,V) = softmax(QK^T/sqrt(d)) * V",
        "connection":     "softmax(QK^T) is a path weight; V is the 'amplitude'",
    },
    "wave_function": {
        "Griffiths":      "Schrodinger: i*hbar*dpsi/dt = H*psi",
        "EM (Ch9)":       "i*dE/dt = -(c^2/omega)*nabla^2*E  (same structure)",
        "GS phase":       "E(t) = |E(t)| * exp(i*phi(t))  (amplitude + phase)",
        "AI":             "complex-valued NN weights = same amplitude+phase structure",
        "measurement":    "I = |E|^2 = |psi|^2  (same operation: mod-square)",
    },
    "eigenvalue_problem": {
        "Griffiths":      "H*psi = E*psi  (Hamiltonian eigenstates)",
        "Linear algebra": "A*v = lambda*v  (matrix eigenvectors)",
        "PCA":            "C*v = lambda*v  (covariance matrix, lambda = variance)",
        "FNO":            "R(f)*x_hat = y_hat  (frequency-domain linear map)",
        "FPGA":           "LUT truth table = lookup of eigenvalue index",
    },
    "taylor_series": {
        "Calculus":       "f(x) = f(a) + f'(a)*(x-a) + f''(a)*(x-a)^2/2 + ...",
        "Griffiths":      "V(x) ~ V(x0) + V'*(x-x0) + V''*(x-x0)^2/2  (potential well)",
        "GVD":            "k(omega) ~ k0 + k1*(dw) + k2*(dw)^2/2  (dispersion Taylor)",
                           # k2 = beta_2 = GVD = the D in H(f)
        "AI":             "ReLU(x) ~ x for x>0 (linear approximation); Taylor enables backprop",
        "SPICE":          "Nonlinear element I(V) linearized as Taylor: gm = dI/dV|_{V0}",
    },
    "fourier_transform": {
        "Math":           "F(f) = integral f(t)*exp(-2*pi*i*f*t) dt",
        "Griffiths Ch9":  "E(k) = FT[E(x)]  (wave in k-space = momentum space)",
        "GS algorithm":   "FFT is O(N log N); the entire GS loop is 2*FFT + phase apply",
        "dB":             "Power spectrum |F(f)|^2 in dB = 10*log10|F(f)|^2",
        "FPGA":           "FFT IP core on Cyclone V: 1024-point in 4us at 200MHz",
        "AI (FNO)":       "SpectralConv1d: learned weights in Fourier domain",
    },
    "gauge_invariance": {
        "Griffiths Ch10": "E,B unchanged by phi->phi-dLambda/dt, A->A+grad(Lambda)",
        "QM":             "psi -> psi*exp(i*q*Lambda/hbar)  (local U(1) symmetry)",
        "GS algorithm":   "E and exp(i*alpha)*E give same I=|E|^2  (global phase)",
        "AI training":    "Loss function L(E) = L(exp(i*alpha)*E): degenerate minimum",
        "fix":            "Fix global phase: minimize angle(E[0]) = 0  or use |E|^2 only",
    },
}


def feynman_griffiths_ai_print():
    """Print the cross-domain connection table."""
    for concept, mapping in FEYNMAN_GRIFFITHS_AI.items():
        print(f"\n  [{concept.upper()}]")
        for domain, desc in mapping.items():
            print(f"    {domain:20s}: {desc}")


# ── Unified demo ──────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  dgs/middleware.py  --  dB / SPICE / Quartus / Trig / Feynman")
    print("=" * 65)

    print("\n--- dB table ---")
    print(f"  {'ratio':>10}  {'dB(power)':>10}  {'dB(voltage)':>12}")
    for row in db_table():
        print(f"  {row['ratio']:>10.4g}  {row['dB_power']:>10.2f}  {row['dB_voltage']:>12.2f}")

    print("\n--- SNR budget: laser -> fiber -> detector -> ADC ---")
    chain = [
        {"name": "laser_output",  "gain_dB": +20},
        {"name": "fiber_loss",    "gain_dB":  -3},
        {"name": "detector",      "gain_dB":  -6},
        {"name": "ADC_noise",     "gain_dB":  -3},
    ]
    budget = snr_budget(chain)
    for c in budget["chain"]:
        print(f"  {c['name']:20s}  {c['gain_dB']:+5.1f} dB  cumulative: {c['cumulative_dB']:+6.1f} dB")
    print(f"  Total: {budget['total_gain_dB']:+.1f} dB = {budget['total_linear']:.4f}x linear")

    print("\n--- SPICE RC low-pass (R=1kOhm, C=1nF) ---")
    rc = spice_netlist_rc_lowpass(R=1e3, C=1e-9)
    print(f"  f_3dB = {rc['f3dB_Hz']:.2e} Hz = {rc['f3dB_Hz']/1e3:.1f} kHz")
    print(f"  At f_3dB: attenuation = {rc['attenuation_at_f3dB_dB']:.2f} dB (should be -3.01)")
    print(f"  tau = {rc['tau_s']*1e9:.0f} ns")
    f_test = np.array([1e3, 1e5, 1e6, 159.2e3])
    for f in f_test:
        print(f"  |H({f:.0f} Hz)| = {spice_transfer_function_db(f,1e3,1e-9):.2f} dB")

    print("\n--- Quartus NAND2 LUT value ---")
    lut_val = nand2_lut_value()
    print(f"  NAND2 LUT init value: 0x{lut_val:04X} = 0b{lut_val:016b}")
    print("  (Each bit is the truth table output for inputs 0000..1111)")
    rows = lut_truth_table(lut_val)
    print(f"  A3 A2 A1 A0 | NAND(A0,A1)")
    for r in rows[:8]:
        print(f"   {r['A3']}  {r['A2']}  {r['A1']}  {r['A0']} |     {r['out']}")

    print("\n--- Trig substitution (SymPy verify) ---")
    for form in [1, 2]:
        r = trig_sub_sympy(form=form)
        print(f"  Form {form}: d/dx[integral] - integrand = {r['derivative_check_residual']}  (0=correct)")

    print("\n--- Feynman-Griffiths-AI connections ---")
    feynman_griffiths_ai_print()


if __name__ == "__main__":
    demo()
