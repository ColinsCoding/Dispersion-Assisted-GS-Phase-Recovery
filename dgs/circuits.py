"""Circuit analysis as calculus: integrals and ODEs for RC / RL / RLC.

The two reactive elements are defined by calculus:
    capacitor:  V_C = (1/C) integral i dt        (voltage is the integral of current)
    inductor:   V_L = L di/dt                     (voltage is the derivative)
so every circuit is a differential/integral equation. An RC low-pass is an
*integrator*; its 3 dB bandwidth f = 1/(2 pi R C) is exactly what limits how
fast the carrier-less receiver can sample I1, I2 (the detector front-end), and
sets the anti-alias filter before the ADC.

SymPy evaluates the integrals/ODEs symbolically; NumPy gives the step responses.
Civilian electronics / education.
"""

import numpy as np
import sympy as sp


# ── the defining calculus of the reactive elements ─────────────────
def capacitor_voltage(i_expr, t, C, v0=0):
    """V_C(t) = v0 + (1/C) integral_0^t i(t') dt' -- the capacitor integrates current."""
    tau = sp.Symbol("tau", real=True)
    return v0 + sp.integrate(i_expr.subs(t, tau), (tau, 0, t)) / C


def inductor_voltage(i_expr, t, L):
    """V_L(t) = L di/dt -- the inductor differentiates current."""
    return L * sp.diff(i_expr, t)


# ── RC: the first-order workhorse (and the detector front-end) ─────
def solve_rc_symbolic():
    """Solve R C dV/dt + V = V_in (step input, V(0)=0) -> V(t)=V_in(1-e^{-t/RC})."""
    t = sp.Symbol("t", positive=True)
    R, C, Vin = sp.symbols("R C V_in", positive=True)
    V = sp.Function("V")
    ode = sp.Eq(R * C * V(t).diff(t) + V(t), Vin)
    return sp.dsolve(ode, V(t), ics={V(0): 0})


def rc_step(t, R, C, Vin=1.0):
    """Capacitor voltage charging through R: Vin (1 - e^{-t/RC}). tau = RC."""
    if R <= 0 or C <= 0:
        raise ValueError("R, C must be > 0")
    return Vin * (1.0 - np.exp(-np.asarray(t, dtype=float) / (R * C)))


def rc_bandwidth(R, C):
    """-3 dB cutoff of an RC low-pass: f = 1/(2 pi R C). The detector's speed limit."""
    if R <= 0 or C <= 0:
        raise ValueError("R, C must be > 0")
    return 1.0 / (2 * np.pi * R * C)


# ── RL and RLC ──────────────────────────────────────────────────────
def rl_step(t, R, L, Vin=1.0):
    """Inductor-circuit current: (Vin/R)(1 - e^{-t/(L/R)}). tau = L/R."""
    if R <= 0 or L <= 0:
        raise ValueError("R, L must be > 0")
    return (Vin / R) * (1.0 - np.exp(-np.asarray(t, dtype=float) * R / L))


def rlc_damping(R, L, C):
    """Series-RLC damping: returns (omega0, zeta, regime).

    omega0 = 1/sqrt(LC), zeta = (R/2) sqrt(C/L). zeta<1 underdamped (rings),
    zeta=1 critically damped (fastest no overshoot), zeta>1 overdamped.
    """
    if R < 0 or L <= 0 or C <= 0:
        raise ValueError("L, C > 0 and R >= 0")
    omega0 = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)
    regime = ("underdamped" if zeta < 1 - 1e-12 else
              "critically damped" if abs(zeta - 1) <= 1e-12 else "overdamped")
    return omega0, zeta, regime


# ── AC steady state: phasor impedance and complex power ─────────────
def impedance_resistor(R):
    """Z_R = R (real; dissipates, no phase shift)."""
    return complex(R, 0)


def impedance_inductor(omega, L):
    """Z_L = j omega L (current lags voltage by 90 deg)."""
    return 1j * omega * L


def impedance_capacitor(omega, C):
    """Z_C = 1/(j omega C) = -j/(omega C) (current leads voltage by 90 deg)."""
    if omega == 0 or C == 0:
        raise ValueError("omega and C must be nonzero")
    return 1.0 / (1j * omega * C)


def series_impedance(*Z):
    """Total impedance of elements in series: Z = sum Z_k."""
    return sum(complex(z) for z in Z)


def parallel_impedance(*Z):
    """Total impedance in parallel: 1/Z = sum 1/Z_k."""
    return 1.0 / sum(1.0 / complex(z) for z in Z)


def complex_power(V, I):
    """Complex power S = V . conj(I) for RMS phasors V, I. Returns a dict:

    P = Re S  (active power, W, the part that does work / heats),
    Q = Im S  (reactive power, VAR, sloshes in L/C; +inductive, -capacitive),
    apparent = |S| (VA),  pf = P/|S| (power factor, cos of the V-I angle).
    """
    S = complex(V) * np.conj(complex(I))
    ap = abs(S)
    return {"S": S, "P": S.real, "Q": S.imag, "apparent": ap,
            "pf": S.real / ap if ap > 0 else 1.0}


def resonant_frequency(L, C):
    """Series/parallel LC resonance omega_0 = 1/sqrt(LC) (reactances cancel)."""
    if L <= 0 or C <= 0:
        raise ValueError("L, C must be > 0")
    return 1.0 / np.sqrt(L * C)


# ── Seven-segment display: numbers 0-9 and hex letters A-F ───────────────────
#
# Segment layout:
#   _
#  |_|   segments: a=top, b=top-right, c=bot-right,
#  |_|             d=bottom, e=bot-left, f=top-left, g=middle
#
# Each entry: (a, b, c, d, e, f, g) -- 1=ON, 0=OFF

_SEG7 = {
    '0': (1,1,1,1,1,1,0),
    '1': (0,1,1,0,0,0,0),
    '2': (1,1,0,1,1,0,1),
    '3': (1,1,1,1,0,0,1),
    '4': (0,1,1,0,0,1,1),
    '5': (1,0,1,1,0,1,1),
    '6': (1,0,1,1,1,1,1),
    '7': (1,1,1,0,0,0,0),
    '8': (1,1,1,1,1,1,1),
    '9': (1,1,1,1,0,1,1),
    'A': (1,1,1,0,1,1,1),
    'B': (0,0,1,1,1,1,1),
    'C': (1,0,0,1,1,1,0),
    'D': (0,1,1,1,1,0,1),
    'E': (1,0,0,1,1,1,1),
    'F': (1,0,0,0,1,1,1),
}

SEVEN_SEG_CHARS = list(_SEG7.keys())


def seven_seg_encode(char):
    """Return (a,b,c,d,e,f,g) tuple for a digit or hex letter.

    Inputs: '0'-'9', 'A'-'F' (case-insensitive).
    Each bit: 1=segment ON, 0=OFF.
    Hardware: connect each bit to a transistor driving the LED segment.
    """
    c = str(char).upper()
    if c not in _SEG7:
        raise ValueError(f"'{char}' not in 0-9, A-F")
    return _SEG7[c]


def seven_seg_decode(segments):
    """Reverse lookup: (a,b,c,d,e,f,g) tuple -> character or '?'."""
    t = tuple(int(bool(s)) for s in segments)
    for ch, segs in _SEG7.items():
        if segs == t:
            return ch
    return '?'


def seven_seg_ascii(char):
    """Return 3-line ASCII art of a seven-segment digit/letter.

    Example for '8':
      _
     |_|
     |_|
    """
    a, b, c, d, e, f, g = seven_seg_encode(char)
    top    = f" {'_' if a else ' '} "
    middle = f"{'|' if f else ' '}{'_' if g else ' '}{'|' if b else ' '}"
    bottom = f"{'|' if e else ' '}{'_' if d else ' '}{'|' if c else ' '}"
    return [top, middle, bottom]


def seven_seg_display(text):
    """Render a string of digits/hex letters as ASCII seven-segment art."""
    text = str(text).upper()
    rows = [[], [], []]
    for ch in text:
        if ch == ' ':
            for r in rows:
                r.append('   ')
            continue
        lines = seven_seg_ascii(ch)
        for i, line in enumerate(lines):
            rows[i].append(line)
    return '\n'.join(' '.join(r) for r in rows)


def seven_seg_truth_table():
    """Return list of dicts: char, binary index, segments a-g, hex byte."""
    table = []
    for i, ch in enumerate(SEVEN_SEG_CHARS):
        a, b, c, d, e, f, g = _SEG7[ch]
        byte = (a<<6)|(b<<5)|(c<<4)|(d<<3)|(e<<2)|(f<<1)|g
        table.append({
            'char': ch, 'index': i,
            'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g,
            'hex_byte': f"0x{byte:02X}",
        })
    return table


if __name__ == "__main__":
    sp.init_printing()
    print("RC step solution:", solve_rc_symbolic())
    R, C = 1e3, 1e-9
    print(f"tau = {R*C*1e9:.0f} ns,  f_3dB = {rc_bandwidth(R, C)/1e3:.1f} kHz")
    print("at t=tau, V/Vin =", round(float(rc_step(R*C, R, C)), 4), "(~0.632)")
    print("RLC (R=10,L=1m,C=1u):", rlc_damping(10, 1e-3, 1e-6))
    print("\nSeven-segment 0-F:")
    print(seven_seg_display("0123456789ABCDEF"))
    print("\nTruth table (first 4):")
    for row in seven_seg_truth_table()[:4]:
        print(f"  {row['char']}: a={row['a']} b={row['b']} c={row['c']} "
              f"d={row['d']} e={row['e']} f={row['f']} g={row['g']}  "
              f"-> {row['hex_byte']}")
