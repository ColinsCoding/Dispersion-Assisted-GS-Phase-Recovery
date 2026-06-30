"""Beat frequency, standing waves, and wave superposition.

Two sinusoids at nearby frequencies f1 and f2 add to a carrier at
(f1+f2)/2 amplitude-modulated at the BEAT frequency |f1-f2|.
This is audible as a "wah-wah" at (f1-f2) Hz when two instruments
are slightly out of tune -- and it is the same math as the AM radio
carrier, heterodyne mixing in RF, and the optical beat note in
coherent detection.

Connection to this repo: the dispersive fiber maps frequency -> time
(TS-DFT). The beat frequency between two optical tones separated by
delta_f appears as a sinusoidal wiggle in the time-domain output at
period T = 1/delta_f -- the exact same physics as acoustic beating.
"""
import numpy as np
import sympy as sp


# ── symbolic beat frequency ──────────────────────────────────────────

def beat_frequency_sympy():
    """Derive the beat-frequency formula symbolically.

    cos(2*pi*f1*t) + cos(2*pi*f2*t) =
        2 * cos(2*pi*(f1-f2)/2*t) * cos(2*pi*(f1+f2)/2*t)

    via sum-to-product identity.
    Returns dict of sp.Eq objects.
    """
    t, f1, f2 = sp.symbols('t f1 f2', real=True, positive=True)
    A1 = sp.cos(2 * sp.pi * f1 * t)
    A2 = sp.cos(2 * sp.pi * f2 * t)
    sum_waves = A1 + A2
    # Sum-to-product: cos(a)+cos(b) = 2*cos((a-b)/2)*cos((a+b)/2)
    a = 2 * sp.pi * f1 * t
    b = 2 * sp.pi * f2 * t
    factored = 2 * sp.cos((a - b) / 2) * sp.cos((a + b) / 2)
    f_beat = sp.Symbol('f_beat')
    f_carrier = sp.Symbol('f_carrier')
    return {
        "superposition":
            sp.Eq(sp.Symbol('y'), sum_waves),
        "factored_form":
            sp.Eq(sp.Symbol('y'), factored),
        "beat_freq":
            sp.Eq(f_beat, sp.Abs(f1 - f2)),
        "carrier_freq":
            sp.Eq(f_carrier, (f1 + f2) / 2),
    }


def beat_frequency(f1_hz, f2_hz):
    """Return beat frequency and carrier frequency in Hz."""
    if f1_hz <= 0 or f2_hz <= 0:
        raise ValueError("frequencies must be positive")
    return {
        "f_beat_hz": abs(f1_hz - f2_hz),
        "f_carrier_hz": (f1_hz + f2_hz) / 2.0,
        "period_beat_s": 1.0 / abs(f1_hz - f2_hz) if f1_hz != f2_hz else float('inf'),
    }


def superpose_waves(f1_hz, f2_hz, A1=1.0, A2=1.0, t_max=None, n_pts=4000):
    """Numerically compute y(t) = A1*cos(2*pi*f1*t) + A2*cos(2*pi*f2*t).

    t_max defaults to 3 beat periods (or 10/f_carrier if f1==f2).
    Returns dict with t, y, envelope arrays.
    """
    if f1_hz <= 0 or f2_hz <= 0:
        raise ValueError("frequencies must be positive")
    f_beat = abs(f1_hz - f2_hz)
    f_c = (f1_hz + f2_hz) / 2.0
    if t_max is None:
        t_max = 3.0 / f_beat if f_beat > 0 else 10.0 / f_c
    t = np.linspace(0, t_max, n_pts)
    y = A1 * np.cos(2 * np.pi * f1_hz * t) + A2 * np.cos(2 * np.pi * f2_hz * t)
    envelope = 2 * np.abs(A1 * np.cos(np.pi * (f1_hz - f2_hz) * t))
    return {"t": t, "y": y, "envelope": envelope,
            "f_beat_hz": f_beat, "f_carrier_hz": f_c}


# ── standing waves ───────────────────────────────────────────────────

def standing_wave_modes(L_m, v_ms, n_max=5):
    """Natural resonant frequencies of a 1D standing wave (string fixed at both ends).

    f_n = n * v / (2*L)   n = 1, 2, 3, ...

    Returns list of (n, f_n_hz) tuples up to n_max.
    Works for sound in a pipe (closed-closed), EM in a cavity, guitar string.
    """
    if L_m <= 0 or v_ms <= 0:
        raise ValueError("L_m and v_ms must be positive")
    modes = [(n, n * v_ms / (2.0 * L_m)) for n in range(1, n_max + 1)]
    return modes


def standing_wave_sympy():
    """Standing wave as superposition of equal-amplitude counter-propagating waves:

    y+ = A*sin(kx - wt),  y- = A*sin(kx + wt)
    y+ + y- = 2*A*sin(kx)*cos(wt)   <-- separable: space * time
    """
    x, t_s, A, k, w = sp.symbols('x t A k omega', real=True, positive=True)
    y_plus = A * sp.sin(k * x - w * t_s)
    y_minus = A * sp.sin(k * x + w * t_s)
    # trig identity: sin(a-b)+sin(a+b) = 2*sin(a)*cos(b)
    y_stand = sp.trigsimp(y_plus + y_minus)
    return {
        "traveling_right": sp.Eq(sp.Symbol('y+'), y_plus),
        "traveling_left":  sp.Eq(sp.Symbol('y-'), y_minus),
        "standing_wave":   sp.Eq(sp.Symbol('y_stand'), y_stand),
        "mode_condition":  sp.Eq(k * sp.Symbol('L'), sp.pi * sp.Symbol('n')),
    }


# ── Dirac delta as a resonance / impulse connection ──────────────────

def dirac_delta_fourier_pair_sympy():
    """The Fourier transform of delta(t) = 1 for ALL frequencies.

    This is why an impulse (delta function) excites ALL modes of a resonator:
    it has equal energy at every frequency, so every standing-wave mode rings.
    FT pair:  delta(t) <-> 1,   and   1 <-> delta(f)
    """
    t_s, f_s = sp.symbols('t f', real=True)
    delta_t = sp.DiracDelta(t_s)
    # Symbolic FT: int_{-inf}^{inf} delta(t) * exp(-2*pi*i*f*t) dt = 1
    ft_of_delta = sp.Symbol('FT{delta(t)}')
    ft_of_unity = sp.Symbol('FT{1}')
    return {
        "ft_of_delta":
            sp.Eq(ft_of_delta, sp.Integer(1)),
        "ft_of_unity":
            sp.Eq(ft_of_unity, sp.DiracDelta(f_s)),
        "impulse_excites_all_modes":
            "delta(t) in time -> flat spectrum -> all resonant modes excited equally",
    }


# ── AM radio / optical heterodyne analogy ────────────────────────────

def heterodyne_mixing(f_signal_hz, f_lo_hz):
    """Heterodyne: multiply signal at f_signal by local oscillator at f_lo.

    Product: cos(2*pi*f_s*t) * cos(2*pi*f_lo*t)
           = 0.5 * [cos(2*pi*(f_s + f_lo)*t) + cos(2*pi*(f_s - f_lo)*t)]

    The difference term (f_s - f_lo) is the IF (intermediate frequency).
    A lowpass filter keeps IF, discards the sum. This is exactly how:
      - AM/FM radio receivers demodulate the carrier
      - Optical coherent receivers mix the signal with a local oscillator laser
      - TS-DFT receivers use dispersion instead of a LO (implicit heterodyne)
    """
    f_if = abs(f_signal_hz - f_lo_hz)
    f_sum = f_signal_hz + f_lo_hz
    return {
        "f_IF_hz": f_if,
        "f_sum_hz": f_sum,
        "note": "LPF keeps IF, discards sum -- same math as the beat frequency",
    }


def beat_frequency_sympy_5():
    """Five key SymPy equations."""
    t, f1, f2, L, v, n_sym = sp.symbols('t f1 f2 L v n', real=True, positive=True)
    A, k, w = sp.symbols('A k omega', real=True, positive=True)
    f_s = sp.Symbol('f')
    return {
        "Beat_frequency":
            sp.Eq(sp.Symbol('f_beat'), sp.Abs(f1 - f2)),
        "Sum_to_product":
            sp.Eq(sp.cos(2*sp.pi*f1*t) + sp.cos(2*sp.pi*f2*t),
                  2*sp.cos(sp.pi*(f1-f2)*t)*sp.cos(sp.pi*(f1+f2)*t)),
        "Standing_wave":
            sp.Eq(sp.Symbol('y'), 2*A*sp.sin(k*sp.Symbol('x'))*sp.cos(w*t)),
        "Mode_condition":
            sp.Eq(sp.Symbol('f_n'), n_sym*v/(2*L)),
        "FT_delta":
            sp.Eq(sp.Symbol('FT{delta(t)}'), sp.Integer(1)),
    }


if __name__ == "__main__":
    print("=== Beat frequency: 440 Hz + 443 Hz (A above middle C + 3 Hz sharp) ===")
    r = beat_frequency(440, 443)
    print(f"  f_beat={r['f_beat_hz']} Hz, period={r['period_beat_s']:.3f} s")

    print("\n=== Guitar string standing wave modes (L=0.65 m, v=300 m/s) ===")
    for n, fn in standing_wave_modes(0.65, 300, n_max=4):
        print(f"  n={n}: f={fn:.1f} Hz")

    print("\n=== Heterodyne mixing: 1 GHz signal, 900 MHz LO ===")
    h = heterodyne_mixing(1e9, 0.9e9)
    print(f"  IF = {h['f_IF_hz']/1e6:.0f} MHz, sum = {h['f_sum_hz']/1e9:.2f} GHz")

    print("\n=== SymPy 5 ===")
    for k, eq in beat_frequency_sympy_5().items():
        print(f"  {k}: {eq}")
