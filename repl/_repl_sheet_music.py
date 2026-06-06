"""
_repl_sheet_music.py
====================
SymPy + music21 + matplotlib sheet music for RogueGuard signal literacy.

S1: Tuning math with SymPy
    - Equal temperament ratios (exact algebraic)
    - Pythagorean tuning vs 12-TET comparison
    - Pythagorean comma derivation

S2: music21 note + chord + stream objects
    - Build a C-major scale
    - Build a ii-V-I progression (Dm7, G7, Cmaj7)
    - Export MusicXML + MIDI
    - Print text render (ASCII staff)

S3: matplotlib sheet music renderer
    - 5-line staff drawn with matplotlib lines
    - Noteheads placed at correct pitch positions
    - Treble clef label, bar lines, time signature
    - A-major chord + scale passage rendered

S4: Fourier <-> sheet music duality
    - Sheet music = time-domain (notes in sequence)
    - FFT spectrum = frequency-domain (all at once)
    - RogueGuard: optical signal = "sheet music" for light

Output: repl/_out_sheet_music.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
import sympy as sp
import os

OUT = os.path.join(os.path.dirname(__file__), "_out_sheet_music.png")

# ============================================================
# S1: TUNING MATH WITH SYMPY
# ============================================================
print("=" * 65)
print("SECTION 1: TUNING MATH (SYMPY)")
print("=" * 65)

# Equal temperament: ratio between adjacent semitones = 2^(1/12)
r = sp.Rational(1, 12)
semitone = sp.Integer(2)**r           # 2^(1/12) -- algebraic, exact
octave   = semitone**12               # = 2 exactly
fifth_ET = semitone**7                # 7 semitones = perfect fifth in 12-TET

# Pythagorean fifth: pure 3/2 ratio
fifth_pyth = sp.Rational(3, 2)

# Pythagorean comma: 12 perfect fifths vs 7 octaves
# 12 fifths = (3/2)^12;  7 octaves = 2^7 = 128
comma_num   = fifth_pyth**12          # (3/2)^12 = 531441/4096
comma_denom = sp.Integer(2)**7        # 128
comma       = comma_num / comma_denom # = 531441/524288

print("\n  EQUAL TEMPERAMENT (12-TET):")
print(f"    Semitone ratio  = 2^(1/12)  = {float(semitone):.8f}")
print(f"    12 semitones    = (2^(1/12))^12 = {sp.simplify(octave)} (exact)")
print(f"    Perfect 5th ET  = 2^(7/12)  = {float(fifth_ET):.8f}")
print(f"    Perfect 5th pure= 3/2       = {float(fifth_pyth):.8f}")
print(f"    5th error (ET vs pure): {(float(fifth_ET) - float(fifth_pyth))*1200:.2f} cents")

print("\n  PYTHAGOREAN COMMA:")
print(f"    12 pure fifths  = (3/2)^12  = {comma_num} = {float(comma_num):.6f}")
print(f"    7 octaves       = 2^7       = {comma_denom}")
print(f"    Comma           = {comma}  = {float(comma):.8f}")
print(f"    Comma in cents  = {1200*float(sp.log(comma, 2)):.4f} cents")
print(f"    (This gap is why 12-TET uses irrational ratios: to spread the comma.)")

# Just intonation: C-major scale ratios
just_ratios = {
    "C": sp.Rational(1, 1),
    "D": sp.Rational(9, 8),
    "E": sp.Rational(5, 4),
    "F": sp.Rational(4, 3),
    "G": sp.Rational(3, 2),
    "A": sp.Rational(5, 3),
    "B": sp.Rational(15, 8),
    "C'": sp.Rational(2, 1),
}

et_semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11, "C'": 12}

print("\n  JUST vs EQUAL TEMPERAMENT (C-major scale):")
print(f"  {'Note':<5} {'Just ratio':<12} {'Just Hz':<10} {'ET Hz':<10} {'Error (cents)'}")
print(f"  {'-'*55}")
C4 = 261.63
for note, ratio in just_ratios.items():
    just_hz = float(ratio) * C4
    et_hz   = C4 * 2**(et_semitones[note]/12)
    cents   = 1200 * np.log2(just_hz / et_hz) if just_hz > 0 else 0
    print(f"  {note:<5} {str(ratio):<12} {just_hz:<10.2f} {et_hz:<10.2f} {cents:+.2f}")

# Harmonic series: SymPy symbols
n, f0 = sp.symbols("n f_0", positive=True)
f_n = n * f0
print(f"\n  HARMONIC SERIES: f_n = {f_n}")
print(f"  Interval between n and n+1: {sp.simplify((n+1)*f0 / (n*f0))} = (n+1)/n")
print(f"  As n -> inf: ratio -> {sp.limit((n+1)/n, n, sp.oo)} (unison, dense spectrum)")

# ============================================================
# S2: MUSIC21 OBJECTS
# ============================================================
print("\n" + "=" * 65)
print("SECTION 2: MUSIC21 NOTE / CHORD / STREAM")
print("=" * 65)

try:
    import music21 as m21
    from music21 import note, chord, stream, tempo, meter, key, clef

    # --- C-major scale ---
    scale_stream = stream.Stream()
    scale_stream.append(clef.TrebleClef())
    scale_stream.append(key.KeySignature(0))          # C major, 0 sharps
    scale_stream.append(meter.TimeSignature("4/4"))
    scale_stream.append(tempo.MetronomeMark(number=120))

    c_major_pitches = ["C4","D4","E4","F4","G4","A4","B4","C5"]
    for p in c_major_pitches:
        n_ = note.Note(p)
        n_.duration.type = "quarter"
        scale_stream.append(n_)

    print("\n  C-major scale:")
    for n_ in scale_stream.getElementsByClass(note.Note):
        print(f"    {n_.nameWithOctave:<5}  MIDI={n_.pitch.midi:<4}  "
              f"freq={n_.pitch.frequency:.2f} Hz  dur={n_.duration.type}")

    # --- ii-V-I progression (Dm7, G7, Cmaj7) in 4/4 ---
    prog_stream = stream.Stream()
    prog_stream.append(clef.TrebleClef())
    prog_stream.append(key.KeySignature(0))
    prog_stream.append(meter.TimeSignature("4/4"))

    chord_data = [
        (["D4","F4","A4","C5"], "Dm7",   "whole"),
        (["G3","B3","D4","F4"], "G7",    "whole"),
        (["C4","E4","G4","B4"], "Cmaj7", "whole"),
    ]
    for pitches, name, dur in chord_data:
        c_ = chord.Chord(pitches)
        c_.duration.type = dur
        prog_stream.append(c_)
        freqs = [p_.frequency for p_ in c_.pitches]
        print(f"\n  {name}: pitches = {[p.nameWithOctave for p in c_.pitches]}")
        print(f"    Frequencies: {[f'{f:.1f}' for f in freqs]} Hz")
        # Interval ratios relative to root
        root_f = freqs[0]
        ratios = [f/root_f for f in freqs]
        print(f"    Ratios: {[f'{r:.4f}' for r in ratios]}")

    # Export MusicXML
    xml_path = os.path.join(os.path.dirname(__file__), "_out_scale.xml")
    scale_stream.write("musicxml", fp=xml_path)
    print(f"\n  MusicXML exported: {xml_path}")

    # Text show (ASCII)
    print("\n  ASCII text of C-major scale:")
    txt = scale_stream.write("text")
    print(f"  (written to {txt})")

    MUSIC21_OK = True

except Exception as e:
    print(f"  music21 error: {e}")
    MUSIC21_OK = False

# ============================================================
# S3: MATPLOTLIB SHEET MUSIC RENDERER
# ============================================================
print("\n" + "=" * 65)
print("SECTION 3: MATPLOTLIB SHEET MUSIC RENDERER")
print("=" * 65)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#FAFAF5")

# ---- Panel layout ----
gs_layout = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                              top=0.93, bottom=0.06, left=0.06, right=0.97)

ax_staff  = fig.add_subplot(gs_layout[0, :])   # full-width staff
ax_just   = fig.add_subplot(gs_layout[1, 0])   # just vs ET cents
ax_fft    = fig.add_subplot(gs_layout[1, 1])   # chord FFT
ax_harm   = fig.add_subplot(gs_layout[2, 0])   # harmonic series
ax_dual   = fig.add_subplot(gs_layout[2, 1])   # Fourier duality

fig.suptitle("SymPy + Python Sheet Music  |  Tuning, Notation, Fourier",
             fontsize=15, fontweight="bold", color="#1a1a2e")

# ----------------------------------------------------------------
# AX_STAFF: draw a 5-line treble staff with noteheads
# ----------------------------------------------------------------
ax = ax_staff
ax.set_facecolor("#FEFEF8")
ax.set_xlim(0, 20)
ax.set_ylim(-3, 12)
ax.axis("off")
ax.set_title("C-Major Scale + A-Major Chord  (Treble Clef)", fontsize=11, pad=4)

# 5 staff lines: y = 0, 1, 2, 3, 4  (each line = one staff step above)
LINE_Y = [0, 1, 2, 3, 4]
for y in LINE_Y:
    ax.plot([0.5, 19.5], [y, y], "k-", lw=1.2, zorder=2)

# Treble clef symbol (ASCII approximation)
ax.text(0.7, -0.5, "𝄞", fontsize=38, va="bottom", color="#333", zorder=5,
        fontfamily="DejaVu Sans")

# Time signature 4/4
ax.text(2.0, 2.3, "4", fontsize=14, ha="center", va="center",
        fontweight="bold", color="#222")
ax.text(2.0, 0.7, "4", fontsize=14, ha="center", va="center",
        fontweight="bold", color="#222")

# Pitch -> staff position mapping (treble clef, C4 = ledger line below staff)
# Staff line 0 = E4, line 1 = G4, line 2 = B4, line 3 = D5, line 4 = F5
# Spaces: below 0 = D4 (-0.5), C4 = -1 (ledger line)
# Step = 0.5 per diatonic pitch
pitch_to_y = {
    "C4": -1.0, "D4": -0.5, "E4":  0.0, "F4":  0.5,
    "G4":  1.0, "A4":  1.5, "B4":  2.0,
    "C5":  2.5, "D5":  3.0, "E5":  3.5, "F5":  4.0,
    "G5":  4.5, "A5":  5.0,
    # sharps (for A-major)
    "C#5": 2.5, "E5":  3.5,
}

# C-major scale noteheads (x positions 3..10)
scale_notes = ["C4","D4","E4","F4","G4","A4","B4","C5"]
scale_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
for i, pname in enumerate(scale_notes):
    x = 3.0 + i * 1.8
    y = pitch_to_y[pname]
    # notehead ellipse
    el = Ellipse((x, y), width=0.55, height=0.38,
                 angle=-15, color=scale_colors[i], zorder=4)
    ax.add_patch(el)
    # stem up
    ax.plot([x + 0.25, x + 0.25], [y, y + 2.8], color="#333", lw=1.2, zorder=3)
    # ledger line for C4
    if pname == "C4":
        ax.plot([x - 0.4, x + 0.4], [-1, -1], "k-", lw=1.2, zorder=3)
    if pname == "C5":
        ax.plot([x - 0.4, x + 0.4], [2.5, 2.5], "k-", lw=1.2, zorder=3)
    ax.text(x, y - 0.9, pname, ha="center", va="top",
            fontsize=7.5, color="#555", zorder=5)

# Bar line after scale
ax.plot([17.0, 17.0], [0, 4], "k-", lw=1.5, zorder=3)

# A-major chord (A4, C#5, E5) stacked
chord_x = 18.0
a_chord = [("A4", 1.5, "#1a6eb5"), ("C#5", 2.5, "#e05c1a"), ("E5", 3.5, "#2ca02c")]
for pname, y, color in a_chord:
    el = Ellipse((chord_x, y), width=0.55, height=0.38,
                 angle=-15, color=color, zorder=4)
    ax.add_patch(el)
    # sharp sign for C#5
    if "C#" in pname:
        ax.text(chord_x - 0.5, y, "#", fontsize=9, va="center",
                color=color, zorder=5, fontweight="bold")
# Ledger line C5 space
ax.plot([chord_x - 0.4, chord_x + 0.4], [2.5, 2.5], "k-", lw=1.2, zorder=3)
# Common stem
ax.plot([chord_x + 0.25, chord_x + 0.25], [1.5, 6.0], color="#333", lw=1.2, zorder=3)
ax.text(chord_x, 0.2, "Amaj", ha="center", va="top",
        fontsize=8, color="#444", fontweight="bold")

# Double bar at end
ax.plot([19.3, 19.3], [0, 4], "k-", lw=1.2, zorder=3)
ax.plot([19.5, 19.5], [0, 4], "k-", lw=3.0, zorder=3)

# ----------------------------------------------------------------
# AX_JUST: Just intonation vs ET error in cents
# ----------------------------------------------------------------
ax = ax_just
ax.set_facecolor("#F8F8FF")

notes_list = list(just_ratios.keys())
cents_err  = []
for note_name in notes_list:
    just_hz = float(just_ratios[note_name]) * C4
    et_hz   = C4 * 2**(et_semitones[note_name]/12)
    cents_err.append(1200 * np.log2(just_hz / et_hz))

colors_j = ["#d62728" if c < 0 else "#2ca02c" for c in cents_err]
bars = ax.bar(notes_list, cents_err, color=colors_j, edgecolor="k",
              linewidth=0.6, zorder=3)
ax.axhline(0, color="k", lw=0.8, zorder=4)
ax.set_xlabel("Note", fontsize=9)
ax.set_ylabel("Just - ET  (cents)", fontsize=9)
ax.set_title("Just Intonation vs 12-TET Error", fontsize=10, pad=4)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, cents_err):
    if abs(val) > 0.5:
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.3 if val >= 0 else -0.5),
                f"{val:+.1f}", ha="center", fontsize=7)

# ----------------------------------------------------------------
# AX_FFT: FFT of A-major chord
# ----------------------------------------------------------------
ax = ax_fft
ax.set_facecolor("#F8F8FF")

fs = 44100
T  = 2.0
t  = np.linspace(0, T, int(fs * T), endpoint=False)
f_A4   = 440.0
f_Cs5  = 440.0 * 2**(4/12)   # C#5 = 4 semitones above A4
f_E5   = 440.0 * 2**(7/12)   # E5  = 7 semitones above A4

signal = (np.sin(2*np.pi*f_A4*t) +
          0.8*np.sin(2*np.pi*f_Cs5*t) +
          0.9*np.sin(2*np.pi*f_E5*t))
# Add harmonics for richness
signal += (0.4*np.sin(2*np.pi*2*f_A4*t) +
           0.3*np.sin(2*np.pi*3*f_A4*t) +
           0.2*np.sin(2*np.pi*2*f_Cs5*t))

N   = len(signal)
fft = np.fft.rfft(signal)
freqs = np.fft.rfftfreq(N, 1/fs)
amp   = np.abs(fft) / N

# Plot only up to 2500 Hz
mask = freqs <= 2500
ax.fill_between(freqs[mask], amp[mask], color="#4e91d4", alpha=0.6)
ax.plot(freqs[mask], amp[mask], color="#1a5c9e", lw=0.8)

# Label peaks
for f_label, name, col in [
    (f_A4, "A4\n440Hz",   "#d62728"),
    (f_Cs5,"C#5\n554Hz",  "#e07b1a"),
    (f_E5, "E5\n659Hz",   "#2ca02c"),
    (2*f_A4,"A5\n880Hz",  "#9467bd"),
]:
    idx = np.argmin(np.abs(freqs - f_label))
    if idx < len(amp[mask]):
        ax.axvline(f_label, color=col, lw=1.0, ls="--", alpha=0.7)
        ax.text(f_label, amp[idx]*1.05, name, ha="center",
                fontsize=7, color=col, fontweight="bold")

ax.set_xlabel("Frequency (Hz)", fontsize=9)
ax.set_ylabel("Amplitude", fontsize=9)
ax.set_title("A-Major Chord FFT Spectrum (44.1 kHz, harmonics)", fontsize=10, pad=4)
ax.set_xlim(0, 2500)
ax.grid(alpha=0.3)

# ----------------------------------------------------------------
# AX_HARM: Harmonic series amplitude vs harmonic number
# ----------------------------------------------------------------
ax = ax_harm
ax.set_facecolor("#F8F8FF")

n_harm = 12
harmonics = np.arange(1, n_harm+1)
# Sawtooth wave: amplitude = 1/n
saw_amp   = 1.0 / harmonics
# Square wave: only odd harmonics, amplitude = 1/n
sq_amp    = np.where(harmonics % 2 == 1, 1.0/harmonics, 0.0)
# Triangle: only odd, amplitude = 1/n^2
tri_amp   = np.where(harmonics % 2 == 1, 1.0/harmonics**2, 0.0)

w = 0.25
ax.bar(harmonics - w,   saw_amp, width=w, label="Sawtooth (1/n)",     color="#4e91d4", alpha=0.8)
ax.bar(harmonics,       sq_amp,  width=w, label="Square (1/n odd)",   color="#e05c1a", alpha=0.8)
ax.bar(harmonics + w,   tri_amp, width=w, label="Triangle (1/n^2)",   color="#2ca02c", alpha=0.8)

ax.set_xlabel("Harmonic number n", fontsize=9)
ax.set_ylabel("Relative amplitude", fontsize=9)
ax.set_title("Harmonic Series: Sawtooth / Square / Triangle", fontsize=10, pad=4)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_xticks(harmonics)
ax.grid(axis="y", alpha=0.3)

# Fourier series sum annotation
ax.text(0.02, 0.97,
        r"$f(t) = \sum_{n=1}^{\infty} a_n \sin(2\pi n f_0 t)$",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(fc="#ffffee", ec="#aaa", pad=3))

# ----------------------------------------------------------------
# AX_DUAL: Fourier duality -- time domain <-> frequency domain
# ----------------------------------------------------------------
ax = ax_dual
ax.set_facecolor("#F0F4FF")
ax.axis("off")
ax.set_title("Sheet Music <-> FFT: Fourier Duality", fontsize=10, pad=4)

lines = [
    ("DOMAIN",           "TIME  (sheet music)",       "FREQUENCY (spectrum)", "#222"),
    ("",                 "",                           "",                    "#aaa"),
    ("Representation",   "Notes in sequence",          "Peaks in spectrum",   "#333"),
    ("Axis",             "t (seconds, bars)",          "f (Hz, semitones)",   "#333"),
    ("Duration",         "Note length (1/4, 1/8...)",  "Spectral resolution", "#333"),
    ("Pitch",            "Note position on staff",     "Peak frequency",      "#333"),
    ("Volume",           "Dynamic marking (ff, pp)",   "Peak amplitude",      "#333"),
    ("Timbre",           "Instrument (violin, piano)", "Harmonic envelope",   "#333"),
    ("Chord",            "Simultaneous noteheads",     "Multiple peaks",      "#333"),
    ("",                 "",                           "",                    "#aaa"),
    ("ROGUEGUARD",       "Optical signal I(t)",        "H(nu)=exp(i*pi*D*nu^2)","#1a4e8c"),
    ("GS phase",         "Measured intensity",         "Recovered phase phi(nu)","#1a4e8c"),
    ("Sagnac",           "I(t) = cos^2(phi/2)",        "phi from FFT peak",   "#1a4e8c"),
]

col_x = [0.01, 0.28, 0.64]
row_y  = np.linspace(0.97, 0.03, len(lines))

for j, (label, left, right, color) in enumerate(lines):
    y = row_y[j]
    if label == "DOMAIN":
        ax.text(col_x[0], y, label,    fontsize=8,  color="#888",  va="top", style="italic")
        ax.text(col_x[1], y, left,     fontsize=8.5,color="#1a3a8c",va="top",fontweight="bold")
        ax.text(col_x[2], y, right,    fontsize=8.5,color="#8c1a1a",va="top",fontweight="bold")
        ax.axhline(y - 0.02, color="#999", lw=0.6, xmin=0.01, xmax=0.99)
    elif label == "":
        ax.axhline(y, color="#ddd", lw=0.5, xmin=0.01, xmax=0.99)
    elif label == "ROGUEGUARD":
        ax.axhline(y + 0.01, color="#1a4e8c", lw=0.8, xmin=0.01, xmax=0.99)
        ax.text(col_x[0], y, label, fontsize=8, color="#1a4e8c", va="top", fontweight="bold")
        ax.text(col_x[1], y, left,  fontsize=8, color="#1a4e8c", va="top")
        ax.text(col_x[2], y, right, fontsize=8, color="#1a4e8c", va="top")
    else:
        ax.text(col_x[0], y, label, fontsize=8,  color="#555", va="top", style="italic")
        ax.text(col_x[1], y, left,  fontsize=8,  color=color,  va="top")
        ax.text(col_x[2], y, right, fontsize=8,  color=color,  va="top")

# FT arrow
ax.annotate("", xy=(0.62, 0.82), xytext=(0.27, 0.82),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
ax.text(0.445, 0.84, "FT", ha="center", va="bottom",
        fontsize=9, color="#555", transform=ax.transAxes, fontweight="bold")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\n  Plots saved to {OUT}")

# ============================================================
# S4: FOURIER <-> SHEET MUSIC DUALITY (SUMMARY)
# ============================================================
print("\n" + "=" * 65)
print("SECTION 4: FOURIER <-> SHEET MUSIC DUALITY")
print("=" * 65)

print("""
  SHEET MUSIC = TIME DOMAIN
  FFT SPECTRUM = FREQUENCY DOMAIN

  Domain       Time (sheet music)        Frequency (spectrum)
  ---------    -------------------------  ----------------------
  Axis         t (bars, beats)            f (Hz)
  Pitch        Note position on staff     Peak frequency
  Duration     Quarter / eighth note      Spectral resolution
  Volume       pp / mf / ff               Peak amplitude
  Timbre       Instrument type            Harmonic envelope
  Chord        Simultaneous noteheads     Multiple FFT peaks

  FOURIER TRANSFORM PAIR:
    x(t) = SUM_k  A_k * sin(2*pi*f_k*t + phi_k)    [time: notes]
    X(f) = INT x(t) e^{-i 2 pi f t} dt              [freq: spectrum]

  Composer writes sheet music -> encodes X(f) as notes on staff.
  FFT reads recorded audio   -> recovers X(f) from x(t).
  SAME information, dual representations.

  ROGUEGUARD CONNECTION:
    Optical signal  -> analogous to audio waveform
    Dispersive fiber H(nu) = exp(i*pi*D*nu^2) -> frequency-to-time mapper
    GS phase retrieval -> reads the "sheet music" of the phase phi(nu)
    Sagnac: I(t) = cos^2(delta_phi/2) -> phase encoded in intensity
    Same FFT duality: time-stretch maps frequency to time (Jalali/STEAM)
""")

print("=" * 65)
print("Done.")
print("=" * 65)
