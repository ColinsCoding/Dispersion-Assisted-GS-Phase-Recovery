"""The trivial ambiguities of phase retrieval: what you can NEVER recover from a magnitude.

Phase retrieval (the whole point of this repo's dispersion-GS pipeline) reconstructs a complex
signal x from only its Fourier MAGNITUDE |X(k)| -- the phase is thrown away by the measurement. But
several distinct signals share the exact same magnitude, so no algorithm can tell them apart. These
are the TRIVIAL AMBIGUITIES, and there are three:

  1. GLOBAL PHASE:      x -> e^{i phi} x           (the same everywhere)   -> |X| unchanged.
  2. CONJUGATE / TWIN:   x[n] -> conj(x[-n])                               -> X -> conj(X), |X| unchanged.
  3. TRANSLATION:        x[n] -> x[n - n0]          (a circular shift)     -> X -> e^{-i k n0} X, |X| unchanged.

The first is exactly the statement that "the phase of A carries no physical significance": just as a
quantum wavefunction psi and e^{i phi} psi describe the SAME physics (only |psi|^2 is observable --
the overall phase of a square-well amplitude A is free), a phase-retrieval solution is only ever
defined UP TO these three transformations. `dgs.gs_core` says as much when it returns phi "up to a
global phase offset."

So a reconstruction can't be compared to ground truth by naive subtraction -- you must first factor
the ambiguities out. This module builds each ambiguity, proves it leaves |X| invariant, and provides
the tools to compare solutions modulo global phase: the phase-aligned residual, the phase-invariant
distance min_phi ||a - e^{i phi} b|| = sqrt(||a||^2 + ||b||^2 - 2|<a,b>|), and a canonical
representative so ambiguity-equivalent signals map to one thing. Only the RELATIVE phases carry
information; the global one is gauge. NumPy; py-3.13.
"""

import numpy as np


def fourier_magnitude(x):
    """The measured quantity in phase retrieval: |FFT(x)|. Everything below leaves it invariant."""
    return np.abs(np.fft.fft(np.asarray(x, complex)))


def apply_global_phase(x, phi):
    """Ambiguity 1 -- multiply by a single global phase e^{i phi} (physically invisible)."""
    return np.asarray(x, complex) * np.exp(1j * phi)


def conjugate_reflection(x):
    """Ambiguity 2 -- the twin image y[n] = conj(x[(-n) mod N]); its FFT is conj(X), same magnitude."""
    x = np.asarray(x, complex)
    n = len(x)
    return np.conj(x[(-np.arange(n)) % n])


def translate(x, shift):
    """Ambiguity 3 -- a circular shift x[n] -> x[n-shift]; the FFT only picks up a linear phase."""
    return np.roll(np.asarray(x, complex), shift)


def global_phase_align(reference, x):
    """Rotate x by the single global phase that best matches `reference` (minimizes ||reference -
    e^{i phi} x||). If x = e^{i psi} * reference, this returns reference exactly."""
    ref, x = np.asarray(reference, complex), np.asarray(x, complex)
    phi = np.angle(np.vdot(x, ref))          # <x, ref> = sum conj(x) ref
    return x * np.exp(1j * phi)


def phase_invariant_distance(a, b):
    """Distance between a and b after removing the global-phase freedom:
    min_phi ||a - e^{i phi} b|| = sqrt(||a||^2 + ||b||^2 - 2|<a,b>|). Zero iff a and b differ only
    by a global phase."""
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    aa = float(np.vdot(a, a).real)
    bb = float(np.vdot(b, b).real)
    cross = abs(np.vdot(a, b))
    return float(np.sqrt(max(aa + bb - 2 * cross, 0.0)))


def canonicalize_phase(x):
    """A canonical global-phase representative: rotate so the largest-magnitude sample is real and
    positive. Any two signals related by a global phase map to the SAME canonical signal."""
    x = np.asarray(x, complex)
    i = int(np.argmax(np.abs(x)))
    return x * np.exp(-1j * np.angle(x[i]))


def leaves_magnitude_invariant(x, y, tol=1e-9):
    """True if x and y have the same Fourier magnitude (are phase-retrieval-indistinguishable)."""
    return bool(np.allclose(fourier_magnitude(x), fourier_magnitude(y), atol=tol))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16) + 1j * rng.standard_normal(16)

    print("=== the three trivial ambiguities all preserve |FFT(x)| ===")
    print(f"  global phase (phi=1.2):   |FFT| unchanged? {leaves_magnitude_invariant(x, apply_global_phase(x, 1.2))}")
    print(f"  conjugate twin:           |FFT| unchanged? {leaves_magnitude_invariant(x, conjugate_reflection(x))}")
    print(f"  translation (shift 5):    |FFT| unchanged? {leaves_magnitude_invariant(x, translate(x, 5))}")
    combo = translate(apply_global_phase(conjugate_reflection(x), 0.7), 3)
    print(f"  all three composed:       |FFT| unchanged? {leaves_magnitude_invariant(x, combo)}")

    print("\n=== comparing reconstructions modulo global phase ===")
    y = apply_global_phase(x, 2.5)              # 'reconstruction' differing only by global phase
    print(f"  naive ||x - y||          = {np.linalg.norm(x - y):.4f}  (nonzero, but they're the same physics)")
    print(f"  phase-invariant distance = {phase_invariant_distance(x, y):.2e}  (~0: identical up to phase)")
    print(f"  after global_phase_align = {np.linalg.norm(x - global_phase_align(x, y)):.2e}")
    print(f"  same canonical form?       {np.allclose(canonicalize_phase(x), canonicalize_phase(y))}")

    print("\n=== but a RELATIVE phase change is real (only the global phase is free) ===")
    z = x.copy(); z[3] *= np.exp(1j * 1.0)      # rotate ONE sample
    print(f"  rotate one sample -> |FFT| unchanged? {leaves_magnitude_invariant(x, z)}  "
          f"(False: relative phase carries the information)")
