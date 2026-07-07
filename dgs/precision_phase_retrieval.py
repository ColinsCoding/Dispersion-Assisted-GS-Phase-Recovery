"""64-bit vs 32-bit phase retrieval: where the float format sets the floor.

The Gerchberg-Saxton loop in dgs.gs_core is a chain of FFTs, quadratic-phase
multiplies, and amplitude projections. None of that arithmetic is exact -- every
operation rounds to the nearest representable float, and those roundings pile up.
The FORMAT decides how small they are:

    float32 (IEEE 754 single): 24-bit mantissa, eps = 2^-23 ~ 1.19e-7 (~7 digits)
    float64 (IEEE 754 double): 53-bit mantissa, eps = 2^-52 ~ 2.22e-16 (~16 digits)

Same formats C calls `float` and `double` (see dgs.c_type_precision) -- this is the
"compiler math" (how a number is stored in bits) deciding what the algorithm
running in the interpreter can actually achieve. The consequence is an ERROR FLOOR:
run GS on perfectly noiseless measurements and the residual does NOT go to zero, it
bottoms out near the machine epsilon of whatever dtype the field is stored in.
float32 stalls around 1e-6-1e-7; float64 pushes ~9 orders of magnitude lower. So
"32-bit phase retrieval" isn't wrong -- it's just floor-limited about a million
times higher, which matters the moment your measurement SNR is better than ~140 dB
or you iterate for a long time. The mantissa bits map to dynamic range at the same
6.02 dB/bit as an ADC (dgs.adc_snr_bits): float32 ~ 144 dB, float64 ~ 319 dB.

This module reuses dgs.gs_core for the physics and the QPSK test signal, and
re-implements disperse/undisperse/iterate with an explicit dtype cast at each step
so the field's state really lives in the chosen precision. NumPy only.
"""

import numpy as np
from dgs import gs_core


def ieee754_summary():
    """The two formats' hard numbers straight from numpy.finfo (not quoted):
    mantissa bits, machine epsilon, decimal digits, and the ADC-style dynamic
    range 6.02*(mantissa+1) dB. float32/float64 ARE C's float/double."""
    out = {}
    for name, dt in (("float32", np.float32), ("float64", np.float64)):
        fi = np.finfo(dt)
        mant = fi.nmant + 1                      # + implicit leading 1
        out[name] = {
            "mantissa_bits": mant,
            "eps": float(fi.eps),
            "decimal_digits": mant * np.log10(2),
            "dynamic_range_dB": 6.02 * mant,     # same "6 dB per bit" as an ADC
        }
    return out


# ── precision-parametrized dispersion + GS (state stored in `cdtype`) ────────

def _cast(dtype):
    """Real float dtype paired with its complex partner for a given precision."""
    if dtype in (np.float32, np.complex64):
        return np.float32, np.complex64
    if dtype in (np.float64, np.complex128):
        return np.float64, np.complex128
    raise ValueError("dtype must be float32/complex64 or float64/complex128")


def disperse_p(E, D, dtype=np.float64):
    """dgs.gs_core.disperse with every array pinned to `dtype`'s precision:
    the FFT, the quadratic phase H = exp(i*pi*D*nu^2), and the result are all
    stored in complex64 or complex128, so single-precision rounding actually
    enters the pipeline between steps."""
    rdt, cdt = _cast(dtype)
    E = np.asarray(E, cdt)
    N = len(E)
    nu = np.fft.fftfreq(N).astype(rdt)
    H = np.exp(1j * np.pi * D * nu ** 2).astype(cdt)
    return (np.fft.ifft(np.fft.fft(E) * H)).astype(cdt)


def undisperse_p(E_d, D, dtype=np.float64):
    """Inverse of disperse_p at the same precision (conjugate phase)."""
    rdt, cdt = _cast(dtype)
    E_d = np.asarray(E_d, cdt)
    N = len(E_d)
    nu = np.fft.fftfreq(N).astype(rdt)
    H = np.exp(-1j * np.pi * D * nu ** 2).astype(cdt)
    return (np.fft.ifft(np.fft.fft(E_d) * H)).astype(cdt)


def fft_roundtrip_floor(N=1024, dtype=np.float64, rng_seed=0):
    """The simplest precision probe: ifft(fft(x)) should return x, but rounds
    to ~eps. Returns the RMS round-trip error -- ~1e-7 for float32, ~1e-16 for
    float64. This is the noise the GS loop can never get below."""
    rdt, cdt = _cast(dtype)
    rng = np.random.default_rng(rng_seed)
    x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(cdt)
    y = np.fft.ifft(np.fft.fft(x)).astype(cdt)
    return float(np.sqrt(np.mean(np.abs(y - x) ** 2)))


def _phase_rms_error(phi_ret, phi_true):
    """RMS phase error after removing the unobservable global phase offset
    (GS recovers phase only up to a constant). Wrapped into (-pi, pi]."""
    offset = np.angle(np.mean(np.exp(1j * (phi_true - phi_ret))))
    d = np.angle(np.exp(1j * (phi_ret + offset - phi_true)))
    return float(np.sqrt(np.mean(d ** 2)))


def retrieve_phase_p(I1, I2, D1, D2, n_iter=200, dtype=np.float64,
                     phi_true=None):
    """GS phase retrieval with the field state held in `dtype` precision the
    whole way. Mirrors dgs.gs_core.retrieve_phase but through disperse_p /
    undisperse_p, so the achievable floor is set by the format. Returns
    (phi, amp_errors, phase_error) where amp_errors is the RMS intensity
    residual per iteration and phase_error is the final RMS phase error vs
    phi_true (or None if phi_true not given)."""
    rdt, cdt = _cast(dtype)
    I1 = np.asarray(I1, rdt)
    I2 = np.asarray(I2, rdt)
    E = undisperse_p(np.sqrt(np.maximum(I1, 0)).astype(cdt), D1, dtype)
    errors = []
    for _ in range(n_iter):
        # project onto D1, then D2, enforcing unit amplitude (constant envelope)
        Ed = disperse_p(E, D1, dtype)
        Ed = np.sqrt(np.maximum(I1, 0)).astype(cdt) * np.exp(1j * np.angle(Ed))
        E = np.exp(1j * np.angle(undisperse_p(Ed, D1, dtype))).astype(cdt)
        Ed = disperse_p(E, D2, dtype)
        Ed = np.sqrt(np.maximum(I2, 0)).astype(cdt) * np.exp(1j * np.angle(Ed))
        E = np.exp(1j * np.angle(undisperse_p(Ed, D2, dtype))).astype(cdt)
        resid = np.abs(disperse_p(E, D2, dtype)) ** 2 - I2
        errors.append(float(np.sqrt(np.mean(resid ** 2))))
    phase_err = None if phi_true is None else _phase_rms_error(np.angle(E), phi_true)
    return np.angle(E), errors, phase_err


def compare_precisions(n_iter=200, rng_seed=0):
    """Run identical NOISELESS GS retrieval in float32 and float64 and report
    the error floor each bottoms out at. The physics is the same; only the
    storage format differs, and that alone sets a floor ~a million times apart.
    Returns dict keyed by 'float32'/'float64' with final amp error, phase
    error, and the per-iteration error curve, plus 'floor_ratio'."""
    data = gs_core.make_qpsk_measurements(n_symbols=64, sps=8,
                                          D1=-5000.0, D2=-5750.0,
                                          snr_db=np.inf, rng_seed=rng_seed)
    out = {}
    for name, dt in (("float32", np.float32), ("float64", np.float64)):
        phi, errs, perr = retrieve_phase_p(data["I1"], data["I2"],
                                           data["D1"], data["D2"],
                                           n_iter=n_iter, dtype=dt,
                                           phi_true=data["phi_true"])
        out[name] = {"amp_error": errs[-1], "phase_error": perr, "errors": errs}
    out["floor_ratio"] = out["float32"]["amp_error"] / out["float64"]["amp_error"]
    return out


if __name__ == "__main__":
    s = ieee754_summary()
    for name, d in s.items():
        if name == "floor_ratio":
            continue
        print(f"{name}: {d['mantissa_bits']:2d}-bit mantissa, eps={d['eps']:.2e}, "
              f"~{d['decimal_digits']:.1f} digits, {d['dynamic_range_dB']:.0f} dB range")

    print("\nFFT round-trip floor (ifft(fft(x)) - x):")
    print(f"  float32: {fft_roundtrip_floor(dtype=np.float32):.2e}")
    print(f"  float64: {fft_roundtrip_floor(dtype=np.float64):.2e}")

    print("\nGS phase retrieval on NOISELESS data (same physics, different format):")
    cmp = compare_precisions(n_iter=200)
    for name in ("float32", "float64"):
        d = cmp[name]
        print(f"  {name}: amp-error floor {d['amp_error']:.2e}, "
              f"phase RMS {d['phase_error']:.2e} rad")
    print(f"  float32 floor is {cmp['floor_ratio']:.0f}x higher than float64 "
          f"-- purely the mantissa.")
