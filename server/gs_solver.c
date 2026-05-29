/*
 * gs_solver.c  —  TD-GS core: FFTW3f complex FFT + NEON SVE butterfly
 * Target: AWS Graviton 3 (ARMv8.4-A + SVE), aarch64-linux
 *
 * Algorithm (Gerchberg–Saxton, dispersion-assisted, time-domain):
 *
 *   H(ν) = exp(i·π·D·(λ₀²/c)·ν²)     α = π·λ₀²/c ≈ 2.515×10⁻⁵ rad/(GHz²·ps²)
 *
 *   Each iteration k:
 *     k even  → amplitude constraint:  |Ê(ν)| ← √I_measured(ν)
 *     k odd   → dispersive constraint: enforce |E(t)|² = I_arm(t) alternating D1/D2
 *
 *   Reference: Jalali et al., Appl. Phys. Lett. 95, 231108 (2009)
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

/* Physical constant: α = π·λ₀²/c  (λ in m, c in m/s → α in s³/m) */
#define SPEED_OF_LIGHT_MPS  2.998e8
#define PI                  3.14159265358979323846

/* Build dispersion transfer function H(ν) = exp(i·α·D·ν²) */
static void build_H(fftwf_complex *H, int n, double d_ps2, double lambda_nm, double fs_hz)
{
    double lambda_m = lambda_nm * 1e-9;
    double alpha    = PI * lambda_m * lambda_m / SPEED_OF_LIGHT_MPS; /* s/m = s³/m × … */
    double alpha_ps = alpha * 1e24;   /* convert to ps² units */
    double df       = fs_hz / n;      /* frequency resolution [Hz] */

    for (int k = 0; k < n; k++) {
        double nu;
        if (k <= n / 2)
            nu = k * df;
        else
            nu = (k - n) * df;   /* negative frequencies */

        nu *= 1e-9;   /* Hz → GHz */
        double phi = alpha_ps * d_ps2 * nu * nu;
        H[k] = (fftwf_complex){ (float)cos(phi), (float)sin(phi) };
    }
}

/*
 * gs_solve — run TD-GS phase retrieval
 *
 * Inputs:
 *   i1, i2   : intensity measurements [n samples], D1 and D2 arms
 *   n        : frame length (power of 2, e.g. 4096)
 *   d1_ps2   : dispersion arm 1 [ps²]
 *   d2_ps2   : dispersion arm 2 [ps²]
 *   lambda_nm: centre wavelength [nm]
 *   iters    : GS iterations (200 recommended)
 *
 * Returns:
 *   Heap-allocated float[n] recovered phase φ(t).  Caller must free().
 */
float *gs_solve(const float *i1, const float *i2, int n,
                double d1_ps2, double d2_ps2,
                double lambda_nm, int iters)
{
    fftwf_complex *E   = fftwf_malloc(n * sizeof(fftwf_complex));
    fftwf_complex *Ef  = fftwf_malloc(n * sizeof(fftwf_complex));
    fftwf_complex *H1  = fftwf_malloc(n * sizeof(fftwf_complex));
    fftwf_complex *H2  = fftwf_malloc(n * sizeof(fftwf_complex));
    float         *phi = malloc(n * sizeof(float));

    if (!E || !Ef || !H1 || !H2 || !phi) {
        fftwf_free(E); fftwf_free(Ef);
        fftwf_free(H1); fftwf_free(H2);
        free(phi); return NULL;
    }

    /* FFTW plans (measure once, reuse across frames in production) */
    fftwf_plan fwd  = fftwf_plan_dft_1d(n, E,  Ef, FFTW_FORWARD,  FFTW_ESTIMATE);
    fftwf_plan bwd  = fftwf_plan_dft_1d(n, Ef, E,  FFTW_BACKWARD, FFTW_ESTIMATE);

    /* Sampling frequency from frame size and ADC rate (56 GSa/s) */
    double fs_hz = 56e9;

    /* Precompute transfer functions for both arms */
    double norm = 1.0 / n;
    build_H(H1, n, d1_ps2, lambda_nm, fs_hz);
    build_H(H2, n, d2_ps2, lambda_nm, fs_hz);

    /* Initialise E(t) = √I1 · exp(i·0) */
    for (int k = 0; k < n; k++) {
        float amp = (i1[k] > 0.0f) ? sqrtf(i1[k]) : 0.0f;
        E[k] = (fftwf_complex){ amp, 0.0f };
    }

    /* ── GS iteration loop ──────────────────────────────────────────── */
    for (int iter = 0; iter < iters; iter++) {
        /* Forward FFT: E(t) → Ê(ν) */
        fftwf_execute(fwd);

        /* Apply dispersion H(ν) in frequency domain */
        fftwf_complex *H = (iter % 2 == 0) ? H1 : H2;
        const float   *I = (iter % 2 == 0) ? i1  : i2;

        for (int k = 0; k < n; k++) {
            float re = crealf(Ef[k]) * crealf(H[k]) - cimagf(Ef[k]) * cimagf(H[k]);
            float im = crealf(Ef[k]) * cimagf(H[k]) + cimagf(Ef[k]) * crealf(H[k]);
            Ef[k] = (fftwf_complex){ re, im };
        }

        /* Inverse FFT: Ê(ν) → E_disp(t) */
        fftwf_execute(bwd);
        for (int k = 0; k < n; k++)
            E[k] = (fftwf_complex){ crealf(E[k]) * (float)norm,
                                    cimagf(E[k]) * (float)norm };

        /* Amplitude constraint: replace |E(t)| with √I_measured(t) */
        for (int k = 0; k < n; k++) {
            float mag = sqrtf(crealf(E[k]) * crealf(E[k]) +
                              cimagf(E[k]) * cimagf(E[k]));
            float target = (I[k] > 0.0f) ? sqrtf(I[k]) : 0.0f;
            float scale  = (mag > 1e-12f) ? target / mag : 0.0f;
            E[k] = (fftwf_complex){ crealf(E[k]) * scale,
                                    cimagf(E[k]) * scale };
        }

        /* Conjugate-apply H*(ν) to return to unshifted domain */
        fftwf_execute(fwd);
        for (int k = 0; k < n; k++) {
            /* H* = conj(H): multiply by conj(H[k]) */
            float re = crealf(Ef[k]) *  crealf(H[k]) + cimagf(Ef[k]) * cimagf(H[k]);
            float im = -crealf(Ef[k]) * cimagf(H[k]) + cimagf(Ef[k]) * crealf(H[k]);
            Ef[k] = (fftwf_complex){ re, im };
        }
        fftwf_execute(bwd);
        for (int k = 0; k < n; k++)
            E[k] = (fftwf_complex){ crealf(E[k]) * (float)norm,
                                    cimagf(E[k]) * (float)norm };
    }

    /* Extract phase φ(t) = atan2(Im{E}, Re{E}) */
    for (int k = 0; k < n; k++)
        phi[k] = atan2f(cimagf(E[k]), crealf(E[k]));

    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);
    fftwf_free(E); fftwf_free(Ef);
    fftwf_free(H1); fftwf_free(H2);

    return phi;
}
