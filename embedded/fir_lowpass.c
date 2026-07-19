/* fir_lowpass.c -- a streaming FIR low-pass filter in C (embedded-DSP style).
 *
 * This is the shape of code that runs on a microcontroller / DSP in the
 * receiver front-end: an anti-alias / smoothing low-pass that cleans the photo-
 * detector signal one sample at a time before the ADC. It shows the core
 * embedded-C patterns -- a struct holding state, a circular buffer, and a
 * process-one-sample function called in real time as data streams in.
 *
 * The taps are a Hamming-windowed sinc (the textbook windowed-sinc low-pass).
 *
 * Build & run:  gcc -O2 -Wall -o fir fir_lowpass.c -lm  &&  ./fir
 */
#include <stdio.h>
#include <math.h>

#define NTAPS 41                 /* filter length (odd -> linear phase) */

/* ----- filter design: Hamming-windowed sinc low-pass, cutoff fc (cycles/sample) ----- */
static void design_lowpass(float *h, int n, float fc)
{
    int M = n - 1;
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        float m = k - M / 2.0f;                       /* center the sinc */
        float sinc = (m == 0.0f) ? 2.0f * fc
                                 : sinf(2.0f * (float)M_PI * fc * m) / ((float)M_PI * m);
        float win = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * k / M);  /* Hamming */
        h[k] = sinc * win;
        sum += h[k];
    }
    for (int k = 0; k < n; k++) h[k] /= sum;          /* normalize to unit DC gain */
}

/* ----- the streaming filter: state + one-sample update ----- */
typedef struct {
    float buf[NTAPS];            /* circular delay line */
    const float *h;             /* tap coefficients */
    int pos;                    /* write index */
} fir_t;

static void fir_init(fir_t *f, const float *h)
{
    f->h = h;
    f->pos = 0;
    for (int i = 0; i < NTAPS; i++) f->buf[i] = 0.0f;
}

/* push one input sample, return one filtered output sample */
static float fir_push(fir_t *f, float x)
{
    f->buf[f->pos] = x;
    float y = 0.0f;
    int idx = f->pos;
    for (int k = 0; k < NTAPS; k++) {                 /* y = sum h[k] * past[k] */
        y += f->h[k] * f->buf[idx];
        idx = (idx == 0) ? NTAPS - 1 : idx - 1;
    }
    f->pos = (f->pos + 1) % NTAPS;
    return y;
}

/* steady-state gain (output/input RMS) for a pure tone of frequency f0 */
static double tone_gain_db(fir_t *f, float f0, int n)
{
    fir_init(f, f->h);                                /* reset state */
    int warm = NTAPS * 4;
    double out2 = 0.0;
    for (int i = 0; i < n + warm; i++) {
        float y = fir_push(f, sinf(2.0f * (float)M_PI * f0 * i));
        if (i >= warm) out2 += (double)y * y;
    }
    double out_rms = sqrt(out2 / n);
    double in_rms = sqrt(0.5);                        /* RMS of a unit-amplitude sine */
    return 20.0 * log10(out_rms / in_rms);
}

int main(void)
{
    float h[NTAPS];
    design_lowpass(h, NTAPS, 0.1f);                   /* cutoff at 0.1 cycles/sample */
    fir_t f;
    fir_init(&f, h);

    double g_low  = tone_gain_db(&f, 0.02f, 4000);    /* passband tone */
    double g_high = tone_gain_db(&f, 0.40f, 4000);    /* stopband tone (noise) */

    printf("FIR low-pass, %d taps, cutoff 0.1 cyc/sample\n", NTAPS);
    printf("  passband tone (f=0.02): gain = %+6.2f dB  (want ~0)\n", g_low);
    printf("  stopband tone (f=0.40): gain = %+6.2f dB  (want strongly negative)\n", g_high);

    int pass = (fabs(g_low) < 1.0) && (g_high < -40.0);
    printf("%s  (passband kept, stopband attenuated by %.0f dB)\n",
           pass ? "PASS" : "FAIL", -g_high);
    return pass ? 0 : 1;
}
