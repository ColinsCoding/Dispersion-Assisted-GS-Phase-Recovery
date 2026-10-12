/* ldv_demod.cu -- heterodyne laser Doppler vibrometer demodulation on the GPU,
 * for C channels at once (a scanning vibrometer measures many surface points).
 *
 * Per channel c (v_peak = 0.5 (c+1)/C m/s, 5 kHz vibration, HeNe 632.8 nm):
 *   signal   s[n] = cos(2 pi f_B t + 4 pi x(t)/lambda) + noise,  x = v_peak (1 - cos wt)/w
 *   mix      z[n] = s[n] exp(-i 2 pi f_B t)                            (to baseband)
 *   CIC      three cascaded length-5 boxcars = one 13-tap causal FIR    (nulls 80 MHz)
 *   phase    unwrap(atan2(Im z, Re z))
 *   velocity v[n] = lambda/(4 pi) (phase[n+1] - phase[n-1]) fs / 2, delayed by 6 samples
 *
 * Same formulas as dgs/laser_doppler_vibrometer.py (vibration_signal + cic_demodulate),
 * so the noise-free result matches Python to roundoff.
 *
 * Kernels: one thread per sample for synth+mix+FIR+atan2 (embarrassingly parallel);
 * one thread per CHANNEL for unwrap + derivative + error (a running sum along time,
 * inherently sequential within a channel, parallel across channels).
 *
 * Build (x64 Native Tools prompt, or after vcvars64.bat):
 *     nvcc -O2 -o ldv_demod.exe ldv_demod.cu
 * Run:
 *     ldv_demod.exe C N snr_db      (snr_db < 0 means noise-free)
 * Writes ldv_rms.csv (channel, v_peak, rms_error) and ldv_last_channel.csv
 * (every 100th sample of the last channel: n, v_true_delayed, v_rec).
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

constexpr double PI = 3.14159265358979323846;
constexpr double FS = 200e6, FB = 40e6, FV = 5e3, LAM = 632.8e-9;
constexpr int TAPS = 13, DELAY = 6;
__constant__ double d_w[TAPS];

__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ double gauss(uint64_t key) {               /* counter-based Box-Muller */
    double u1 = ((splitmix64(key) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double u2 = ((splitmix64(key ^ 0xD1B54A32D192ED03ULL) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

__device__ double signal(int c, long n, int C, double sigma) {
    double t = n / FS, w = 2 * PI * FV, vp = 0.5 * (c + 1) / C;
    double x = vp * (1 - cos(w * t)) / w;
    double s = cos(2 * PI * FB * t + 4 * PI * x / LAM);
    if (sigma > 0) s += sigma * gauss(((uint64_t)c << 40) ^ (uint64_t)n);
    return s;
}

/* one thread per (channel, sample): synthesize the 13 samples it needs, mix, FIR, angle */
__global__ void demod_phase(double *phase, int C, long N, double sigma) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)C * N) return;
    int c = (int)(idx / N);
    long n = idx % N;
    double re = 0, im = 0;
    for (int k = 0; k < TAPS && k <= n; k++) {
        long m = n - k;
        double s = signal(c, m, C, sigma), arg = 2 * PI * FB * (m / FS);
        re += d_w[k] * s * cos(arg);
        im -= d_w[k] * s * sin(arg);
    }
    phase[idx] = atan2(im, re);
}

/* one thread per channel: unwrap, differentiate, compare with the delayed true velocity */
__global__ void unwrap_velocity(double *phase, double *rms, int C, long N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    double *p = phase + (long)c * N;
    double off = 0, prev = p[0];
    for (long n = 1; n < N; n++) {                 /* in-place unwrap */
        double raw = p[n], d = raw - prev;
        if (d > PI) off -= 2 * PI; else if (d < -PI) off += 2 * PI;
        prev = raw;
        p[n] = raw + off;
    }
    double vp = 0.5 * (c + 1) / C, w = 2 * PI * FV, se = 0;
    long cnt = 0;
    for (long n = TAPS; n < N - 1; n++) {
        double v = LAM / (4 * PI) * (p[n + 1] - p[n - 1]) * FS / 2;
        double vt = vp * sin(w * (n - DELAY) / FS);
        se += (v - vt) * (v - vt); cnt++;
        p[n - TAPS] = v;                            /* reuse buffer: velocity, shifted by TAPS */
    }
    rms[c] = sqrt(se / cnt);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ldv_demod C N snr_db\n"); return 1; }
    int C = atoi(argv[1]);
    long N = atol(argv[2]);
    double snr_db = atof(argv[3]);
    if (C < 1 || N < 100) { fprintf(stderr, "need C >= 1, N >= 100\n"); return 1; }
    double sigma = snr_db < 0 ? 0.0 : sqrt(0.5) * pow(10.0, -snr_db / 20.0);

    double w[TAPS] = {0};                           /* (boxcar/5) convolved with itself 3 times */
    double b[5] = {0.2, 0.2, 0.2, 0.2, 0.2}, tmp[TAPS];
    for (int i = 0; i < 5; i++) w[i] = b[i];
    for (int pass = 1; pass < 3; pass++) {
        for (int i = 0; i < TAPS; i++) tmp[i] = 0;
        for (int i = 0; i < TAPS; i++) for (int j = 0; j < 5; j++) if (i + j < TAPS) tmp[i + j] += w[i] * b[j];
        for (int i = 0; i < TAPS; i++) w[i] = tmp[i];
    }
    CHECK(cudaMemcpyToSymbol(d_w, w, sizeof(w)));

    double *phase, *rms;
    CHECK(cudaMalloc(&phase, sizeof(double) * C * N));
    CHECK(cudaMalloc(&rms, sizeof(double) * C));
    cudaEvent_t t0, t1, t2;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1)); CHECK(cudaEventCreate(&t2));
    CHECK(cudaEventRecord(t0));
    long total = (long)C * N;
    demod_phase<<<(unsigned)((total + 255) / 256), 256>>>(phase, C, N, sigma);
    CHECK(cudaEventRecord(t1));
    unwrap_velocity<<<(C + 63) / 64, 64>>>(phase, rms, C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(t2)); CHECK(cudaEventSynchronize(t2));
    float ms1, ms2;
    CHECK(cudaEventElapsedTime(&ms1, t0, t1)); CHECK(cudaEventElapsedTime(&ms2, t1, t2));
    printf("C=%d N=%ld snr_db=%g  demod_ms=%.2f unwrap_ms=%.2f  samples_per_s=%.3e\n",
           C, N, snr_db, ms1, ms2, (double)total / ((ms1 + ms2) / 1e3));

    std::vector<double> h_rms(C), h_last(N);
    CHECK(cudaMemcpy(h_rms.data(), rms, sizeof(double) * C, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_last.data(), phase + (long)(C - 1) * N, sizeof(double) * N, cudaMemcpyDeviceToHost));
    FILE *f = fopen("ldv_rms.csv", "w");
    if (!f) { perror("ldv_rms.csv"); return 1; }
    fprintf(f, "channel,v_peak,rms_error\n");
    for (int c = 0; c < C; c++) fprintf(f, "%d,%.17g,%.17g\n", c, 0.5 * (c + 1) / C, h_rms[c]);
    fclose(f);
    f = fopen("ldv_last_channel.csv", "w");
    if (!f) { perror("ldv_last_channel.csv"); return 1; }
    fprintf(f, "n,v_rec\n");
    for (long n = TAPS; n < N - 1; n += 100) fprintf(f, "%ld,%.17g\n", n, h_last[n - TAPS]);
    fclose(f);
    printf("channel %d: v_peak=%.4f m/s  rms_error=%.4e m/s\n", C - 1, 0.5, h_rms[C - 1]);
    cudaFree(phase); cudaFree(rms);
    return 0;
}
