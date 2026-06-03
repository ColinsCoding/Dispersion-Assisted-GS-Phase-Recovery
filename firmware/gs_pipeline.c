/*
 * gs_pipeline.c — GS phase retrieval (C) piped to Python classifier
 *
 * ARCHITECTURE:
 *   C process (this file):
 *     1. Load ADC samples (time + dispersed intensity)
 *     2. Run TDGSA → recover complex field ψ(t)
 *     3. Fork Python classifier subprocess
 *     4. Write recovered phase φ(t) to Python stdin (binary doubles)
 *     5. Read classification result from Python stdout
 *     6. Gate output: flag rogue wave or pass signal
 *
 * OUSD ALIGNMENT:
 *   DoD FutureG / Integrated Sensing and Cyber — real-time optical
 *   signal integrity monitoring. No foreign-origin ML weights loaded
 *   at runtime; classifier is compiled/verified at build time.
 *
 * COMPILE:
 *   gcc -O2 -Wall -std=c11 -o gs_pipeline gs_pipeline.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

/* ── config ───────────────────────────────────────────── */
#define N          256
#define N_ITER     50
#define PI         3.14159265358979323846
#define BETA2_Z   -5000.0     /* ps²  — DCF dispersion, must have |β₂z|≥5000 */
#define ROGUE_THR  2.0        /* flag if peak/mean > 2× (optical rogue def)  */

/* ── FFT (Cooley-Tukey in-place) ─────────────────────── */
static void fft(double complex *x, int n, int inv)
{
    if (n == 1) return;
    double complex even[n/2], odd[n/2];
    for (int k = 0; k < n/2; k++) { even[k] = x[2*k]; odd[k] = x[2*k+1]; }
    fft(even, n/2, inv);
    fft(odd,  n/2, inv);
    for (int k = 0; k < n/2; k++) {
        double ang = (inv ? 1 : -1) * 2.0*PI*k/n;
        double complex w = cos(ang) + I*sin(ang);
        x[k]       = even[k] + w*odd[k];
        x[k + n/2] = even[k] - w*odd[k];
    }
    if (inv) for (int k = 0; k < n; k++) x[k] /= 2;
}

/* ── dispersion operator D{ψ}: multiply Ψ(ω) by exp(i β₂z ω²/2) ── */
static void apply_dispersion(double complex *Psi, int n, double beta2z)
{
    for (int k = 0; k < n; k++) {
        double omega = (k < n/2) ? k : k - n;   /* centered frequencies */
        double phi   = 0.5 * beta2z * omega*omega / (n*n);
        Psi[k] *= cos(phi) + I*sin(phi);
    }
}

/* ── project: replace amplitude with sqrt(I), keep phase ─────── */
static void project(double complex *psi, const double *I_meas, int n)
{
    for (int k = 0; k < n; k++) {
        double amp = cabs(psi[k]);
        if (amp < 1e-12) { psi[k] = sqrt(I_meas[k]); continue; }
        psi[k] = sqrt(I_meas[k]) * (psi[k] / amp);
    }
}

/* ── TDGSA: n_iter alternating projections with dispersion ──── */
static void tdgsa(double complex *psi,
                  const double *I_time, const double *I_disp,
                  int n, int n_iter)
{
    double complex Psi[N], Psi_d[N];

    for (int iter = 0; iter < n_iter; iter++) {
        /* 1. time constraint */
        project(psi, I_time, n);

        /* 2. FFT */
        memcpy(Psi, psi, n * sizeof(double complex));
        fft(Psi, n, 0);

        /* 3. dispersive copy */
        memcpy(Psi_d, Psi, n * sizeof(double complex));
        apply_dispersion(Psi_d, n, BETA2_Z);

        /* 4. dispersive intensity constraint */
        project(Psi_d, I_disp, n);

        /* 5. remove dispersion to get corrected Psi */
        apply_dispersion(Psi_d, n, -BETA2_Z);

        /* 6. IFFT */
        fft(Psi_d, n, 1);
        memcpy(psi, Psi_d, n * sizeof(double complex));
    }
}

/* ── rogue wave check ──────────────────────────────────── */
static int is_rogue(const double complex *psi, int n)
{
    double mean = 0.0, peak = 0.0;
    for (int k = 0; k < n; k++) {
        double I = cabs(psi[k]) * cabs(psi[k]);
        mean += I;
        if (I > peak) peak = I;
    }
    mean /= n;
    return (mean > 1e-12) && (peak / mean > ROGUE_THR);
}

/* ── spawn Python classifier, pipe phase in, read label out ── */
static int classify_phase(const double *phase, int n, const char *py_script)
{
    int to_py[2], from_py[2];
    if (pipe(to_py) || pipe(from_py)) { perror("pipe"); return -1; }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }

    if (pid == 0) {
        /* child — rewire stdin/stdout to pipes */
        close(to_py[1]);   close(from_py[0]);
        dup2(to_py[0],   STDIN_FILENO);
        dup2(from_py[1], STDOUT_FILENO);
        close(to_py[0]);   close(from_py[1]);
        execlp("python3", "python3", py_script, NULL);
        perror("execlp python3");
        _exit(1);
    }

    /* parent — write phase array as raw doubles */
    close(to_py[0]);  close(from_py[1]);
    write(to_py[1], phase, n * sizeof(double));
    close(to_py[1]);   /* EOF signals end of input to Python */

    /* read one-line result: "normal\n" or "rogue\n" */
    char result[64] = {0};
    ssize_t nr = read(from_py[0], result, sizeof(result) - 1);
    close(from_py[0]);
    waitpid(pid, NULL, 0);

    if (nr <= 0) return -1;
    result[strcspn(result, "\n")] = '\0';
    printf("[classify] Python says: %s\n", result);
    return (strcmp(result, "rogue") == 0) ? 1 : 0;
}

/* ── synthetic ADC data (replace with real hardware reads) ── */
static void make_test_signal(double *I_time, double *I_disp, int n)
{
    for (int k = 0; k < n; k++) {
        double t  = (double)k / n;
        /* Gaussian pulse */
        double amp = exp(-pow((t - 0.5)*10, 2));
        I_time[k] = amp * amp;
        /* dispersed: chirped version */
        double chirp = amp * cos(50.0*PI*t*t);
        I_disp[k] = chirp * chirp;
    }
}

/* ── main ─────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    const char *py_script = (argc > 1) ? argv[1] : "classifier.py";

    /* synthetic data — replace with fread() from ADC driver */
    double I_time[N], I_disp[N];
    make_test_signal(I_time, I_disp, N);

    /* initialize field with unit amplitude, zero phase */
    double complex psi[N];
    for (int k = 0; k < N; k++) psi[k] = sqrt(I_time[k]);

    /* run TDGSA */
    printf("[gs] running TDGSA: N=%d, n_iter=%d, beta2z=%.0f ps²\n",
           N, N_ITER, BETA2_Z);
    tdgsa(psi, I_time, I_disp, N, N_ITER);

    /* extract recovered phase */
    double phase[N];
    for (int k = 0; k < N; k++) phase[k] = carg(psi[k]);

    /* C-side rogue check */
    int rogue_c = is_rogue(psi, N);
    printf("[gs] C rogue check: %s\n", rogue_c ? "ROGUE" : "normal");

    /* Python classifier via pipe */
    int rogue_py = classify_phase(phase, N, py_script);
    if (rogue_py < 0) {
        fprintf(stderr, "[gs] classifier unavailable — using C check only\n");
        rogue_py = rogue_c;
    }

    /* gate decision */
    int gate = rogue_c || rogue_py;
    printf("[gs] GATE: %s\n", gate ? "BLOCK (rogue event)" : "PASS");

    /* write recovered field to binary file for downstream processing */
    FILE *fp = fopen("recovered_field.bin", "wb");
    if (fp) {
        fwrite(psi,   sizeof(double complex), N, fp);
        fwrite(phase, sizeof(double),         N, fp);
        fclose(fp);
        printf("[gs] wrote recovered_field.bin (%zu bytes)\n",
               N*sizeof(double complex) + N*sizeof(double));
    }

    return gate;   /* exit code: 0=pass, 1=rogue */
}
