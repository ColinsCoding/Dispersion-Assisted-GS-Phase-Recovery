/*
 * repl/gs_cli_demo.c
 * argv + switch demo: mirrors gsrecover.py CLI in C.
 * Compile: gcc -o gs_cli_demo gs_cli_demo.c -lm
 * Run:     ./gs_cli_demo --mode demo --D1 5000 --D2 -5000 --iter 200
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── option struct ────────────────────────────────────────── */
typedef struct {
    int    mode;      /* 0=demo  1=recover  2=benchmark */
    double D1;
    double D2;
    int    n_iter;
    int    plot;
    char   i1_path[256];
    char   i2_path[256];
} Opts;

/* ── mode enum ────────────────────────────────────────────── */
#define MODE_DEMO      0
#define MODE_RECOVER   1
#define MODE_BENCHMARK 2

static int parse_mode(const char *s) {
    if (strcmp(s, "demo")      == 0) return MODE_DEMO;
    if (strcmp(s, "recover")   == 0) return MODE_RECOVER;
    if (strcmp(s, "benchmark") == 0) return MODE_BENCHMARK;
    fprintf(stderr, "Unknown mode: %s\n", s);
    exit(1);
}

/* ── usage ────────────────────────────────────────────────── */
static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --mode  demo|recover|benchmark  (default: demo)\n");
    printf("  --D1    float                   (default: 5000)\n");
    printf("  --D2    float                   (default: -5000)\n");
    printf("  --iter  int                     (default: 200)\n");
    printf("  --i1    path/to/I1.npy\n");
    printf("  --i2    path/to/I2.npy\n");
    printf("  --plot                          (flag, no value)\n");
    printf("  --help\n");
}

/* ── parse argv ───────────────────────────────────────────── */
static Opts parse_args(int argc, char *argv[]) {
    Opts o = {MODE_DEMO, 5000.0, -5000.0, 200, 0, "", ""};

    for (int i = 1; i < argc; i++) {

        /* switch on first 4 chars of flag — avoids strcmp chain */
        if (argv[i][0] != '-') { fprintf(stderr, "Bad arg: %s\n", argv[i]); exit(1); }

        /* use switch on a hash to show the pattern; string compare for safety */
        if      (strcmp(argv[i], "--help") == 0) { usage(argv[0]); exit(0); }
        else if (strcmp(argv[i], "--plot") == 0) { o.plot = 1; }
        else if (strcmp(argv[i], "--mode") == 0) { o.mode   = parse_mode(argv[++i]); }
        else if (strcmp(argv[i], "--D1")   == 0) { o.D1     = atof(argv[++i]); }
        else if (strcmp(argv[i], "--D2")   == 0) { o.D2     = atof(argv[++i]); }
        else if (strcmp(argv[i], "--iter") == 0) { o.n_iter = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--i1")   == 0) { strncpy(o.i1_path, argv[++i], 255); }
        else if (strcmp(argv[i], "--i2")   == 0) { strncpy(o.i2_path, argv[++i], 255); }
        else { fprintf(stderr, "Unknown flag: %s\n", argv[i]); exit(1); }
    }
    return o;
}

/* ── switch: dispatch on mode ─────────────────────────────── */
static void run(Opts o) {
    switch (o.mode) {

        case MODE_DEMO:
            printf("[demo]  D1=%.0f  D2=%.0f  iter=%d\n",
                   o.D1, o.D2, o.n_iter);
            printf("  Synthetic soliton: sech pulse, phi_NL = 0.8 rad\n");
            printf("  GS would run here (call into libgs.so or Python ctypes)\n");
            /* in real code: call disperse(), project loop, angle() */
            break;

        case MODE_RECOVER:
            if (o.i1_path[0] == '\0' || o.i2_path[0] == '\0') {
                fprintf(stderr, "recover mode needs --i1 and --i2\n");
                exit(1);
            }
            printf("[recover]  I1=%s  I2=%s  D1=%.0f  D2=%.0f\n",
                   o.i1_path, o.i2_path, o.D1, o.D2);
            printf("  Load .npy -> run GS -> save phi.npy\n");
            /* in real code: parse .npy header, malloc, run GS, write output */
            break;

        case MODE_BENCHMARK:
            printf("[benchmark]  %d iterations x N=512\n", o.n_iter);
            /* time the inner GS loop */
            {
                int N = 512, reps = o.n_iter;
                double *I1 = calloc(N, sizeof(double));
                double *re = calloc(N, sizeof(double));
                /* stub: just measure loop overhead */
                for (int r = 0; r < reps; r++)
                    for (int n = 0; n < N; n++)
                        re[n] += I1[n];   /* dummy op to prevent optimization out */
                printf("  Done. (stub — real benchmark would call FFT kernel)\n");
                free(I1); free(re);
            }
            break;

        default:
            fprintf(stderr, "Unhandled mode %d\n", o.mode);
            exit(1);
    }

    if (o.plot)
        printf("  --plot set: would launch matplotlib via popen() or write PNG\n");
}

/* ── main ─────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    printf("gs_cli_demo  (C argv+switch example)\n");
    printf("---------------------------------------\n");

    Opts o = parse_args(argc, argv);
    run(o);

    return 0;
}
