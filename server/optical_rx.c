/*
 * optical_rx.c  —  TCP listener + ADC frame receiver + ring buffer
 * Target: AWS Graviton 3 (Neoverse V1, ARMv8.4-A, aarch64-linux)
 * Build:  aarch64-linux-gnu-gcc -O3 -march=armv8.4-a+sve -o optical_rx optical_rx.c gs_solver.c opa_control.c -lm -lfftw3f -lpthread
 *
 * Architecture:
 *   Optical front-end → dual 14-bit ADC @ 56 GSa/s (I1, I2)
 *   TCP port 9000 → recv() → ring buffer → gs_solver → /tmp/gs.sock (JSON)
 *
 * Dispersion parameters:
 *   D1 = -600  ps²     (arm 1, SMF-28 ~100 m + DCF)
 *   D2 = -1200 ps²     (arm 2, SMF-28 ~200 m + DCF)
 *   λ₀ = 1550 nm       H(ν) = exp(i·π·D·λ₀²/c · ν²)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <math.h>
#include <fftw3.h>

/* ── Configuration ────────────────────────────────────────────────── */
#define ADC_SAMPLE_RATE_GSa   56.0          /* GSa/s per arm               */
#define FRAME_SAMPLES         4096          /* samples per ADC frame        */
#define RING_DEPTH            8             /* ring buffer depth (frames)   */
#define TCP_PORT              9000          /* ADC data ingest port         */
#define UNIX_SOCK_PATH        "/tmp/gs.sock"

#define D1_PS2               -600.0         /* dispersion arm 1  [ps²]     */
#define D2_PS2               -1200.0        /* dispersion arm 2  [ps²]     */
#define LAMBDA0_NM            1550.0        /* centre wavelength [nm]      */
#define GS_ITERS              200           /* TD-GS iterations            */

/* ── Ring buffer ──────────────────────────────────────────────────── */
typedef struct {
    float i1[FRAME_SAMPLES];
    float i2[FRAME_SAMPLES];
    int   valid;
} Frame;

static Frame   ring[RING_DEPTH];
static int     ring_head = 0;
static int     ring_tail = 0;
static pthread_mutex_t ring_lock  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  ring_ready = PTHREAD_COND_INITIALIZER;

static volatile int running = 1;

/* ── Signal handler ───────────────────────────────────────────────── */
static void handle_sigint(int sig) { (void)sig; running = 0; }

/* ── ADC receiver thread ──────────────────────────────────────────── */
static void *rx_thread(void *arg)
{
    (void)arg;
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return NULL; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port        = htons(TCP_PORT)
    };
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(srv); return NULL;
    }
    listen(srv, 4);
    fprintf(stderr, "[optical_rx] listening on TCP :%d\n", TCP_PORT);

    while (running) {
        int conn = accept(srv, NULL, NULL);
        if (conn < 0) continue;

        /* Read interleaved I1/I2 float32 frame: 2 × FRAME_SAMPLES × 4 bytes */
        uint8_t buf[2 * FRAME_SAMPLES * sizeof(float)];
        ssize_t n = 0, total = sizeof(buf);
        while (n < (ssize_t)total) {
            ssize_t r = recv(conn, buf + n, total - n, 0);
            if (r <= 0) break;
            n += r;
        }
        close(conn);
        if (n < (ssize_t)total) continue;

        pthread_mutex_lock(&ring_lock);
        int slot = ring_head % RING_DEPTH;
        float *raw = (float*)buf;
        memcpy(ring[slot].i1, raw,                 FRAME_SAMPLES * sizeof(float));
        memcpy(ring[slot].i2, raw + FRAME_SAMPLES, FRAME_SAMPLES * sizeof(float));
        ring[slot].valid = 1;
        ring_head++;
        pthread_cond_signal(&ring_ready);
        pthread_mutex_unlock(&ring_lock);
    }

    close(srv);
    return NULL;
}

/* ── Unix-socket JSON publisher ───────────────────────────────────── */
static void publish_result(const float *phase, int n, double latency_ms)
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return;

    struct sockaddr_un un;
    memset(&un, 0, sizeof(un));
    un.sun_family = AF_UNIX;
    strncpy(un.sun_path, UNIX_SOCK_PATH, sizeof(un.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&un, sizeof(un)) < 0) {
        close(fd); return;
    }

    /* Emit first 64 samples as JSON array (Flask reads /tmp/gs.sock) */
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "{\"latency_ms\":%.3f,\"n_samples\":%d,\"phase\":[", latency_ms, n < 64 ? n : 64);
    write(fd, hdr, hlen);
    for (int i = 0; i < (n < 64 ? n : 64); i++) {
        char val[32];
        int vlen = snprintf(val, sizeof(val), i ? ",%.6f" : "%.6f", (double)phase[i]);
        write(fd, val, vlen);
    }
    const char *tail = "]}\n";
    write(fd, tail, strlen(tail));
    close(fd);
}

/* ── Main ─────────────────────────────────────────────────────────── */
int main(void)
{
    signal(SIGINT,  handle_sigint);
    signal(SIGTERM, handle_sigint);

    fprintf(stderr,
        "[optical_rx] Dispersion-Assisted GS Phase Recovery\n"
        "  D1 = %.0f ps²   D2 = %.0f ps²   λ₀ = %.0f nm\n"
        "  ADC %.0f GSa/s   frame %d samples   GS %d iter\n",
        D1_PS2, D2_PS2, LAMBDA0_NM,
        ADC_SAMPLE_RATE_GSa, FRAME_SAMPLES, GS_ITERS);

    /* Spawn RX thread */
    pthread_t rx;
    if (pthread_create(&rx, NULL, rx_thread, NULL)) {
        perror("pthread_create"); return 1;
    }

    /* GS solver loop (runs in main thread) */
    extern float *gs_solve(const float *i1, const float *i2, int n,
                            double d1_ps2, double d2_ps2,
                            double lambda_nm, int iters);

    while (running) {
        pthread_mutex_lock(&ring_lock);
        while (ring_head == ring_tail && running)
            pthread_cond_wait(&ring_ready, &ring_lock);

        if (!running) { pthread_mutex_unlock(&ring_lock); break; }

        int slot = ring_tail % RING_DEPTH;
        float i1[FRAME_SAMPLES], i2[FRAME_SAMPLES];
        memcpy(i1, ring[slot].i1, sizeof(i1));
        memcpy(i2, ring[slot].i2, sizeof(i2));
        ring_tail++;
        pthread_mutex_unlock(&ring_lock);

        /* Run GS solver */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        float *phi = gs_solve(i1, i2, FRAME_SAMPLES,
                              D1_PS2, D2_PS2, LAMBDA0_NM, GS_ITERS);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3
                  + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

        fprintf(stderr, "[optical_rx] frame processed  latency %.2f ms\n", ms);
        publish_result(phi, FRAME_SAMPLES, ms);
        free(phi);
    }

    pthread_join(rx, NULL);
    fprintf(stderr, "[optical_rx] stopped.\n");
    return 0;
}
