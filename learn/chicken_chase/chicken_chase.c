/* chicken_chase.c -- a CS:GO-style chicken chase in plain C99.
 *
 * Teaching goals (CSC 60 / digital logic / binary algebra):
 *   - a 1-bit black/white framebuffer: 8 pixels per byte, set with shifts and masks
 *       byte  = i >> 3        (i / 8)
 *       mask  = 0x80 >> (i & 7)   (i % 8, MSB = leftmost pixel)
 *     The buffer is written verbatim as a binary PBM (P4) image -- the file IS memory.
 *   - xorshift32 random numbers: three shifts and three XORs
 *   - a physics check: "pure pursuit" (Bouguer 1732). A chaser at speed vp always
 *     heads at a chicken running in a straight line at speed vc, starting a
 *     distance d away perpendicular to its path. Capture time T = d vp / (vp^2 - vc^2)
 *     for vc < vp, never for vc >= vp.
 *
 * Build:  gcc -std=c99 -O2 -Wall -Wextra -o chicken_chase chicken_chase.c -lm
 * Usage:  chicken_chase pursuit <vc_over_vp> <d>          -> prints capture time or -1
 *         chicken_chase sim <seed> <steps> <every> <outdir> -> writes outdir/frame_####.pbm,
 *                                                            prints "step caught" lines
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI                     /* M_PI is POSIX, not ISO C99: -std=c99 hides it */
#define M_PI 3.14159265358979323846
#endif

#define W 128
#define H 96
#define FB_BYTES (W * H / 8)
#define N_CHICKENS 12

/* ── 1-bit framebuffer ─────────────────────────────────────────────── */

__attribute__((noinline)) void px_set(uint8_t *fb, int x, int y)
{
    if ((unsigned)x >= W || (unsigned)y >= H)
        return;
    unsigned i = (unsigned)(y * W + x);
    fb[i >> 3] |= (uint8_t)(0x80u >> (i & 7u));
}

int px_get(const uint8_t *fb, int x, int y)
{
    unsigned i = (unsigned)(y * W + x);
    return (fb[i >> 3] >> (7u - (i & 7u))) & 1u;
}

static void draw_disc(uint8_t *fb, double cx, double cy, int r)
{
    for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
            if (dx * dx + dy * dy <= r * r)
                px_set(fb, (int)lround(cx) + dx, (int)lround(cy) + dy);
}

static void draw_ring(uint8_t *fb, double cx, double cy, int r)
{
    for (int k = 0; k < 64; k++) {
        double a = 2 * M_PI * k / 64;
        px_set(fb, (int)lround(cx + r * cos(a)), (int)lround(cy + r * sin(a)));
    }
}

static int write_pbm(const char *path, const uint8_t *fb)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fprintf(f, "P4\n%d %d\n", W, H);           /* rows are W/8 bytes, MSB first */
    fwrite(fb, 1, FB_BYTES, f);
    fclose(f);
    return 0;
}

/* ── xorshift32 RNG ────────────────────────────────────────────────── */

static uint32_t rng_state = 2463534242u;

static uint32_t xorshift32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}

static double urand(void) { return (xorshift32() >> 8) * (1.0 / 16777216.0); }

/* ── pure pursuit check ────────────────────────────────────────────── */

static double pursuit_capture_time(double k, double d)
{
    /* chaser at (0,0), speed 1; chicken starts at (d,0), runs along +y at speed k */
    double px = 0, py = 0, cx = d, cy = 0, t = 0;
    const double dt = 1e-5 * d, t_max = 200.0 * d, eps = 1e-4 * d;
    while (t < t_max) {
        double dx = cx - px, dy = cy - py, r = sqrt(dx * dx + dy * dy);
        if (r < eps)
            return t;
        px += dt * dx / r;
        py += dt * dy / r;
        cy += dt * k;
        t += dt;
    }
    return -1.0;
}

/* ── the chase ─────────────────────────────────────────────────────── */

typedef struct { double x, y, heading; int alive; } Chicken;

static int simulate(uint32_t seed, int steps, int every, const char *outdir)
{
    rng_state = seed ? seed : 1u;
    Chicken c[N_CHICKENS];
    for (int i = 0; i < N_CHICKENS; i++)
        c[i] = (Chicken){ 8 + urand() * (W - 16), 8 + urand() * (H - 16),
                          2 * M_PI * urand(), 1 };
    double px = W / 2.0, py = H / 2.0;
    const double v_player = 0.9, v_wander = 0.25, v_flee = 0.75;
    const double flee_radius = 18.0, catch_radius = 2.0;
    int caught = 0;
    uint8_t fb[FB_BYTES];
    char path[512];

    for (int s = 0; s <= steps; s++) {
        /* player: head for the nearest live chicken (pure pursuit) */
        int best = -1; double bd = 1e9;
        for (int i = 0; i < N_CHICKENS; i++) {
            if (!c[i].alive) continue;
            double d = hypot(c[i].x - px, c[i].y - py);
            if (d < bd) { bd = d; best = i; }
        }
        if (best < 0) break;
        px += v_player * (c[best].x - px) / bd;
        py += v_player * (c[best].y - py) / bd;

        for (int i = 0; i < N_CHICKENS; i++) {
            if (!c[i].alive) continue;
            double dx = c[i].x - px, dy = c[i].y - py, d = hypot(dx, dy);
            if (d < catch_radius) { c[i].alive = 0; caught++; printf("%d %d\n", s, caught); continue; }
            if (d < flee_radius) {                 /* panic: run straight away */
                c[i].heading = atan2(dy, dx);
                c[i].x += v_flee * cos(c[i].heading);
                c[i].y += v_flee * sin(c[i].heading);
            } else {                               /* wander: jittered heading */
                c[i].heading += 0.6 * (urand() - 0.5);
                c[i].x += v_wander * cos(c[i].heading);
                c[i].y += v_wander * sin(c[i].heading);
            }
            /* walls: bounce */
            if (c[i].x < 2 || c[i].x > W - 3) { c[i].heading = M_PI - c[i].heading; c[i].x = fmin(fmax(c[i].x, 2), W - 3); }
            if (c[i].y < 2 || c[i].y > H - 3) { c[i].heading = -c[i].heading;       c[i].y = fmin(fmax(c[i].y, 2), H - 3); }
        }

        if (outdir && every > 0 && s % every == 0) {
            memset(fb, 0, sizeof fb);
            for (int x = 0; x < W; x++) { px_set(fb, x, 0); px_set(fb, x, H - 1); }
            for (int y = 0; y < H; y++) { px_set(fb, 0, y); px_set(fb, W - 1, y); }
            for (int i = 0; i < N_CHICKENS; i++)
                if (c[i].alive) draw_disc(fb, c[i].x, c[i].y, 1);
            draw_disc(fb, px, py, 2);
            draw_ring(fb, px, py, (int)flee_radius);
            snprintf(path, sizeof path, "%s/frame_%04d.pbm", outdir, s / every);
            if (write_pbm(path, fb) != 0) { fprintf(stderr, "cannot write %s\n", path); return 1; }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && strcmp(argv[1], "pursuit") == 0) {
        double k = atof(argv[2]), d = atof(argv[3]);
        if (k < 0 || d <= 0) { fprintf(stderr, "need k >= 0, d > 0\n"); return 2; }
        printf("%.9f\n", pursuit_capture_time(k, d));
        return 0;
    }
    if (argc >= 6 && strcmp(argv[1], "sim") == 0)
        return simulate((uint32_t)strtoul(argv[2], NULL, 10), atoi(argv[3]),
                        atoi(argv[4]), argv[5]);
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) {
        uint8_t fb[FB_BYTES] = {0};
        px_set(fb, 0, 0); px_set(fb, 9, 0); px_set(fb, W - 1, H - 1);
        printf("%02x %02x %02x %d %d\n", fb[0], fb[1], fb[FB_BYTES - 1],
               px_get(fb, 9, 0), px_get(fb, 8, 0));
        return 0;
    }
    fprintf(stderr, "usage: %s pursuit <k> <d> | sim <seed> <steps> <every> <outdir> | selftest\n", argv[0]);
    return 2;
}
