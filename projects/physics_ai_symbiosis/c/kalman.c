/*
 * kalman.c -- physics model + noisy sensor = Kalman filter (ANSI C89)
 *
 * The introduction of "Physics-AI Symbiosis" (Jalali, Kadambi, Roychowdhury, 2021)
 * starts with the Kalman filter: neither the physics model nor the sensor alone
 * is good enough, but blending them is. Here a cart moves at roughly constant
 * velocity with small random accelerations.
 *   physics only : dead reckoning from a slightly wrong initial velocity
 *   sensor only  : position measurements with 2 m noise
 *   Kalman       : constant-velocity model + measurements, optimally weighted
 *
 * Build: gcc -std=c89 -pedantic -Wall -Wextra -O2 kalman.c -o kalman.exe -lm
 * Run:   kalman.exe          (writes kalman.csv)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static unsigned long seed = 2021UL;

static double urand(void)
{
    seed = (1103515245UL * seed + 12345UL) & 0x7fffffffUL;
    return ((double)seed + 0.5) / 2147483648.0;
}

static double gauss(void)
{
    double u1 = urand(), u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

int main(void)
{
    const int n = 200;
    const double dt = 0.1, sig_meas = 2.0, sig_acc = 0.3;
    double x = 0.0, v = 5.0;                 /* truth */
    double xp = 0.0, vp = 4.0;               /* physics only: wrong initial velocity */
    double xk = 0.0, vk = 0.0;               /* Kalman state */
    double P00 = 100.0, P01 = 0.0, P10 = 0.0, P11 = 100.0;
    double q00, q01, q11, R, a, z, S, K0, K1, innov, n00, n01, n10, n11;
    double se_meas = 0.0, se_phys = 0.0, se_kal = 0.0;
    int i;
    FILE *fp = fopen("kalman.csv", "w");

    if (!fp) { perror("kalman.csv"); return 1; }
    q00 = sig_acc * sig_acc * dt * dt * dt * dt / 4.0;
    q01 = sig_acc * sig_acc * dt * dt * dt / 2.0;
    q11 = sig_acc * sig_acc * dt * dt;
    R = sig_meas * sig_meas;
    fprintf(fp, "t,truth,measurement,physics_only,kalman,kalman_velocity\n");

    for (i = 1; i <= n; i++) {
        /* the real world */
        a = sig_acc * gauss();
        x += v * dt + 0.5 * a * dt * dt;
        v += a * dt;
        z = x + sig_meas * gauss();

        /* physics only */
        xp += vp * dt;

        /* Kalman predict: x <- F x, P <- F P F^T + Q with F = [[1, dt], [0, 1]] */
        xk += vk * dt;
        n00 = P00 + dt * (P10 + P01) + dt * dt * P11 + q00;
        n01 = P01 + dt * P11 + q01;
        n10 = P10 + dt * P11 + q01;
        n11 = P11 + q11;
        P00 = n00; P01 = n01; P10 = n10; P11 = n11;

        /* Kalman update with H = [1, 0] */
        S = P00 + R;
        K0 = P00 / S;
        K1 = P10 / S;
        innov = z - xk;
        xk += K0 * innov;
        vk += K1 * innov;
        n00 = (1.0 - K0) * P00;
        n01 = (1.0 - K0) * P01;
        n10 = P10 - K1 * P00;
        n11 = P11 - K1 * P01;
        P00 = n00; P01 = n01; P10 = n10; P11 = n11;

        se_meas += (z - x) * (z - x);
        se_phys += (xp - x) * (xp - x);
        se_kal += (xk - x) * (xk - x);
        fprintf(fp, "%.3f,%.6f,%.6f,%.6f,%.6f,%.6f\n", i * dt, x, z, xp, xk, vk);
    }
    fclose(fp);
    printf("position RMSE over %d steps:\n", n);
    printf("  sensor only  %.3f m\n", sqrt(se_meas / n));
    printf("  physics only %.3f m\n", sqrt(se_phys / n));
    printf("  Kalman       %.3f m\n", sqrt(se_kal / n));
    printf("final velocity: truth %.3f, Kalman %.3f m/s\n", v, vk);
    return 0;
}
