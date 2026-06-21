/* faraday_emf.c -- Faraday's law in 20-odd lines of C.
 * Induced emf = -dPhi/dt, computed by a central finite difference, and compared to
 * the exact derivative. Flux through a loop in a sinusoidally varying field:
 *     Phi(t) = B0 * A * sin(w t)  ->  emf(t) = -dPhi/dt = -B0 A w cos(w t).
 * Compile:  cc -O2 -o faraday_emf faraday_emf.c -lm     Run: ./faraday_emf
 */
#include <stdio.h>
#include <math.h>

int main(void) {
    double B0 = 1.0, A = 0.01, w = 100.0, dt = 1e-4;   /* tesla, m^2, rad/s, s */
    int N = 21;
    double phi[64];
    for (int i = 0; i < N; i++)
        phi[i] = B0 * A * sin(w * i * dt);             /* sample the flux */

    printf("  t(ms)   Phi(mWb)   emf=-dPhi/dt(mV)   exact(mV)\n");
    for (int i = 1; i < N - 1; i++) {                  /* central difference, O(dt^2) */
        double emf   = -(phi[i + 1] - phi[i - 1]) / (2 * dt);
        double exact = -B0 * A * w * cos(w * i * dt);
        printf("  %5.2f  %8.4f  %14.4f  %10.4f\n",
               i * dt * 1e3, phi[i] * 1e3, emf * 1e3, exact * 1e3);
    }
    return 0;
}
