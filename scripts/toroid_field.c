/* toroid_field.c -- numerically stable Biot-Savart field of a toroid, in C.
 *
 * WHY C: the field map is a tight O(field points x wire segments) loop -- exactly
 * the kind of inner numeric kernel that is slow in pure Python and fast in C.
 * NUMERICAL STABILITY: the Biot-Savart integrand goes as 1/r^3, so it blows up when
 * a field point lands on a wire. We guard it -- any point closer than DMIN to a
 * segment is reported as masked instead of returning a garbage spike.
 *
 * Scans |B| along the +x radial line (z=0) and compares to the Ampere closed form
 * B = mu0 N I / (2 pi s) inside the tube, 0 outside.
 * Compile:  cc -O2 -o toroid_field toroid_field.c -lm     Run: ./toroid_field
 */
#include <stdio.h>
#include <math.h>

static const double PI = 3.14159265358979323846;

int main(void) {
    const double R = 0.5, a = 0.08, I = 2.0;     /* major radius, tube radius, current */
    const int N = 80, nper = 60;                 /* turns, segments per turn */
    const double MU0 = 4e-7 * PI, DMIN = 0.3 * a;
    const double scale = MU0 * I / (4 * PI);

    printf("   s(m)   |B|(uT)    Ampere mu0 N I/(2 pi s)(uT)\n");
    for (double s = 0.10; s <= 1.20 + 1e-9; s += 0.05) {
        double Bx = 0, By = 0, Bz = 0;
        int masked = 0;
        double px = s, py = 0, pz = 0;           /* field point on the +x axis, z=0 */

        for (int k = 0; k < N && !masked; k++) {
            double phi = 2 * PI * k / N, cx = cos(phi), cy = sin(phi);
            double prevx = 0, prevy = 0, prevz = 0;
            for (int j = 0; j <= nper; j++) {
                double t = 2 * PI * j / nper;
                double wx = R * cx + a * cos(t) * cx;   /* point on turn k */
                double wy = R * cy + a * cos(t) * cy;
                double wz = a * sin(t);
                if (j > 0) {
                    double dlx = wx - prevx, dly = wy - prevy, dlz = wz - prevz;
                    double mx = 0.5*(wx+prevx), my = 0.5*(wy+prevy), mz = 0.5*(wz+prevz);
                    double rx = px-mx, ry = py-my, rz = pz-mz;
                    double r = sqrt(rx*rx + ry*ry + rz*rz);
                    if (r < DMIN) { masked = 1; break; }    /* stability guard */
                    double inv = 1.0 / (r*r*r);
                    Bx += (dly*rz - dlz*ry) * inv;          /* dl x r-hat / r^2 */
                    By += (dlz*rx - dlx*rz) * inv;
                    Bz += (dlx*ry - dly*rx) * inv;
                }
                prevx = wx; prevy = wy; prevz = wz;
            }
        }
        if (masked) { printf("  %5.2f   (near wire -- masked)\n", s); continue; }
        double Bmag = scale * sqrt(Bx*Bx + By*By + Bz*Bz);
        double amp = (s > R - a && s < R + a) ? MU0 * N * I / (2 * PI * s) : 0.0;
        printf("  %5.2f  %8.3f  %14.3f\n", s, Bmag * 1e6, amp * 1e6);
    }
    return 0;
}
