/* statics_hw2.c -- Hibbeler Statics Homework 2 (1-12, 1-18, 1-19, 2-8,
 * 2-10, 2-46, 2-101) in plain C, as an independent check of
 * notebooks/statics_hw2_sympy.ipynb.
 *
 * Prints one "key value" pair per line so the notebook can parse it.
 *
 * Build: gcc -O2 -Wall -o statics_hw2 scripts/statics_hw2.c -lm
 */
#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846
#define DEG (PI / 180.0)

int main(void)
{
    /* ---- Chapter 1: units ---- */
    const double ft = 0.3048;             /* m, exact */
    const double lb = 4.4482216152605;    /* N, exact */
    const double slug = lb / ft;          /* kg */
    const double g = 9.81;                /* m/s^2 */

    printf("1-12_rho_Mg_m3 %.6f\n", 1.94 * slug / (ft * ft * ft) / 1000.0);
    printf("1-18_um_kg_N %.6f\n", 0.204 * 0.00457 / 34.6 * 1e6);
    printf("1-19_rho_Mg_m3 %.6f\n", 520.0 * lb / (ft * ft * ft) / g / 1000.0);

    /* ---- 2-8: law of sines, triangle angles 30, 75, 75 deg ---- */
    printf("2-8_F2u_kN %.6f\n", 6.0 * sin(75 * DEG) / sin(75 * DEG));
    printf("2-8_F2v_kN %.6f\n", 6.0 * sin(30 * DEG) / sin(75 * DEG));

    /* ---- 2-10: FR = F1 + F3 ---- */
    double frx = 400 * cos(30 * DEG) - 250 * sin(30 * DEG);
    double fry = -400 * sin(30 * DEG) + 250 * cos(30 * DEG);
    printf("2-10_FR_N %.6f\n", hypot(frx, fry));
    printf("2-10_theta_deg %.6f\n", atan2(fry, frx) / DEG);

    /* ---- 2-46: FB so the resultant is 1500 N along +y ---- */
    double fbx = 700 * sin(30 * DEG);          /* magnitude of x component */
    double fby = 1500 - 700 * cos(30 * DEG);
    printf("2-46_FB_N %.6f\n", hypot(fbx, fby));
    printf("2-46_theta_deg %.6f\n", atan2(fby, fbx) / DEG);

    /* ---- 2-101: B(0,0,4); anchors (-3,0,0), (2,-3,0), A(x,y,0) ---- */
    double f1[3] = {350 * -3 / 5.0, 0.0, 350 * -4 / 5.0};
    double r2 = sqrt(2 * 2 + 3 * 3 + 4 * 4);
    double f2[3] = {250 * 2 / r2, 250 * -3 / r2, 250 * -4 / r2};
    double l = -(f1[0] + f2[0]) / 300.0;       /* direction cosines of F3 */
    double m = -(f1[1] + f2[1]) / 300.0;
    double n = sqrt(1 - l * l - m * m);
    double len = 4.0 / n;
    printf("2-101_x_m %.6f\n", l * len);
    printf("2-101_y_m %.6f\n", m * len);
    printf("2-101_FR_N %.6f\n", -(f1[2] + f2[2] - 300 * n));
    return 0;
}
