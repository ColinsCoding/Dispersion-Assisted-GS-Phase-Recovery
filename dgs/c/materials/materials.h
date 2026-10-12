/* materials.h -- header-only Sellmeier material library (C99, no dependencies but libm).
 *
 * The same table and formulas as dgs/materials.py:
 *   n^2(lam) = 1 + sum_i B_i lam^2 / (lam^2 - C_i),  lam in micrometers.
 * Derivatives are in closed form, so n, n_g, beta2 and D cost a few dozen flops each.
 * Sources: see the docstring of dgs/materials.py (Malitson 1965, SCHOTT N-BK7,
 * Malitson & Dodge 1972, Malitson 1963, Zelmon 1997).
 */
#ifndef DGS_MATERIALS_H
#define DGS_MATERIALS_H

#include <math.h>
#include <string.h>

#ifdef __CUDACC__
#define MAT_FN __host__ __device__ static inline
#else
#define MAT_FN static inline
#endif

#define MAT_C0 299792458.0
#define MAT_PI 3.14159265358979323846

typedef struct {
    const char *name;
    double B[3];
    double C[3];          /* um^2 */
    double lam_min_um, lam_max_um;
} material_t;

#define MAT_COUNT 6
static const material_t MATERIALS[MAT_COUNT] = {
    {"fused_silica", {0.6961663, 0.4079426, 0.8974794},
     {0.0684043*0.0684043, 0.1162414*0.1162414, 9.896161*9.896161}, 0.21, 6.7},
    {"n_bk7", {1.03961212, 0.231792344, 1.01046945},
     {0.00600069867, 0.0200179144, 103.560653}, 0.3, 2.5},
    {"sapphire_o", {1.4313493, 0.65054713, 5.3414021},
     {0.0726631*0.0726631, 0.1193242*0.1193242, 18.028251*18.028251}, 0.2, 5.5},
    {"caf2", {0.5675888, 0.4710914, 3.8484723},
     {0.050263605*0.050263605, 0.1003909*0.1003909, 34.649040*34.649040}, 0.23, 9.7},
    {"linbo3_e", {2.9804, 0.5981, 8.9543}, {0.02047, 0.0666, 416.08}, 0.4, 5.0},
    {"linbo3_o", {2.6734, 1.2290, 12.614}, {0.01764, 0.05914, 474.6}, 0.4, 5.0},
};

/* index of a material by exact name, or -1 */
static inline int mat_find(const char *name) {
    for (int i = 0; i < MAT_COUNT; i++)
        if (strcmp(MATERIALS[i].name, name) == 0) return i;
    return -1;
}

/* n and its first two wavelength derivatives (1/um, 1/um^2) from B and C */
MAT_FN void mat_n_derivs(const double *B, const double *C, double lam, double *n, double *n1, double *n2) {
    double lam2 = lam*lam, f = 1.0, f1 = 0.0, f2 = 0.0;
    for (int i = 0; i < 3; i++) {
        double d = lam2 - C[i];
        f  += B[i]*lam2/d;
        f1 -= 2.0*B[i]*C[i]*lam/(d*d);
        f2 += 2.0*B[i]*C[i]*(3.0*lam2 + C[i])/(d*d*d);
    }
    double nn = sqrt(f), d1 = f1/(2.0*nn);
    *n = nn; *n1 = d1; *n2 = (f2 - 2.0*d1*d1)/(2.0*nn);
}

MAT_FN double mat_n(const double *B, const double *C, double lam) {
    double n, n1, n2; mat_n_derivs(B, C, lam, &n, &n1, &n2); return n;
}
MAT_FN double mat_n_group(const double *B, const double *C, double lam) {
    double n, n1, n2; mat_n_derivs(B, C, lam, &n, &n1, &n2); return n - lam*n1;
}
/* beta2 in ps^2/km */
MAT_FN double mat_beta2(const double *B, const double *C, double lam) {
    double n, n1, n2; mat_n_derivs(B, C, lam, &n, &n1, &n2);
    double l = lam*1e-6;
    return l*l*l/(2.0*MAT_PI*MAT_C0*MAT_C0)*(n2*1e12)*1e27;
}
/* D in ps/(nm km) */
MAT_FN double mat_D(const double *B, const double *C, double lam) {
    double n, n1, n2; mat_n_derivs(B, C, lam, &n, &n1, &n2);
    return -(lam*1e-6)/MAT_C0*(n2*1e12)*1e6;
}

#endif /* DGS_MATERIALS_H */
