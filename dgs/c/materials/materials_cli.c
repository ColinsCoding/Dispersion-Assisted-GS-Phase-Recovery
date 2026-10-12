/* materials_cli.c -- print n, n_g, beta2, D for every material valid at one wavelength.
 *   gcc -O2 -Wall -std=c99 materials_cli.c -o materials_cli -lm
 *   ./materials_cli 1.55
 * Output: CSV, one material per line. Exit codes: 0 ok, 2 bad arguments.
 */
#include <stdio.h>
#include <stdlib.h>
#include "materials.h"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: materials_cli wavelength_um\n"); return 2; }
    char *end; double lam = strtod(argv[1], &end);
    if (*end != '\0' || lam <= 0.0) { fprintf(stderr, "wavelength must be a positive number in um\n"); return 2; }
    printf("name,n,n_group,beta2_ps2_per_km,D_ps_per_nm_km\n");
    for (int i = 0; i < MAT_COUNT; i++) {
        const material_t *m = &MATERIALS[i];
        if (lam < m->lam_min_um || lam > m->lam_max_um) continue;
        printf("%s,%.12f,%.12f,%.9f,%.9f\n", m->name, mat_n(m->B, m->C, lam),
               mat_n_group(m->B, m->C, lam), mat_beta2(m->B, m->C, lam), mat_D(m->B, m->C, lam));
    }
    return 0;
}
