/*
 * puck_test.c - CSC 60 style unit tests: one function per behavior, a CHECK macro, and a
 * summary line. Exit code is the number of failures (0 = all passed).
 *
 *   gcc -std=c99 -Wall -Wextra -pedantic puck.c puck_test.c -o puck_test -lm && ./puck_test
 */
#include <math.h>
#include <stdio.h>

#include "puck.h"

static int failures = 0, checks = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        checks++;                                                            \
        if (!(cond)) {                                                       \
            failures++;                                                      \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                    \
    } while (0)

#define NEAR(a, b, tol) (fabs((a) - (b)) <= (tol) * (1.0 + fabs(b)))

static Table frictionless(void)
{
    Table t = {1.0, 0.0, 0.0};
    return t;
}

static void test_statics(void)
{
    Puck p = puck_make(0.0325, 0.005, 0.017);
    CHECK(NEAR(inertia_axis(&p), 0.5 * 0.017 * 0.0325 * 0.0325, 1e-12));
    CHECK(inertia_diameter(&p) < inertia_axis(&p)); /* thin disk: easier to flip than to spin */
    CHECK(NEAR(hover_pressure(&p), 50.24, 1e-3));   /* 17 g on a 6.5 cm face: ~50 Pa */
}

static void test_rotation_matrix(void)
{
    double xr, yr;
    rot2(1.2345, 3.0, 4.0, &xr, &yr);
    CHECK(NEAR(xr * xr + yr * yr, 25.0, 1e-12)); /* rotation preserves length */
    rot2(1.5707963267948966, 1.0, 0.0, &xr, &yr);
    CHECK(NEAR(xr, 0.0, 1e-12) && NEAR(yr, 1.0, 1e-12));
}

static void test_energy_conserved_without_friction(void)
{
    Table t = frictionless();
    Puck p = puck_make(0.03, 0.005, 0.02);
    double E0;
    int i;
    p.x = 0.5; p.y = 0.3; p.vx = 1.7; p.vy = 0.9; p.omega = 25.0;
    E0 = kinetic_energy(&p);
    for (i = 0; i < 20000; i++)
        if (puck_step(&p, &t, 1e-4) == PUCK_GOAL)
            break;
    CHECK(NEAR(kinetic_energy(&p), E0, 1e-12));
}

static void test_bounce_reverses_and_loses_energy(void)
{
    Table t = {0.8, 0.0, 0.0};
    Puck p = puck_make(0.03, 0.005, 0.02);
    p.x = 0.04; p.y = 0.2; p.vx = -2.0;
    puck_step(&p, &t, 0.01);
    CHECK(NEAR(p.vx, 1.6, 1e-12));
    CHECK(p.x >= p.R);
}

static void test_goal_and_stop(void)
{
    Table t = {0.9, 0.5, 1.0};
    Puck p = puck_make(0.03, 0.005, 0.02);
    int n;
    p.x = 0.2; p.y = TABLE_H / 2.0; p.vx = 3.0;
    CHECK(puck_run(&p, &t, 1e-3, 100000, &n) == PUCK_GOAL);
    p = puck_make(0.03, 0.005, 0.02);
    p.x = 0.2; p.y = 0.1; p.vx = 0.3;                  /* slow, off-center: drag stops it */
    CHECK(puck_run(&p, &t, 1e-3, 100000, &n) == PUCK_STOPPED);
    CHECK(n > 0);
}

int main(void)
{
    test_statics();
    test_rotation_matrix();
    test_energy_conserved_without_friction();
    test_bounce_reverses_and_loses_energy();
    test_goal_and_stop();
    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures;
}
