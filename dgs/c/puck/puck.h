/*
 * puck.h - air hockey puck physics in C99 (statics, dynamics, 2-D rotation).
 *
 * Units: SI (m, kg, s, rad). The table spans 0..TABLE_W in x and 0..TABLE_H in y; the goal
 * is an opening of width GOAL_W centered on the right wall (x = TABLE_W).
 */
#ifndef PUCK_H
#define PUCK_H

#define TABLE_W 2.0
#define TABLE_H 1.0
#define GOAL_W 0.30
#define G_ACCEL 9.80665

typedef struct {
    double x, y;      /* center position (m) */
    double vx, vy;    /* velocity (m/s) */
    double theta;     /* spin angle (rad) */
    double omega;     /* spin rate (rad/s) */
    double R, h, M;   /* radius (m), height (m), mass (kg) */
} Puck;

typedef struct {
    double restitution; /* wall bounce: 1 = elastic */
    double drag;        /* linear air-cushion drag (1/s): dv/dt = -drag v */
    double spin_drag;   /* domega/dt = -spin_drag omega */
} Table;

typedef enum { PUCK_MOVING = 0, PUCK_STOPPED = 1, PUCK_GOAL = 2 } PuckState;

Puck puck_make(double R, double h, double M);

/* linear algebra: rotate (x, y) by theta with the matrix [[c, -s], [s, c]] */
void rot2(double theta, double x, double y, double *xr, double *yr);

/* statics */
double inertia_axis(const Puck *p);          /* M R^2 / 2: spinning flat */
double inertia_diameter(const Puck *p);      /* M (R^2/4 + h^2/12): flipping */
double hover_pressure(const Puck *p);        /* air pressure (Pa) under the face that balances M g */

/* dynamics */
double kinetic_energy(const Puck *p);        /* M v^2 / 2 + I w^2 / 2 */
PuckState puck_step(Puck *p, const Table *t, double dt);
PuckState puck_run(Puck *p, const Table *t, double dt, int max_steps, int *steps_taken);

#endif
