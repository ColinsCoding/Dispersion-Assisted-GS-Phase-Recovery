/* puck.c - implementation of puck.h */
#include <math.h>
#include <stddef.h>

#include "puck.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Puck puck_make(double R, double h, double M)
{
    Puck p = {0};
    p.R = R;
    p.h = h;
    p.M = M;
    return p;
}

void rot2(double theta, double x, double y, double *xr, double *yr)
{
    double c = cos(theta), s = sin(theta);
    *xr = c * x - s * y;
    *yr = s * x + c * y;
}

double inertia_axis(const Puck *p)
{
    return 0.5 * p->M * p->R * p->R;
}

double inertia_diameter(const Puck *p)
{
    return p->M * (p->R * p->R / 4.0 + p->h * p->h / 12.0);
}

double hover_pressure(const Puck *p)
{
    /* statics: sum of vertical forces = 0  ->  pressure * area = M g */
    return p->M * G_ACCEL / (M_PI * p->R * p->R);
}

double kinetic_energy(const Puck *p)
{
    double v2 = p->vx * p->vx + p->vy * p->vy;
    return 0.5 * p->M * v2 + 0.5 * inertia_axis(p) * p->omega * p->omega;
}

static int in_goal_mouth(double y)
{
    return fabs(y - TABLE_H / 2.0) <= GOAL_W / 2.0;
}

PuckState puck_step(Puck *p, const Table *t, double dt)
{
    /* semi-implicit Euler: update velocity (drag), then position */
    p->vx *= 1.0 - t->drag * dt;
    p->vy *= 1.0 - t->drag * dt;
    p->omega *= 1.0 - t->spin_drag * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    p->theta = fmod(p->theta + p->omega * dt, 2.0 * M_PI);

    if (p->x + p->R >= TABLE_W && in_goal_mouth(p->y))
        return PUCK_GOAL;
    /* walls: reflect the normal velocity only when moving into the wall */
    if (p->x - p->R <= 0.0 && p->vx < 0.0) {
        p->x = p->R;
        p->vx = -t->restitution * p->vx;
    }
    if (p->x + p->R >= TABLE_W && p->vx > 0.0) {
        p->x = TABLE_W - p->R;
        p->vx = -t->restitution * p->vx;
    }
    if (p->y - p->R <= 0.0 && p->vy < 0.0) {
        p->y = p->R;
        p->vy = -t->restitution * p->vy;
    }
    if (p->y + p->R >= TABLE_H && p->vy > 0.0) {
        p->y = TABLE_H - p->R;
        p->vy = -t->restitution * p->vy;
    }
    if (p->vx * p->vx + p->vy * p->vy < 1e-6)
        return PUCK_STOPPED;
    return PUCK_MOVING;
}

PuckState puck_run(Puck *p, const Table *t, double dt, int max_steps, int *steps_taken)
{
    PuckState s = PUCK_MOVING;
    int k;
    for (k = 0; k < max_steps && s == PUCK_MOVING; k++)
        s = puck_step(p, t, dt);
    if (steps_taken != NULL)
        *steps_taken = k;
    return s;
}
