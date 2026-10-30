/*
 * puck_game.c - terminal mini game: shoot the puck into the goal on the right wall.
 *
 *   puck_game                       interactive: 3 shots, type "angle_deg speed spin"
 *   puck_game 10 2.5 30             one shot from the command line (used by the tests)
 *
 * The table is drawn in ASCII: '.' trail, 'O' puck, '|' goal mouth, and the puck's spin shown as
 * a marker on its rim rotated with the 2-D rotation matrix from puck.c.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "puck.h"

#define COLS 60
#define ROWS 16

static void plot(char grid[ROWS][COLS + 1], double x, double y, char ch)
{
    int c = (int)(x / TABLE_W * (COLS - 1) + 0.5);
    int r = (int)((1.0 - y / TABLE_H) * (ROWS - 1) + 0.5);
    if (c >= 0 && c < COLS && r >= 0 && r < ROWS)
        grid[r][c] = ch;
}

static int shoot(double angle_deg, double speed, double spin)
{
    Table t = {0.9, 0.15, 0.5};
    Puck p = puck_make(0.0325, 0.005, 0.017);
    char grid[ROWS][COLS + 1];
    PuckState s = PUCK_MOVING;
    double a = angle_deg * 3.14159265358979323846 / 180.0, mx, my;
    int r, k;

    if (speed <= 0.0 || speed > 20.0) {
        fprintf(stderr, "speed must be in (0, 20] m/s\n");
        return -1;
    }
    for (r = 0; r < ROWS; r++) {
        for (k = 0; k < COLS; k++)
            grid[r][k] = (r == 0 || r == ROWS - 1) ? '-' : ' ';
        grid[r][COLS] = '\0';
    }
    for (r = 0; r < ROWS; r++)
        if (fabs((1.0 - (double)r / (ROWS - 1)) * TABLE_H - TABLE_H / 2.0) <= GOAL_W / 2.0)
            grid[r][COLS - 1] = '|';

    p.x = 0.2; p.y = TABLE_H / 2.0;
    p.vx = speed * cos(a); p.vy = speed * sin(a); p.omega = spin;
    for (k = 0; k < 200000 && s == PUCK_MOVING; k++) {
        s = puck_step(&p, &t, 1e-3);
        if (k % 40 == 0)
            plot(grid, p.x, p.y, '.');
    }
    plot(grid, p.x, p.y, 'O');
    rot2(p.theta, 0.09, 0.0, &mx, &my);              /* spin marker, drawn 9 cm out to be visible */
    plot(grid, p.x + mx, p.y + my, '*');
    for (r = 0; r < ROWS; r++)
        printf("%s\n", grid[r]);
    printf("result=%s x=%.3f y=%.3f t=%.3f KE=%.6f theta=%.3f\n",
           s == PUCK_GOAL ? "GOAL" : s == PUCK_STOPPED ? "stopped" : "timeout",
           p.x, p.y, k * 1e-3, kinetic_energy(&p), p.theta);
    return s == PUCK_GOAL;
}

int main(int argc, char **argv)
{
    double ang, spd, spin;
    int shot, goals = 0;

    if (argc == 4)
        return shoot(atof(argv[1]), atof(argv[2]), atof(argv[3])) == 1 ? 0 : 1;
    if (argc != 1) {
        fprintf(stderr, "usage: %s [angle_deg speed spin]\n", argv[0]);
        return 2;
    }
    for (shot = 1; shot <= 3; shot++) {
        printf("shot %d: angle_deg speed spin > ", shot);
        if (scanf("%lf %lf %lf", &ang, &spd, &spin) != 3)
            break;
        goals += shoot(ang, spd, spin) == 1;
    }
    printf("score: %d goal(s)\n", goals);
    return 0;
}
