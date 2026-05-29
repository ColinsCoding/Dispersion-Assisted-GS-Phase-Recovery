/*
 * beamformer.c  —  Jalali Lab Combat Robot Controller
 * =====================================================
 * Phased-array RF / optical beamformer
 * + Marx-bank supercapacitor EMP discharge sequencer
 * + BLDC motor PID velocity controller
 * + Autonomous combat FSM (PATROL→ACQUIRE→CHARGE→FIRE→EVADE)
 *
 * Compile & run (host simulation):
 *   gcc -O2 -std=c11 -lm firmware/beamformer.c -o beamformer && ./beamformer
 *
 * Cross-compile (ARM Cortex-M4, STM32F446):
 *   arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 \
 *     -mfloat-abi=hard -O2 -std=c11 -DEMBEDDED \
 *     firmware/beamformer.c -lm -o beamformer.elf
 *
 * Electronics vs Photonics battle card:
 *   RF phased array : f=2.4 GHz, N=8,   d=λ/2=62.5 mm  → HPBW≈12.7°
 *   Optical OPA     : λ=1550 nm, N=512, d=2 μm          → HPBW≈0.086°
 *   Marx bank EMP   : 8 × (1 mF @ 1 kV) → 8 kV, ~2 kA peak, ~4 MW
 *   EMP E-field     : >200 V/m at 10 m  → latch-up;  >2 kV/m → destroy
 */

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ── Physical constants ──────────────────────────────────────────────────── */
#define PI          3.14159265358979f
#define TWO_PI      6.28318530717959f
#define C_LIGHT     2.998e8f           /* m/s                              */
#define Z0_FREE     376.73f            /* Ω  free-space impedance          */

/* ── RF phased array config ──────────────────────────────────────────────── */
#define RF_FREQ_GHZ 2.4f               /* GHz                              */
#define RF_N        8                  /* number of elements               */
#define RF_D_LAM    0.5f               /* element spacing / wavelength     */

/* ── Optical phased array (OPA) config ───────────────────────────────────── */
#define OPA_LAMBDA_NM  1550.0f         /* nm, operating wavelength         */
#define OPA_N          512             /* elements on photonic chip        */
#define OPA_D_UM       2.0f            /* μm, element pitch                */
/* d/λ = 2 μm / 1550 nm = 1.29  (grating lobes present, but HPBW still tiny) */
#define OPA_D_LAM    (OPA_D_UM * 1e-6f / (OPA_LAMBDA_NM * 1e-9f))

/* ── Marx bank supercapacitor EMP ────────────────────────────────────────── */
#define MARX_N      8                  /* stages                           */
#define MARX_C      1e-3f              /* F   capacitance per stage        */
#define MARX_V0     1000.0f            /* V   initial charge per stage     */
#define MARX_L_UH   10.0f              /* μH  total stray inductance       */
#define MARX_R_INT  0.05f              /* Ω   ESR per stage                */
#define MARX_R_ANT  1.0f               /* Ω   antenna load resistance      */
#define EFF_RAD     0.25f              /* fraction of P_ant that radiates  */
#define E_LATCHUP   200.0f             /* V/m latch-up threshold           */
#define E_DESTROY   2000.0f            /* V/m hard-kill threshold          */

/* ── BLDC motor (150 W brushless, 24 V bus) ──────────────────────────────── */
#define MOTOR_KT    0.12f              /* N·m/A  torque constant           */
#define MOTOR_KE    0.12f              /* V·s/rad back-EMF constant        */
#define MOTOR_R     0.80f              /* Ω   phase resistance             */
#define MOTOR_L     500e-6f            /* H   phase inductance             */
#define MOTOR_J     0.0015f            /* kg·m²  rotor inertia             */
#define MOTOR_B     8e-4f              /* N·m·s/rad viscous damping        */
#define V_BUS       24.0f              /* V   battery bus voltage          */

/* ── ANSI colour codes ───────────────────────────────────────────────────── */
#define CYN   "\033[0;36m"
#define GRN   "\033[0;32m"
#define YLW   "\033[1;33m"
#define RED   "\033[0;31m"
#define MAG   "\033[0;35m"
#define BLU   "\033[0;34m"
#define BOLD  "\033[1m"
#define NC    "\033[0m"

/* ═══════════════════════════════════════════════════════════════════════════
 *  Data types
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float kp, ki, kd;
    float integrator;
    float prev_error;
    float out_min, out_max;
} PID;

typedef struct {
    float omega;      /* rad/s   angular velocity   */
    float theta;      /* rad     shaft position     */
    float current;    /* A       phase current      */
} Motor;

typedef struct {
    float V_cap;      /* V       remaining cap voltage (series sum) */
    float I_dis;      /* A       discharge current                  */
    float E_stored;   /* J       initial stored energy              */
    float E_ant;      /* J       cumulative energy into antenna      */
    int   charged;
    int   fired;
} Marx;

typedef enum {
    STATE_PATROL = 0,
    STATE_ACQUIRE,
    STATE_CHARGE,
    STATE_FIRE,
    STATE_EVADE,
    STATE_COUNT
} RobotState;

static const char *STATE_NAMES[STATE_COUNT]  = {
    "PATROL", "ACQUIRE", "CHARGE", "FIRE", "EVADE"
};
static const char *STATE_COLORS[STATE_COUNT] = {
    GRN, CYN, YLW, RED, MAG
};

typedef struct {
    RobotState state;
    Motor      drive;
    Motor      turret;
    Marx       marx;
    PID        pid_drive;
    PID        pid_turret;
    float      t;
    float      state_timer;
    float      omega_cmd;      /* rad/s drive velocity setpoint */
    float      theta_target;   /* rad   turret position setpoint */
    int        target_acquired;
} Robot;

/* ═══════════════════════════════════════════════════════════════════════════
 *  Phased array beam pattern
 *
 *  |AF(θ)|² normalised to 1 at θ=θ₀:
 *    ψ   = π (d/λ)(sin θ − sin θ₀)
 *    AF  = [sin(N ψ) / (N sin ψ)]²
 * ═══════════════════════════════════════════════════════════════════════════ */

static float beam_af(float theta_deg, float theta0_deg, float d_lam, int N)
{
    float psi = PI * d_lam * (sinf(theta_deg * PI / 180.0f)
                            - sinf(theta0_deg * PI / 180.0f));
    if (fabsf(psi) < 1e-7f) return 1.0f;
    float s = sinf((float)N * psi);
    float d = (float)N * sinf(psi);
    return (s * s) / (d * d);
}

/* HPBW (degrees): full width at half power via numerical search, 0.02° steps */
static float beam_hpbw_deg(float d_lam, int N, float theta0_deg)
{
    float step  = 0.02f;
    float theta0 = theta0_deg;
    for (float delta = 0.0f; delta < 90.0f; delta += step) {
        if (beam_af(theta0 + delta, theta0, d_lam, N) < 0.5f)
            return 2.0f * delta;
    }
    return 180.0f;
}

/* ASCII beam map: one character per 2° across ±90° */
static void print_beam_map(float d_lam, int N, float steer_deg)
{
    printf("  θ₀=%+5.0f°  [", steer_deg);
    for (int deg = -90; deg <= 90; deg += 2) {
        float af = beam_af((float)deg, steer_deg, d_lam, N);
        char c = deg == 0        ? '|'
               : af > 0.50f      ? '#'
               : af > 0.05f      ? '+'
               : af > 0.005f     ? '.'
               :                   ' ';
        putchar(c);
    }
    printf("]  HPBW=%.2f°\n", beam_hpbw_deg(d_lam, N, steer_deg));
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  PID controller
 * ═══════════════════════════════════════════════════════════════════════════ */

static void pid_init(PID *p, float kp, float ki, float kd,
                     float out_min, float out_max)
{
    p->kp = kp;  p->ki = ki;  p->kd = kd;
    p->integrator  = 0.0f;
    p->prev_error  = 0.0f;
    p->out_min     = out_min;
    p->out_max     = out_max;
}

static float pid_update(PID *p, float setpoint, float meas, float dt)
{
    float e   = setpoint - meas;
    /* Anti-windup: clamp integrator before using */
    p->integrator += e * dt;
    float i_max = (p->ki > 1e-9f) ? p->out_max / p->ki : 1e6f;
    float i_min = (p->ki > 1e-9f) ? p->out_min / p->ki : -1e6f;
    if (p->integrator >  i_max) p->integrator =  i_max;
    if (p->integrator < i_min) p->integrator = i_min;
    float deriv = (dt > 1e-9f) ? (e - p->prev_error) / dt : 0.0f;
    p->prev_error = e;
    float u = p->kp * e + p->ki * p->integrator + p->kd * deriv;
    if (u > p->out_max) u = p->out_max;
    if (u < p->out_min) u = p->out_min;
    return u;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  BLDC motor model (simplified electrical + mechanical)
 *
 *  Electrical:  V = R·I + L·dI/dt + Ke·ω   →   dI/dt = (V−R·I−Ke·ω)/L
 *  Mechanical:  J·dω/dt = Kt·I − B·ω − τ_load
 * ═══════════════════════════════════════════════════════════════════════════ */

static void motor_step(Motor *m, float V_app, float tau_load, float dt)
{
    float dI     = (V_app - MOTOR_R * m->current - MOTOR_KE * m->omega) / MOTOR_L;
    float domega = (MOTOR_KT * m->current - MOTOR_B * m->omega - tau_load) / MOTOR_J;
    m->current += dI     * dt;
    m->omega   += domega * dt;
    m->theta   += m->omega * dt;
    /* Current limits (30 A peak) */
    if (m->current >  30.0f) m->current =  30.0f;
    if (m->current < -30.0f) m->current = -30.0f;
    if (m->omega   <   0.0f) m->omega   =   0.0f;   /* rectified: positive speed only */
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Marx bank RLC discharge (Euler ODE)
 *
 *  Series circuit: L_total, R_total = N·R_int + R_ant, C_eff = C/N
 *
 *  L · dI/dt       = V_C − R_total · I
 *  C_eff · dV_C/dt = −I
 * ═══════════════════════════════════════════════════════════════════════════ */

static void marx_charge(Marx *m)
{
    m->V_cap   = (float)MARX_N * MARX_V0;                        /* series   */
    m->I_dis   = 0.0f;
    m->E_stored = 0.5f * (MARX_C / (float)MARX_N) * m->V_cap * m->V_cap;
    m->E_ant   = 0.0f;
    m->charged = 1;
    m->fired   = 0;
}

/* One Euler step; returns instantaneous power (W) into R_ant */
static float marx_step(Marx *m, float dt)
{
    if (!m->charged) return 0.0f;
    float C_eff  = MARX_C / (float)MARX_N;
    float R_tot  = (float)MARX_N * MARX_R_INT + MARX_R_ANT;
    float L_tot  = MARX_L_UH * 1e-6f;

    float dI = (m->V_cap - R_tot * m->I_dis) / L_tot;
    float dV = -m->I_dis / C_eff;
    m->I_dis  += dI * dt;
    m->V_cap  += dV * dt;
    if (m->I_dis < 0.0f) m->I_dis = 0.0f;
    if (m->V_cap < 0.0f) { m->V_cap = 0.0f; m->I_dis = 0.0f; }

    float P_ant = m->I_dis * m->I_dis * MARX_R_ANT;
    m->E_ant   += P_ant * dt;
    return P_ant;
}

/* Far-field E-field (V/m) at distance r (m) from isotropic P_rad (W) */
static float e_field(float P_rad, float r_m)
{
    return sqrtf(Z0_FREE * P_rad / (4.0f * PI * r_m * r_m + 1e-9f));
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Standalone Marx discharge print (detailed table)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_marx_table(void)
{
    Marx m;
    marx_charge(&m);

    float C_eff  = MARX_C / (float)MARX_N;
    float R_tot  = (float)MARX_N * MARX_R_INT + MARX_R_ANT;
    float L_tot  = MARX_L_UH * 1e-6f;
    float Z_char = sqrtf(L_tot / C_eff);
    float omega0 = 1.0f / sqrtf(L_tot * C_eff);
    float zeta   = R_tot / (2.0f * Z_char);

    printf(BOLD CYN "\n[MARX BANK]  %d stages × (%.0f mF @ %.0f V)\n" NC,
           MARX_N, MARX_C * 1e3f, MARX_V0);
    printf("  V_Marx   = %6.0f V      C_eff = %.3f mF\n",
           (float)MARX_N * MARX_V0, C_eff * 1e3f);
    printf("  L_total  = %6.1f μH    R_total = %.2f Ω\n",
           MARX_L_UH, R_tot);
    printf("  Z_char   = %6.3f Ω    ω₀ = %.1f krad/s  f₀ = %.1f kHz\n",
           Z_char, omega0 / 1e3f, omega0 / TWO_PI / 1e3f);
    printf("  ζ        = %6.3f  (%s)\n", zeta,
           zeta < 0.99f ? "underdamped (ring)" :
           zeta < 1.01f ? "critically damped"  : "overdamped");
    printf("  E_stored = %6.3f kJ\n\n", m.E_stored / 1e3f);

    printf("  %-8s %-9s %-9s %-11s %-12s %-14s %s\n",
           "t[μs]", "V[V]", "I[A]", "P_ant[kW]",
           "E@10m[V/m]", "E@1m[V/m]", "Threat");
    printf("  %s\n", "─────────────────────────────────────────────────────────────────");

    float dt = 1e-6f;   /* 1 μs per step */
    float t_us = 0.0f;
    float I_peak = 0.0f;
    float P_peak = 0.0f;

    for (int step = 0; step <= 500; step++) {
        float P = marx_step(&m, dt);
        t_us += 1.0f;
        if (m.I_dis > I_peak) I_peak = m.I_dis;
        if (P         > P_peak) P_peak = P;

        if (step % 50 == 0) {
            float P_rad  = EFF_RAD * P;
            float E10    = e_field(P_rad, 10.0f);
            float E1     = e_field(P_rad,  1.0f);
            const char *threat =
                E10 >= E_DESTROY  ? RED  BOLD "DESTROY" NC :
                E10 >= E_LATCHUP  ? YLW       "LATCH-UP" NC :
                                    GRN        "safe" NC;
            printf("  %-8.0f %-9.0f %-9.0f %-11.1f %-12.0f %-14.0f %s\n",
                   t_us, m.V_cap, m.I_dis,
                   P / 1e3f, E10, E1, threat);
        }
    }
    printf("  ─────────────────────────────────────────────────────────────────\n");
    printf("  I_peak = %.0f A   P_peak = %.2f MW   E_ant total = %.1f J  (%.0f%% of stored)\n",
           I_peak, P_peak / 1e6f, m.E_ant,
           100.0f * m.E_ant / m.E_stored);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Robot FSM
 * ═══════════════════════════════════════════════════════════════════════════ */

static void robot_init(Robot *r)
{
    memset(r, 0, sizeof(*r));
    r->state       = STATE_PATROL;
    r->omega_cmd   = 20.0f;     /* rad/s  ~190 rpm patrol */
    r->theta_target = 0.0f;
    /* Drive PID: velocity loop, output = motor voltage [0, 24 V] */
    pid_init(&r->pid_drive,   8.0f, 2.0f, 0.05f,  0.0f, V_BUS);
    /* Turret PID: position loop, output = motor voltage [−24, 24 V] */
    pid_init(&r->pid_turret, 12.0f, 1.0f, 0.10f, -V_BUS, V_BUS);
}

static void robot_step(Robot *r, float dt)
{
    r->t           += dt;
    r->state_timer += dt;

    /* Drive motor velocity control */
    float V_d = pid_update(&r->pid_drive, r->omega_cmd, r->drive.omega, dt);
    motor_step(&r->drive, V_d, 0.05f, dt);

    /* Turret position control */
    float V_t = pid_update(&r->pid_turret, r->theta_target, r->turret.theta, dt);
    motor_step(&r->turret, V_t, 0.002f, dt);

    switch (r->state) {

    case STATE_PATROL:
        r->omega_cmd = 20.0f;
        if (r->state_timer > 0.8f) {
            r->state = STATE_ACQUIRE;
            r->state_timer = 0.0f;
        }
        break;

    case STATE_ACQUIRE:
        r->omega_cmd    = 8.0f;           /* slow while scanning        */
        r->theta_target = 0.785f;         /* slew turret to 45°         */
        if (fabsf(r->turret.theta - r->theta_target) < 0.05f) {
            r->target_acquired = 1;
        }
        if (r->target_acquired && r->state_timer > 0.6f) {
            r->state = STATE_CHARGE;
            r->state_timer = 0.0f;
            marx_charge(&r->marx);
        }
        break;

    case STATE_CHARGE:
        r->omega_cmd = 3.0f;              /* near-stop while charging   */
        if (r->state_timer > 1.2f) {
            r->state = STATE_FIRE;
            r->state_timer = 0.0f;
            r->marx.fired = 1;
        }
        break;

    case STATE_FIRE:
        r->omega_cmd = 0.0f;
        marx_step(&r->marx, dt);          /* discharge every FSM tick   */
        if (r->state_timer > 0.60f || r->marx.V_cap < 5.0f) {
            r->state = STATE_EVADE;
            r->state_timer = 0.0f;
        }
        break;

    case STATE_EVADE:
        r->omega_cmd    = 62.8f;          /* 600 rpm escape sprint      */
        r->theta_target = -0.785f;        /* turret rear                */
        if (r->state_timer > 1.5f) {
            r->state       = STATE_PATROL;
            r->state_timer = 0.0f;
            r->target_acquired = 0;
        }
        break;

    default: break;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  main()
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf(BOLD CYN
           "\n━━━ Jalali Lab Combat Robot — Physics Simulation ━━━\n"
           "    Electronics vs Photonics | Marx EMP | BLDC PID\n\n" NC);

    /* ── 1. RF phased array ─────────────────────────────────────────────── */
    printf(BOLD BLU "[RF PHASED ARRAY]  N=%d, f=%.1f GHz, d=λ/%.0f\n" NC,
           RF_N, RF_FREQ_GHZ, 1.0f / RF_D_LAM);
    float lam_rf_mm = (C_LIGHT / (RF_FREQ_GHZ * 1e9f)) * 1e3f;
    printf("  λ = %.1f mm  →  d = %.1f mm  (no grating lobes for d≤λ/2)\n",
           lam_rf_mm, lam_rf_mm * RF_D_LAM);

    float rf_steers[] = { 0.0f, 30.0f, 60.0f };
    for (int i = 0; i < 3; i++)
        print_beam_map(RF_D_LAM, RF_N, rf_steers[i]);

    float hpbw_rf = beam_hpbw_deg(RF_D_LAM, RF_N, 0.0f);

    /* ── 2. Optical phased array (OPA) ──────────────────────────────────── */
    printf(BOLD BLU "\n[OPTICAL PHASED ARRAY (OPA)]  N=%d, λ=%.0f nm, d=%.1f μm\n" NC,
           OPA_N, OPA_LAMBDA_NM, OPA_D_UM);
    printf("  d/λ = %.3f  →  grating lobes at θ_GL = ±%.1f°\n",
           OPA_D_LAM, asinf(1.0f / OPA_D_LAM) * 180.0f / PI);
    print_beam_map(OPA_D_LAM, OPA_N, 0.0f);

    float hpbw_opa = beam_hpbw_deg(OPA_D_LAM, OPA_N, 0.0f);
    printf("  HPBW (RF)  = %.2f°\n", hpbw_rf);
    printf("  HPBW (OPA) = %.4f°  (%.3f mrad)\n",
           hpbw_opa, hpbw_opa * PI / 180.0f * 1e3f);
    printf("  " BOLD GRN "OPA beam is %.0f× narrower" NC
           " → surgical targeting; photons don't EMP\n",
           hpbw_rf / hpbw_opa);

    /* ── 3. Marx bank EMP discharge ─────────────────────────────────────── */
    print_marx_table();

    /* ── 4. Combat robot FSM trace ──────────────────────────────────────── */
    printf(BOLD BLU "\n[COMBAT ROBOT FSM]  dt=5 ms, T_sim=7 s\n" NC);
    Robot robot;
    robot_init(&robot);

    RobotState prev = (RobotState)(-1);
    int n_steps = (int)(7.0f / 0.005f);
    float fire_P_peak = 0.0f;

    for (int i = 0; i < n_steps; i++) {
        robot_step(&robot, 0.005f);
        if (robot.state == STATE_FIRE) {
            float P = robot.marx.I_dis * robot.marx.I_dis * MARX_R_ANT;
            if (P > fire_P_peak) fire_P_peak = P;
        }
        if (robot.state != prev) {
            printf("  t=%5.2f s  %s%-8s%s ω=%.0f rpm  I=%.1f A",
                   robot.t,
                   STATE_COLORS[robot.state], STATE_NAMES[robot.state], NC,
                   robot.drive.omega * 60.0f / TWO_PI,
                   robot.drive.current);
            if (robot.state == STATE_FIRE || robot.state == STATE_EVADE)
                printf("  Marx: V=%.0fV I=%.0fA",
                       robot.marx.V_cap, robot.marx.I_dis);
            printf("\n");
            prev = robot.state;
        }
    }

    /* ── 5. Battle report ───────────────────────────────────────────────── */
    float EIRP_rf   = (float)(RF_N * RF_N) * 5.0f;  /* 5 W/element × N² gain */
    float E_rf_10m  = e_field(EIRP_rf * EFF_RAD, 10.0f);
    float E_emp_10m = e_field(fire_P_peak * EFF_RAD, 10.0f);
    float E_emp_1m  = e_field(fire_P_peak * EFF_RAD,  1.0f);

    printf(BOLD CYN "\n━━━ BATTLE REPORT ━━━\n" NC);
    printf("  RF array EIRP   = %.0f W  →  E-field %.0f V/m at 10 m  (%s)\n",
           EIRP_rf, E_rf_10m,
           E_rf_10m > E_LATCHUP ? YLW "LATCH-UP zone" NC : GRN "safe" NC);
    printf("  Marx peak power = %.2f MW  →  %.0f V/m@10m / %.0f V/m@1m  (%s)\n",
           fire_P_peak / 1e6f, E_emp_10m, E_emp_1m,
           E_emp_10m > E_DESTROY  ? RED BOLD "DESTROY" NC :
           E_emp_10m > E_LATCHUP  ? YLW "LATCH-UP" NC    : GRN "safe" NC);
    printf("  OPA  HPBW       = %.4f°  →  " GRN "surgical, no EM coupling" NC "\n",
           hpbw_opa);
    printf("  " BOLD "Verdict:" NC
           " RF array saturates (area denial), OPA tracks (precision),\n"
           "         Marx bank closes the kill (near-field electronics KO).\n"
           "         " BOLD GRN "HYBRID ARCHITECTURE WINS." NC "\n\n");
    return 0;
}
