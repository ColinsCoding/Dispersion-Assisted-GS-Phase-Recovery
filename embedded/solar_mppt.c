/* solar_mppt.c  --  MPPT solar panel controller in embedded-C style.
 *
 * Panel spec: 1600 W peak, Voc = 36 V (open-circuit), Vmp ≈ 29 V (typical
 * 80% of Voc), Isc ≈ 44 A, Imp ≈ 55 A.
 *
 * Physics foundation (modern physics prereq):
 *   A solar cell is a p-n junction photodiode.  Incident photons with
 *   energy hν > Eg (bandgap) excite electrons across the gap — the
 *   photoelectric effect inside a semiconductor.  The I-V curve is:
 *
 *     I = Iph − I0 (exp(qV / nkT) − 1)
 *
 *   where Iph = photocurrent (∝ irradiance), I0 = dark saturation current,
 *   n = ideality factor, q = electron charge, k = Boltzmann, T = temperature.
 *   The maximum power point (MPP) sits at the knee of this curve.
 *
 * Algorithm: Perturb-and-Observe (P&O) MPPT.
 *   Every sample period:
 *     1. Measure V and I via ADC.
 *     2. Compute P = V × I.
 *     3. If P > P_prev, keep perturbing in the same direction.
 *        If P < P_prev, reverse direction.
 *     4. Adjust PWM duty cycle (controls boost converter) accordingly.
 *
 * Compile & simulate:
 *   gcc -Wall -std=c99 -O2 -o solar_mppt solar_mppt.c -lm && ./solar_mppt
 *
 * On a real MCU (STM32, ATmega, RP2040, etc.) replace adc_read() with the
 * hardware ADC peripheral read, and pwm_set_duty() with the timer/compare
 * register write.  Everything else is identical.
 */

#include <stdio.h>
#include <math.h>
#include <stdint.h>

/* ── Panel constants (1600 W, 36 V class) ─────────────────────────────── */
#define PANEL_VOC    36.0f       /* open-circuit voltage (V) */
#define PANEL_VMP    28.8f       /* max-power voltage  (≈ 0.80 × Voc) */
#define PANEL_ISC    44.0f       /* short-circuit current (A) */
#define PANEL_IMP    55.6f       /* max-power current  (A) */
#define PANEL_PMAX  1600.0f      /* rated peak power (W) */

/* ── Controller tuning ────────────────────────────────────────────────── */
#define DUTY_MIN     0.10f       /* 10% — don't fully short the panel */
#define DUTY_MAX     0.90f       /* 90% — leave some headroom */
#define DUTY_STEP    0.005f      /* perturbation step per cycle */
#define SAMPLE_HZ    1000        /* ADC sampling rate (Hz) */
#define MPPT_HZ      10          /* MPPT update rate (Hz) */
#define MPPT_DIVIDER (SAMPLE_HZ / MPPT_HZ)

/* ── Simulated I-V curve (single-diode model) ─────────────────────────── */
/* A 36 V panel has ~72 cells in series (≈0.5 V/cell).
 * Thermal voltage scales with cell count: V_T_panel = n·k·T/q · N_cells.
 * I0 is derived from the Voc condition I(Voc)=0:
 *   I0 = Iph · exp(−Voc / V_T_panel)                                    */
#define N_CELLS      72          /* series cells for a 36 V string */
#define IDEALITY_N   1.3f
#define K_BOLTZMANN  1.380649e-23f
#define Q_ELECTRON   1.60218e-19f
#define TEMP_K       298.15f     /* 25 °C */

/* V_T for the whole string */
static float vt_panel(void)
{
    return IDEALITY_N * K_BOLTZMANN * TEMP_K / Q_ELECTRON * N_CELLS;
}

/* Simulated photocurrent scales with irradiance fraction (0–1). */
static float panel_current(float V, float irradiance_frac)
{
    float Iph = PANEL_ISC * irradiance_frac;
    /* I0 chosen so I(Voc) = 0 at full irradiance */
    float I0  = PANEL_ISC * expf(-PANEL_VOC / vt_panel());
    float I   = Iph - I0 * (expf(V / vt_panel()) - 1.0f);
    if (I < 0.0f) I = 0.0f;
    if (V < 0.0f) I = 0.0f;
    return I;
}

/* ── Hardware abstraction layer (stub for simulation) ─────────────────── */

/* On real hardware these read ADC registers; here we simulate the panel. */
static float g_duty  = 0.50f;   /* current PWM duty cycle */
static float g_irrad = 1.00f;   /* simulated irradiance fraction */

static float adc_read_voltage(void)
{
    /* Boost converter: Vout = Vin / (1 − duty).
     * We're controlling the input (panel) side voltage by adjusting duty.
     * Panel voltage = Voc × (1 − duty) as a simple model. */
    float V = PANEL_VOC * (1.0f - g_duty);
    if (V < 0.0f) V = 0.0f;
    return V;
}

static float adc_read_current(void)
{
    float V = adc_read_voltage();
    return panel_current(V, g_irrad);
}

static void pwm_set_duty(float duty)
{
    if (duty < DUTY_MIN) duty = DUTY_MIN;
    if (duty > DUTY_MAX) duty = DUTY_MAX;
    g_duty = duty;
}

/* ── MPPT controller state ────────────────────────────────────────────── */
typedef struct {
    float V_prev;
    float P_prev;
    float duty;
    int   direction;   /* +1 increase duty, -1 decrease */
} MpptState;

static void mppt_init(MpptState *s)
{
    s->V_prev   = 0.0f;
    s->P_prev   = 0.0f;
    s->duty     = 0.50f;
    s->direction = +1;
}

/* Called once every MPPT_DIVIDER ADC samples. */
static void mppt_update(MpptState *s)
{
    float V = adc_read_voltage();
    float I = adc_read_current();
    float P = V * I;

    float dP = P - s->P_prev;
    float dV = V - s->V_prev;

    /* Perturb-and-Observe using dP/dV sign:
     *
     *   dP/dV > 0 → operating left of MPP (V < Vmp) → increase V → decrease duty
     *   dP/dV < 0 → operating right of MPP (V > Vmp) → decrease V → increase duty
     *   dP/dV ≈ 0 → at MPP, hold
     *
     * direction = -1 means "decrease duty next cycle" (V rises).
     * direction = +1 means "increase duty next cycle" (V falls).
     *
     * Dead-band |dP| < 0.5 W avoids hunting once locked onto MPP. */
    if (fabsf(dP) > 0.5f) {
        /* sign(dP) * sign(dV): positive → left of MPP, negative → right */
        int same_sign = ((dP > 0.0f) == (dV > 0.0f));
        s->direction  = same_sign ? -1 : +1;   /* -1 = increase V */
    }

    s->duty  += s->direction * DUTY_STEP;
    pwm_set_duty(s->duty);

    s->V_prev = V;
    s->P_prev = P;
}

/* ── Simulation loop ──────────────────────────────────────────────────── */
int main(void)
{
    MpptState mppt;
    mppt_init(&mppt);

    printf("Solar MPPT simulation  (1600 W panel, 36 V class)\n");
    printf("Panel MPP target: V=%.1f V, I=%.1f A, P=%.0f W\n\n",
           PANEL_VMP, PANEL_IMP, PANEL_PMAX);
    printf("%-6s %-8s %-8s %-8s %-8s %-6s\n",
           "tick", "V(V)", "I(A)", "P(W)", "duty", "irrad");
    printf("%s\n", "─────────────────────────────────────────────────────");

    int total_ticks = SAMPLE_HZ * 12;  /* simulate 12 seconds */
    int print_every = SAMPLE_HZ / 2;   /* print 2 times per second */

    for (int tick = 0; tick < total_ticks; tick++) {

        /* Simulate irradiance step at t=2 s (cloud passes over) */
        if (tick == SAMPLE_HZ * 2)
            g_irrad = 0.60f;

        /* MPPT update at lower rate than ADC */
        if (tick % MPPT_DIVIDER == 0)
            mppt_update(&mppt);

        if (tick % print_every == 0) {
            float V = adc_read_voltage();
            float I = adc_read_current();
            float P = V * I;
            printf("%-6d %-8.2f %-8.2f %-8.1f %-8.3f %-6.2f\n",
                   tick, V, I, P, g_duty, g_irrad);
        }
    }

    printf("\n");
    printf("Final operating point:\n");
    float Vf = adc_read_voltage();
    float If = adc_read_current();
    printf("  V = %.2f V   I = %.2f A   P = %.1f W   duty = %.3f\n",
           Vf, If, Vf * If, g_duty);
    printf("  Efficiency vs MPP: %.1f%%\n",
           100.0f * Vf * If / PANEL_PMAX);

    return 0;
}
