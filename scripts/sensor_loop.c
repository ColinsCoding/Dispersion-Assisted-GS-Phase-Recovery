/* sensor_loop.c -- the canonical embedded sensor-control pattern in C.
 *
 *   read sensor  ->  scale to units  ->  filter  ->  decide  ->  act,   forever.
 *
 * Here the "sensor" is SIMULATED (a noisy rising signal) so it runs on a desktop; on
 * real hardware read_adc() would read a memory-mapped ADC register instead. The five
 * embedded-C habits shown: fixed-width types, no dynamic memory, scaling raw counts to
 * physical units, a cheap one-pole filter, and a read/decide/act control loop.
 *
 * Compile:  cc -O2 -o sensor_loop sensor_loop.c -lm     Run:  ./sensor_loop
 */
#include <stdio.h>
#include <stdint.h>     /* fixed-width types: you control the bit width exactly */
#include <math.h>

#define ADC_BITS 12
#define ADC_MAX  ((1u << ADC_BITS) - 1)    /* 4095 for a 12-bit ADC */
#define VREF     3.3                        /* reference voltage [V] */
#define THRESH_V 2.0                        /* act when the reading climbs past 2.0 V */
#define ALPHA    0.2                        /* filter strength (0..1): smaller = smoother */

/* READ the sensor. On real hardware this dereferences a memory-mapped register, which
 * MUST be 'volatile' so the compiler always re-reads it (the hardware changes it):
 *     volatile uint16_t *ADC_DATA = (uint16_t *)0x40012400;
 *     return *ADC_DATA;
 * Here we fake a noisy signal rising 1 V -> 3 V. */
static uint16_t read_adc(int n) {
    double true_v = 1.0 + 2.0 * (1.0 - exp(-n / 30.0));
    /* unsigned arithmetic: overflow is defined wraparound (signed overflow is UB in C) */
    uint32_t prng = (uint32_t)n * 1103515245u + 12345u;
    double noise = 0.06 * ((prng % 1000u) / 1000.0 - 0.5);
    long counts = (long)((true_v + noise) / VREF * ADC_MAX + 0.5);
    if (counts < 0) counts = 0;
    if (counts > (long)ADC_MAX) counts = ADC_MAX;
    return (uint16_t)counts;
}

static double counts_to_volts(uint16_t c) { return (double)c * VREF / ADC_MAX; }

int main(void) {
    double filtered = 0.0;          /* filter state -- STATIC, no malloc on an MCU */
    int alarm = 0;
    printf("  n   raw[cts]  raw[V]  filt[V]  state\n");
    for (int n = 0; n < 60; n++) {                 /* the control loop */
        uint16_t raw = read_adc(n);                /* READ  */
        double v = counts_to_volts(raw);           /* SCALE counts -> volts */
        filtered += ALPHA * (v - filtered);        /* FILTER: 1-pole EMA = discrete RC */
        alarm = (filtered > THRESH_V);             /* DECIDE */
        /* ACT: on hardware, WRITE a control register / GPIO here, e.g.
         *     volatile uint32_t *GPIO_OUT = (uint32_t *)0x48000014;
         *     *GPIO_OUT = alarm ? LED_PIN : 0; */
        if (n % 6 == 0)
            printf("  %2d   %5u    %.3f   %.3f   %s\n",
                   n, raw, v, filtered, alarm ? "ALARM (act!)" : "ok");
    }
    printf("\nfinal: filtered = %.3f V, alarm = %d\n", filtered, alarm);
    return 0;
}
