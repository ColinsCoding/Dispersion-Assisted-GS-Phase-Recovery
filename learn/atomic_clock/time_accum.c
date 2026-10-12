/* time_accum.c -- how a clock "loses time" inside a computer.
 *
 * Add a 0.1 s tick 3,600,000 times (100 hours) four ways:
 *   float     (IEEE 754 binary32, ~7 significant digits)
 *   double    (IEEE 754 binary64, ~16 significant digits)
 *   uint64_t  nanosecond counter (exact integer arithmetic)
 *   24-bit fixed point with 0.1 truncated to 23 fractional bits
 *             (the arithmetic behind the 1991 Patriot clock error)
 * and print each total minus the exact 360000 s.
 *
 *   gcc -std=c99 -O2 -Wall -Wextra time_accum.c -o time_accum && ./time_accum
 * Volatile accumulators keep the compiler from folding the loop into one multiply.
 */
#include <stdio.h>
#include <stdint.h>

int main(void)
{
    const long n = 3600000L;                       /* 100 hours of 0.1 s ticks */
    volatile float f = 0.0f;
    volatile double d = 0.0;
    volatile uint64_t ns = 0;
    /* 0.1 truncated to 23 fractional bits, kept as an integer count of 2^-23 s */
    const uint64_t tenth_q23 = (uint64_t)(0.1 * (1 << 23));   /* floor(838860.8) = 838860 */
    volatile uint64_t q23 = 0;

    for (long i = 0; i < n; ++i) {
        f += 0.1f;
        d += 0.1;
        ns += 100000000ULL;
        q23 += tenth_q23;
    }
    const double exact = 360000.0;
    printf("float    total=%.6f s  error=%+.6f s\n", (double)f, (double)f - exact);
    printf("double   total=%.9f s  error=%+.3e s\n", d, d - exact);
    printf("uint64ns total=%.9f s  error=%+.3e s\n", (double)ns * 1e-9, (double)ns * 1e-9 - exact);
    printf("fixed23  total=%.6f s  error=%+.6f s\n", (double)q23 / (1 << 23), (double)q23 / (1 << 23) - exact);
    return 0;
}
