/* K&R Exercise 2-1
 * Ranges of char, short, int, long (signed & unsigned)
 * and floating-point types -- from headers AND by direct computation.
 */
#include <stdio.h>
#include <limits.h>   /* integer limits */
#include <float.h>    /* floating-point limits */

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* INTEGER RANGES FROM HEADERS                                         */
    /* ------------------------------------------------------------------ */
    printf("=== Integer ranges from <limits.h> ===\n");
    printf("signed char    : %d .. %d\n",       SCHAR_MIN, SCHAR_MAX);
    printf("unsigned char  : 0 .. %u\n",         UCHAR_MAX);
    printf("signed short   : %d .. %d\n",        SHRT_MIN,  SHRT_MAX);
    printf("unsigned short : 0 .. %u\n",         USHRT_MAX);
    printf("signed int     : %d .. %d\n",        INT_MIN,   INT_MAX);
    printf("unsigned int   : 0 .. %u\n",         UINT_MAX);
    printf("signed long    : %ld .. %ld\n",      LONG_MIN,  LONG_MAX);
    printf("unsigned long  : 0 .. %lu\n",        ULONG_MAX);

    /* ------------------------------------------------------------------ */
    /* INTEGER RANGES BY DIRECT COMPUTATION                               */
    /*                                                                    */
    /* For an N-bit signed type using two's complement:                   */
    /*   max =  (1 << (N-1)) - 1   but we avoid UB by using unsigned.    */
    /*                                                                    */
    /* Trick: for any unsigned type T, ~(T)0 == UMAX.                    */
    /* For signed type S with same width: SMAX = UMAX >> 1               */
    /*                                   SMIN = -(SMAX) - 1              */
    /* ------------------------------------------------------------------ */
    printf("\n=== Integer ranges by direct computation ===\n");

    /* --- char --- */
    unsigned char uc_max = ~(unsigned char)0;
    signed char   sc_max = (signed char)(uc_max >> 1);
    signed char   sc_min = (signed char)(-sc_max - 1);
    printf("signed char    : %d .. %d\n", (int)sc_min, (int)sc_max);
    printf("unsigned char  : 0 .. %u\n",  (unsigned)uc_max);

    /* --- short --- */
    unsigned short us_max = ~(unsigned short)0;
    short          ss_max = (short)(us_max >> 1);
    short          ss_min = (short)(-ss_max - 1);
    printf("signed short   : %d .. %d\n", (int)ss_min, (int)ss_max);
    printf("unsigned short : 0 .. %u\n",  (unsigned)us_max);

    /* --- int --- */
    unsigned int ui_max = ~0u;
    int          si_max = (int)(ui_max >> 1);
    int          si_min = -si_max - 1;
    printf("signed int     : %d .. %d\n", si_min, si_max);
    printf("unsigned int   : 0 .. %u\n",  ui_max);

    /* --- long --- */
    unsigned long ul_max = ~0ul;
    long          sl_max = (long)(ul_max >> 1);
    long          sl_min = -sl_max - 1L;
    printf("signed long    : %ld .. %ld\n", sl_min, sl_max);
    printf("unsigned long  : 0 .. %lu\n",   ul_max);

    /* ------------------------------------------------------------------ */
    /* FLOATING-POINT RANGES FROM HEADERS                                 */
    /* ------------------------------------------------------------------ */
    printf("\n=== Floating-point ranges from <float.h> ===\n");
    printf("float       max: %e   min positive normal: %e   epsilon: %e\n",
           FLT_MAX, FLT_MIN, FLT_EPSILON);
    printf("double      max: %e   min positive normal: %e   epsilon: %e\n",
           DBL_MAX, DBL_MIN, DBL_EPSILON);
    printf("long double max: %Le   min positive normal: %Le   epsilon: %Le\n",
           LDBL_MAX, LDBL_MIN, LDBL_EPSILON);

    /* ------------------------------------------------------------------ */
    /* FLOATING-POINT RANGES BY DIRECT COMPUTATION                        */
    /*                                                                    */
    /* IEEE 754 single (float): 1 sign, 8 exp, 23 mantissa bits          */
    /*   max normal = (2 - 2^-23) * 2^127                                 */
    /*   epsilon    = 2^-23                                               */
    /*                                                                    */
    /* We reconstruct via repeated doubling until overflow.               */
    /* ------------------------------------------------------------------ */
    printf("\n=== Floating-point ranges by direct computation ===\n");

    /* float epsilon: smallest e such that 1.0f + e != 1.0f */
    float feps = 1.0f;
    while (1.0f + feps / 2.0f != 1.0f)
        feps /= 2.0f;
    printf("float  epsilon (computed): %e\n", feps);

    /* float max: double until next doubling would overflow (go to inf).
     * Detect: if f*2 / 2 != f, then f*2 overflowed to inf.           */
    float fmax = 1.0f;
    while ((fmax * 2.0f) / 2.0f == fmax)
        fmax *= 2.0f;
    /* fmax is now 2^127 (largest finite power of 2).
     * Actual FLT_MAX = 2^127 * (2 - 2^-23) = fmax * (2 - feps).    */
    fmax *= (2.0f - feps);
    printf("float  max     (computed): %e\n", fmax);

    /* double epsilon */
    double deps = 1.0;
    while (1.0 + deps / 2.0 != 1.0)
        deps /= 2.0;
    printf("double epsilon (computed): %e\n", deps);

    /* double max */
    double dmax = 1.0;
    while ((dmax * 2.0) / 2.0 == dmax)
        dmax *= 2.0;
    dmax *= (2.0 - deps);
    printf("double max     (computed): %e\n", dmax);

    return 0;
}
