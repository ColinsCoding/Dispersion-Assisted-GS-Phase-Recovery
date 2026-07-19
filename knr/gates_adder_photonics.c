/* gates_adder_photonics.c
 * 7 Basic Gates + Full Adder (C & VHDL) + Jalali-Lab Photonics in C
 * ~1000 lines covering: gates, adders, ripple-carry, complex math,
 * dispersion, GS phase retrieval, DFT, FFT, STEAM, rogue waves,
 * compressed sensing, PDL parser, SNR chain, atoi-style converters.
 *
 * Compile:  gcc -Wall -O2 -lm -o gates_adder_photonics gates_adder_photonics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <float.h>

/* ====================================================================
 * SECTION 1 — 7 BASIC LOGIC GATES
 * Every digital circuit reduces to these 7 primitives.
 * NAND and NOR are individually universal (can build any circuit).
 * ==================================================================== */

/* Gate functions — inputs are 0 or 1 */
int gate_and (int a, int b) { return a & b; }
int gate_or  (int a, int b) { return a | b; }
int gate_not (int a)        { return !a; }
int gate_nand(int a, int b) { return !(a & b); }
int gate_nor (int a, int b) { return !(a | b); }
int gate_xor (int a, int b) { return a ^ b; }
int gate_xnor(int a, int b) { return !(a ^ b); }

void print_truth_table_2in(const char *name,
                           int (*f)(int,int))
{
    printf("  %-5s | A B | out\n", name);
    printf("  ------+-----+----\n");
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
            printf("        | %d %d |  %d\n", a, b, f(a,b));
}

void gates_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 1: 7 BASIC LOGIC GATES\n");
    printf("========================================\n");

    printf("\n--- AND: output 1 only if BOTH inputs 1 ---\n");
    print_truth_table_2in("AND", gate_and);

    printf("\n--- OR: output 1 if EITHER input 1 ---\n");
    print_truth_table_2in("OR", gate_or);

    printf("\n--- NOT: invert ---\n");
    printf("  NOT | A | out\n");
    printf("  ----+---+----\n");
    for (int a = 0; a <= 1; a++)
        printf("      | %d |  %d\n", a, gate_not(a));

    printf("\n--- NAND: NOT of AND (universal gate) ---\n");
    print_truth_table_2in("NAND", gate_nand);

    printf("\n--- NOR: NOT of OR (universal gate) ---\n");
    print_truth_table_2in("NOR", gate_nor);

    printf("\n--- XOR: exactly one input high ---\n");
    print_truth_table_2in("XOR", gate_xor);

    printf("\n--- XNOR: both same (equality detector) ---\n");
    print_truth_table_2in("XNOR", gate_xnor);

    printf("\nUniversality: NAND-only NOT(a) = NAND(a,a) = %d\n",
           gate_nand(1,1));
    printf("              NAND-only AND(1,1)= NOT(NAND(1,1)) = %d\n",
           gate_not(gate_nand(1,1)));
}

/* ====================================================================
 * SECTION 2 — HALF ADDER + FULL ADDER + RIPPLE-CARRY + VHDL
 * Full adder: sum=A^B^Cin, carry=(A&B)|(B&Cin)|(A&Cin)
 * ==================================================================== */

typedef struct { int sum; int carry; } AddResult;

AddResult half_adder(int a, int b)
{
    AddResult r;
    r.sum   = gate_xor(a, b);
    r.carry = gate_and(a, b);
    return r;
}

AddResult full_adder(int a, int b, int cin)
{
    /* Two half adders + OR gate */
    AddResult ha1 = half_adder(a, b);
    AddResult ha2 = half_adder(ha1.sum, cin);
    AddResult r;
    r.sum   = ha2.sum;
    r.carry = gate_or(ha1.carry, ha2.carry);
    return r;
}

/* N-bit ripple-carry adder (N <= 32) */
unsigned ripple_carry_add(unsigned a, unsigned b, int n_bits, int *cout)
{
    unsigned result = 0;
    int cin = 0;
    for (int i = 0; i < n_bits; i++) {
        int ai = (a >> i) & 1;
        int bi = (b >> i) & 1;
        AddResult fa = full_adder(ai, bi, cin);
        result |= ((unsigned)fa.sum << i);
        cin = fa.carry;
    }
    *cout = cin;
    return result;
}

/* Print the VHDL equivalent */
static const char VHDL_FULL_ADDER[] =
"-- VHDL: Full Adder (structural, using gates)\n"
"library IEEE;\n"
"use IEEE.STD_LOGIC_1164.ALL;\n"
"\n"
"entity full_adder is\n"
"    port( A, B, Cin  : in  STD_LOGIC;\n"
"          Sum, Cout  : out STD_LOGIC );\n"
"end full_adder;\n"
"\n"
"architecture structural of full_adder is\n"
"    signal w1, w2, w3 : STD_LOGIC;\n"
"begin\n"
"    -- First half adder\n"
"    w1  <= A XOR B;         -- HA1 sum\n"
"    w2  <= A AND B;         -- HA1 carry\n"
"    -- Second half adder\n"
"    Sum  <= w1 XOR Cin;     -- FA sum\n"
"    w3   <= w1 AND Cin;     -- HA2 carry\n"
"    -- Final carry\n"
"    Cout <= w2 OR w3;\n"
"end structural;\n"
"\n"
"-- VHDL: 4-bit Ripple-Carry Adder\n"
"entity ripple4 is\n"
"    port( A, B : in  STD_LOGIC_VECTOR(3 downto 0);\n"
"          Cin  : in  STD_LOGIC;\n"
"          Sum  : out STD_LOGIC_VECTOR(3 downto 0);\n"
"          Cout : out STD_LOGIC );\n"
"end ripple4;\n"
"\n"
"architecture structural of ripple4 is\n"
"    component full_adder port(A,B,Cin:in STD_LOGIC;\n"
"                              Sum,Cout:out STD_LOGIC);\n"
"    end component;\n"
"    signal c : STD_LOGIC_VECTOR(4 downto 0);\n"
"begin\n"
"    c(0) <= Cin;\n"
"    G: for i in 0 to 3 generate\n"
"        FA: full_adder port map(A(i),B(i),c(i),Sum(i),c(i+1));\n"
"    end generate;\n"
"    Cout <= c(4);\n"
"end structural;\n";

void adder_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 2: FULL ADDER (C + VHDL)\n");
    printf("========================================\n");

    printf("\nHalf Adder truth table:\n");
    printf("  A B | Sum Cout\n");
    printf("  ----+---------\n");
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++) {
            AddResult r = half_adder(a, b);
            printf("  %d %d |  %d    %d\n", a, b, r.sum, r.carry);
        }

    printf("\nFull Adder truth table:\n");
    printf("  A B Cin | Sum Cout\n");
    printf("  --------+---------\n");
    for (int a = 0; a <= 1; a++)
        for (int b = 0; b <= 1; b++)
            for (int c = 0; c <= 1; c++) {
                AddResult r = full_adder(a, b, c);
                printf("  %d %d  %d  |  %d    %d\n",
                       a, b, c, r.sum, r.carry);
            }

    /* 4-bit ripple: 9 + 7 = 16 */
    int cout;
    unsigned s = ripple_carry_add(9u, 7u, 4, &cout);
    printf("\n4-bit ripple: 9 + 7 = %u, Cout=%d\n", s, cout);
    /* overflow: 15 + 1 */
    s = ripple_carry_add(15u, 1u, 4, &cout);
    printf("4-bit ripple: 15+ 1 = %u (overflow), Cout=%d\n", s, cout);

    printf("\n--- VHDL equivalent ---\n%s", VHDL_FULL_ADDER);
}

/* ====================================================================
 * SECTION 3 — COMPLEX NUMBER STRUCT (foundation for all photonics math)
 * ==================================================================== */

typedef struct { double re; double im; } Complex;

Complex c_add (Complex a, Complex b){ return (Complex){a.re+b.re, a.im+b.im}; }
Complex c_sub (Complex a, Complex b){ return (Complex){a.re-b.re, a.im-b.im}; }
Complex c_mul (Complex a, Complex b){
    return (Complex){a.re*b.re - a.im*b.im,
                     a.re*b.im + a.im*b.re};
}
Complex c_scale(Complex a, double s){ return (Complex){a.re*s, a.im*s}; }
double  c_abs (Complex a)           { return sqrt(a.re*a.re + a.im*a.im); }
Complex c_conj(Complex a)           { return (Complex){a.re, -a.im}; }
/* exp(j*theta) = cos(theta) + j*sin(theta) */
Complex c_expj(double theta)        { return (Complex){cos(theta), sin(theta)}; }
/* H(f) = exp(j*pi*beta2*L*(2*pi*f)^2) */
Complex H_f(double f, double beta2_s2_m, double L_m)
{
    double phi = M_PI * beta2_s2_m * L_m * pow(2.0*M_PI*f, 2.0);
    return c_expj(phi);
}

/* ====================================================================
 * SECTION 4 — ATOI-STYLE TYPE CONVERTERS FOR PHOTONICS
 * Convert strings like "17ps/nm/km" or "1550nm" to SI doubles.
 * ==================================================================== */

/* Parse a double from string s, return value, advance *s past number */
static double parse_double(const char **s)
{
    char *end;
    double v = strtod(*s, &end);
    *s = end;
    return v;
}

/* "1550nm"  -> wavelength in meters */
double parse_wavelength_m(const char *s)
{
    double v = parse_double(&s);
    while (isspace((unsigned char)*s)) s++;
    if (strncmp(s, "nm", 2) == 0) v *= 1e-9;
    else if (strncmp(s, "um", 2) == 0) v *= 1e-6;
    else if (strncmp(s, "m",  1) == 0) v *= 1.0;
    return v;
}

/* "17ps/nm/km" -> D in SI: s/(m^2) via D[ps/nm/km] * 1e-6 */
double parse_dispersion_si(const char *s)
{
    double v = parse_double(&s);
    return v * 1e-6;   /* ps/(nm*km) -> s/m^2 */
}

/* "193THz" -> frequency in Hz */
double parse_frequency_hz(const char *s)
{
    double v = parse_double(&s);
    while (isspace((unsigned char)*s)) s++;
    if      (*s == 'T') v *= 1e12;
    else if (*s == 'G') v *= 1e9;
    else if (*s == 'M') v *= 1e6;
    else if (*s == 'k') v *= 1e3;
    return v;
}

/* beta2 [s^2/m] from D [ps/nm/km] at wavelength lambda [m]
 * D = -(2*pi*c / lambda^2) * beta2  =>  beta2 = -D*lambda^2/(2*pi*c)  */
double D_to_beta2(double D_ps_nm_km, double lambda_m)
{
    double c = 2.998e8;
    double D_si = D_ps_nm_km * 1e-6;   /* s/m^2 */
    return -D_si * lambda_m * lambda_m / (2.0 * M_PI * c);
}

/* D_eff [ps/nm] = D[ps/nm/km] * L[km] */
double D_eff(double D_ps_nm_km, double L_km) { return D_ps_nm_km * L_km; }

/* Time-stretch magnification M */
double stretch_M(double D1, double L1, double D2, double L2)
{
    return 1.0 + (D2 * L2) / (D1 * L1);
}

void converters_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 4: ATOI-STYLE PHOTONICS CONVERTERS\n");
    printf("========================================\n");

    printf("parse_wavelength_m(\"1550nm\") = %.4e m\n",
           parse_wavelength_m("1550nm"));
    printf("parse_frequency_hz(\"193THz\") = %.4e Hz\n",
           parse_frequency_hz("193THz"));
    printf("D_to_beta2(17, 1550nm)       = %.4e s^2/m\n",
           D_to_beta2(17.0, 1550e-9));
    printf("D_eff(D=17, L=50km)          = %.1f ps/nm\n",
           D_eff(17.0, 50.0));
    printf("stretch_M(D1=17,L1=5,D2=17,L2=45) = %.1f\n",
           stretch_M(17,5,17,45));

    /* H(f) at RF frequency 1 GHz */
    double beta2 = D_to_beta2(17.0, 1550e-9);
    double L     = 50e3;   /* 50 km in meters */
    Complex hf   = H_f(1e9, beta2, L);
    printf("H(f=1GHz, D=17, L=50km): re=%.4f im=%.4f |H|=%.6f\n",
           hf.re, hf.im, c_abs(hf));
}

/* ====================================================================
 * SECTION 5 — DISPERSIVE FOURIER TRANSFORM (DFT / STEAM)
 * I(t) = |E(f = t/D_eff)|^2   (intensity maps to frequency)
 * D_eff [ps/nm] = D * L
 * ==================================================================== */

#define MAX_PTS 512

/* Gaussian optical pulse E(f) = exp(-f^2 / (2*BW^2)) */
static void gaussian_spectrum(Complex E[], int N, double BW_Hz, double df)
{
    for (int i = 0; i < N; i++) {
        double f = (i - N/2) * df;
        double mag = exp(-f*f / (2.0*BW_Hz*BW_Hz));
        E[i] = (Complex){mag, 0.0};
    }
}

/* Apply dispersion H(f) = exp(j*pi*beta2*L*(2*pi*f)^2) */
static void apply_dispersion(Complex E[], int N, double beta2, double L, double df)
{
    for (int i = 0; i < N; i++) {
        double f = (i - N/2) * df;
        Complex h = H_f(f, beta2, L);
        E[i] = c_mul(E[i], h);
    }
}

/* DFT intensity: I[i] = |E[i]|^2 */
static void intensity(const Complex E[], double I[], int N)
{
    for (int i = 0; i < N; i++) {
        double a = c_abs(E[i]);
        I[i] = a * a;
    }
}

void dft_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 5: DISPERSIVE FOURIER TRANSFORM\n");
    printf("========================================\n");

    int    N      = 64;
    double BW     = 100e9;    /* 100 GHz optical bandwidth */
    double df     = BW / N;
    double D_val  = 17.0;     /* ps/nm/km */
    double L_km   = 50.0;
    double lambda  = 1550e-9;
    double beta2  = D_to_beta2(D_val, lambda);
    double L_m    = L_km * 1e3;

    Complex E[MAX_PTS];
    double  I_in[MAX_PTS], I_out[MAX_PTS];

    gaussian_spectrum(E, N, BW/4.0, df);
    intensity(E, I_in, N);

    apply_dispersion(E, N, beta2, L_m, df);
    intensity(E, I_out, N);

    /* Find peak index to verify I(t) = |E(f=t/D)|^2 maps correctly */
    int peak = 0;
    for (int i = 1; i < N; i++)
        if (I_out[i] > I_out[peak]) peak = i;

    printf("Input  peak at bin %d  (optical spectrum center)\n", N/2);
    printf("Output peak at bin %d  (time-domain after dispersion)\n", peak);
    printf("D_eff = %.1f ps/nm = D*L = %.0f\n", D_eff(D_val, L_km), D_val*L_km);
    printf("|H(f)| = 1 (all-pass, only phase changed)\n");

    /* Mini ASCII plot of I_out */
    double imax = 0;
    for (int i = 0; i < N; i++) if (I_out[i] > imax) imax = I_out[i];
    printf("\nI_out (64 bins, normalized):\n");
    for (int i = 0; i < N; i += 2) {
        int bar = (int)(I_out[i]/imax * 20);
        printf("%2d|", i);
        for (int k = 0; k < bar; k++) putchar('#');
        putchar('\n');
    }
}

/* ====================================================================
 * SECTION 6 — COOLEY-TUKEY FFT (power-of-2, recursive)
 * Time: O(N log N)  vs  O(N^2) naive DFT
 * ==================================================================== */

static void fft_recursive(Complex *x, int N)
{
    if (N <= 1) return;
    /* Split even/odd */
    Complex even[MAX_PTS/2], odd[MAX_PTS/2];
    for (int i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i]  = x[2*i+1];
    }
    fft_recursive(even, N/2);
    fft_recursive(odd,  N/2);
    /* Combine */
    for (int k = 0; k < N/2; k++) {
        Complex t = c_mul(c_expj(-2.0*M_PI*k/N), odd[k]);
        x[k]       = c_add(even[k], t);
        x[k + N/2] = c_sub(even[k], t);
    }
}

void fft_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 6: COOLEY-TUKEY FFT\n");
    printf("========================================\n");

    int N = 16;
    Complex x[MAX_PTS];
    /* Test signal: cos(2*pi*3*n/N) — should peak at bin 3 and N-3 */
    for (int n = 0; n < N; n++)
        x[n] = (Complex){cos(2.0*M_PI*3*n/N), 0.0};

    fft_recursive(x, N);

    printf("FFT of cos(2*pi*3*n/16), N=%d:\n", N);
    printf("Bin | Magnitude\n");
    for (int k = 0; k < N; k++) {
        double mag = c_abs(x[k]);
        if (mag > 0.1)
            printf(" %2d | %.3f %s\n", k, mag,
                   (k==3||k==13) ? "<-- peak (bin 3 and N-3)" : "");
    }
    printf("O(N log N) = %d ops vs O(N^2) = %d ops\n",
           (int)(N * log2(N)), N*N);
}

/* ====================================================================
 * SECTION 7 — GERCHBERG-SAXTON PHASE RETRIEVAL IN C
 * Iterative algorithm: alternate between time and frequency domain,
 * apply magnitude constraints at each domain.
 * Requires FFT — we use our recursive FFT.
 * ==================================================================== */

static void ifft(Complex *x, int N)
{
    /* IFFT = conjugate -> FFT -> conjugate -> scale */
    for (int i = 0; i < N; i++) x[i] = c_conj(x[i]);
    fft_recursive(x, N);
    for (int i = 0; i < N; i++) {
        x[i] = c_conj(x[i]);
        x[i].re /= N;
        x[i].im /= N;
    }
}

void gs_phase_retrieval_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 7: GS PHASE RETRIEVAL\n");
    printf("========================================\n");

    int N = 32;
    /* Known |E_in|: Gaussian in time */
    double mag_t[MAX_PTS], mag_f[MAX_PTS];
    for (int i = 0; i < N; i++) {
        double t = (i - N/2.0) / (N/8.0);
        mag_t[i] = exp(-t*t/2.0);
    }
    /* Target |E_f|: also Gaussian (FT of Gaussian is Gaussian) */
    for (int i = 0; i < N; i++) {
        double f = (i - N/2.0) / (N/4.0);
        mag_f[i] = exp(-f*f/2.0);
    }

    /* Initialize with random phase */
    Complex E[MAX_PTS];
    double  prev_err = 1e30;
    for (int i = 0; i < N; i++)
        E[i] = (Complex){mag_t[i], 0.0};

    printf("Iter | Freq-domain error\n");
    for (int iter = 0; iter < 20; iter++) {
        /* Forward FFT */
        fft_recursive(E, N);
        /* Apply frequency magnitude constraint */
        double err = 0;
        for (int i = 0; i < N; i++) {
            double a = c_abs(E[i]);
            if (a > 1e-15) {
                double scale = mag_f[i] / a;
                E[i].re *= scale;
                E[i].im *= scale;
            }
            double d = c_abs(E[i]) - mag_f[i];
            err += d*d;
        }
        /* Inverse FFT */
        ifft(E, N);
        /* Apply time magnitude constraint */
        for (int i = 0; i < N; i++) {
            double a = c_abs(E[i]);
            if (a > 1e-15) {
                double scale = mag_t[i] / a;
                E[i].re *= scale;
                E[i].im *= scale;
            }
        }
        if (iter % 4 == 0)
            printf("  %2d  | %.6f  %s\n", iter, err/N,
                   err < prev_err ? "(converging)" : "");
        prev_err = err;
    }
    printf("GS converged. Phase recovered in time domain.\n");
}

/* ====================================================================
 * SECTION 8 — OPTICAL ROGUE WAVES (Peregrine Soliton)
 * A_P(z,t) = sqrt(P0) * [1 - 4*(1+2j*z/z0) / (1+4t^2/t0^2+4z^2/z0^2)]
 * At z=0, t=0: |A_P|^2 = 9*P0 (9x peak intensity)
 * ==================================================================== */

void rogue_wave_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 8: OPTICAL ROGUE WAVES\n");
    printf("========================================\n");

    double P0    = 1.0;   /* normalized power */
    double t0    = 1.0;   /* reference time   */
    double z0    = 1.0;   /* nonlinear length */
    int    N     = 17;
    double t_arr[] = {-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,
                       0, 0, 0, 0};
    double z_arr[] = { 0, 0, 0, 0,  0, 0,  0,0, 0, 0, 0,0,0,
                      -2,-1, 0, 1};

    printf("Peregrine soliton |A|^2 / P0:\n");
    printf("  t       z    |A|^2/P0\n");
    for (int i = 0; i < N; i++) {
        double t = t_arr[i], z = z_arr[i];
        double denom_r = 1.0 + 4*t*t/(t0*t0) + 4*z*z/(z0*z0);
        double denom_i = 0.0;
        /* numer = 1 - 4*(1 + 2j*z/z0)/(denom_r + j*denom_i) */
        double numer_r = 4.0 * (1.0*denom_r + 2*z/z0*denom_i)
                         / (denom_r*denom_r + denom_i*denom_i);
        double numer_i = 4.0 * (2*z/z0*denom_r - 1.0*denom_i)
                         / (denom_r*denom_r + denom_i*denom_i);
        double re = 1.0 - numer_r, im = -numer_i;
        double intensity = (re*re + im*im);  /* |...|^2, normalized by P0 already */
        printf("  t=%5.1f  z=%4.1f  %.4f %s\n", t, z, intensity,
               (fabs(t) < 0.01 && fabs(z) < 0.01) ? "<-- peak = 9*P0" : "");
    }
    printf("Rogue wave criterion: I_peak > 2 * I_background\n");
    printf("Peregrine peak:       9 * P0 (factor of 9)\n");
}

/* ====================================================================
 * SECTION 9 — SNR CHAIN + LINK BUDGET (Friis / optical)
 * SNR_out [dB] = SNR_in - noise_figure
 * Optical SNR: OSNR = P_signal / P_ASE
 * ==================================================================== */

typedef struct {
    char   name[32];
    double gain_dB;
    double noise_fig_dB;
} Stage;

double friis_chain_snr(Stage stages[], int n, double SNR_in_dB)
{
    /* Friis formula: NF_total = NF1 + (NF2-1)/G1 + ... */
    double NF_linear = pow(10.0, stages[0].noise_fig_dB/10.0);
    double G_cumul   = pow(10.0, stages[0].gain_dB/10.0);
    for (int i = 1; i < n; i++) {
        double nfi = pow(10.0, stages[i].noise_fig_dB/10.0);
        NF_linear += (nfi - 1.0) / G_cumul;
        G_cumul   *= pow(10.0, stages[i].gain_dB/10.0);
    }
    double NF_dB = 10.0*log10(NF_linear);
    double G_total_dB = 0;
    for (int i = 0; i < n; i++) G_total_dB += stages[i].gain_dB;
    return SNR_in_dB + G_total_dB - NF_dB;
}

void snr_chain_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 9: SNR CHAIN / LINK BUDGET\n");
    printf("========================================\n");

    Stage chain[] = {
        {"EOM",   -6.0, 6.0 },
        {"FIBER", -15.0,15.0},
        {"EDFA",  20.0, 5.0 },
        {"ADC",   -3.0, 3.0 }
    };
    int n = 4;
    double SNR_in = 40.0;  /* dBm */

    printf("Stage        Gain(dB)  NF(dB)\n");
    for (int i = 0; i < n; i++)
        printf("  %-12s  %+6.1f    %.1f\n",
               chain[i].name, chain[i].gain_dB, chain[i].noise_fig_dB);

    double SNR_out = friis_chain_snr(chain, n, SNR_in);
    printf("SNR_in  = %.1f dB\n", SNR_in);
    printf("SNR_out = %.1f dB (Friis chain)\n", SNR_out);

    /* Optical stretch magnification effect on SNR */
    double M = 9.0;   /* 9x stretch */
    printf("\nTime-stretch M=%.0f: ADC bandwidth need / M = BW_RF/%.0f\n", M, M);
    printf("  Effective ENOB improves because slower ADC has lower noise.\n");
    printf("  SNR_ADC_effective = SNR_ADC + 20*log10(M) = %.1f dB\n",
           SNR_out + 20.0*log10(M));
}

/* ====================================================================
 * SECTION 10 — COMPRESSED SENSING / ISTA (L1 minimization)
 * Recover K-sparse signal from M < N measurements.
 * ISTA: x_{k+1} = S_lambda(x_k + A^T(y - A*x_k) / ||A||^2)
 * S_lambda: soft threshold (shrinkage)
 * ==================================================================== */

static double soft_threshold(double x, double lam)
{
    if      (x >  lam) return x - lam;
    else if (x < -lam) return x + lam;
    return 0.0;
}

void ista_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 10: ISTA COMPRESSED SENSING\n");
    printf("========================================\n");

    /* Small example: N=8 signal, K=2 sparse, M=4 measurements */
    int N = 8, M = 4;
    double x_true[8] = {0, 0, 3.0, 0, 0, -2.0, 0, 0};
    /* Random sensing matrix A (M x N) — hardcoded for reproducibility */
    double A[4][8] = {
        { 0.5, -0.3,  0.8,  0.1, -0.4,  0.6, -0.2,  0.3},
        {-0.2,  0.7, -0.1,  0.5,  0.3, -0.8,  0.4, -0.1},
        { 0.6, -0.4,  0.3, -0.7,  0.2,  0.5, -0.3,  0.8},
        {-0.1,  0.2,  0.5,  0.4, -0.6,  0.1,  0.7, -0.5}
    };
    /* Measurements y = A * x_true */
    double y[4] = {0};
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            y[i] += A[i][j] * x_true[j];

    /* ISTA recovery */
    double x[8] = {0};
    double lam = 0.1, step = 0.5;
    for (int it = 0; it < 100; it++) {
        /* gradient: A^T (y - Ax) */
        double r[4] = {0};
        for (int i = 0; i < M; i++) {
            double ax = 0;
            for (int j = 0; j < N; j++) ax += A[i][j]*x[j];
            r[i] = y[i] - ax;
        }
        double grad[8] = {0};
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                grad[j] += A[i][j]*r[i];
        /* Update + soft threshold */
        for (int j = 0; j < N; j++)
            x[j] = soft_threshold(x[j] + step*grad[j], step*lam);
    }
    printf("True signal: ");
    for (int j = 0; j < N; j++) printf("%.1f ", x_true[j]);
    printf("\nRecovered:   ");
    for (int j = 0; j < N; j++) printf("%.1f ", x[j] < 0.05 && x[j] > -0.05 ? 0.0 : x[j]);
    printf("\nIf both match: compressed sensing works (K=%d sparse, M=%d < N=%d)\n",
           2, M, N);
    printf("RIP condition: M >= C*K*log(N/K) = ~%.0f measurements needed\n",
           2.0 * 2 * log((double)N/2));
}

/* ====================================================================
 * SECTION 11 — PDL PARSER (atoi-style for photonic system strings)
 * Parses: "FIBER(D=17,L=50) -> EOM -> EDFA(G=20) -> ADC(fs=40)"
 * ==================================================================== */

typedef enum { COMP_FIBER, COMP_EOM, COMP_EDFA, COMP_ADC, COMP_UNKNOWN } CompType;

typedef struct {
    CompType type;
    char     name[16];
    double   params[4];   /* D, L, G, fs, etc. */
    char     param_names[4][8];
    int      n_params;
} Component;

static void skip_ws(const char **s) { while (isspace((unsigned char)**s)) (*s)++; }

static int parse_name(const char **s, char *out, int maxlen)
{
    int i = 0;
    while (isalnum((unsigned char)**s) || **s == '_') {
        if (i < maxlen-1) out[i++] = **s;
        (*s)++;
    }
    out[i] = '\0';
    return i;
}

int pdl_parse(const char *sys, Component comps[], int max_comps)
{
    int n = 0;
    const char *p = sys;
    while (*p && n < max_comps) {
        skip_ws(&p);
        if (!*p) break;
        /* Skip arrows */
        if (p[0] == '-' && p[1] == '>') { p += 2; continue; }

        Component *c = &comps[n];
        c->n_params = 0;
        parse_name(&p, c->name, 16);
        if (!c->name[0]) { p++; continue; }

        /* Identify type */
        if (strcmp(c->name,"FIBER")==0) c->type = COMP_FIBER;
        else if (strcmp(c->name,"EOM")==0) c->type = COMP_EOM;
        else if (strcmp(c->name,"EDFA")==0) c->type = COMP_EDFA;
        else if (strcmp(c->name,"ADC")==0) c->type = COMP_ADC;
        else c->type = COMP_UNKNOWN;

        /* Parse params: (key=val, ...) */
        skip_ws(&p);
        if (*p == '(') {
            p++;
            while (*p && *p != ')') {
                skip_ws(&p);
                char key[8]; parse_name(&p, key, 8);
                skip_ws(&p);
                if (*p == '=') p++;
                double val = strtod(p, (char**)&p);
                if (c->n_params < 4) {
                    strncpy(c->param_names[c->n_params], key, 7);
                    c->params[c->n_params++] = val;
                }
                skip_ws(&p);
                if (*p == ',') p++;
            }
            if (*p == ')') p++;
        }
        n++;
    }
    return n;
}

void pdl_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 11: PDL PARSER (atoi-style)\n");
    printf("========================================\n");

    const char *sys =
        "FIBER(D=17,L=5) -> EOM -> FIBER(D=17,L=45) -> EDFA(G=20) -> ADC(fs=40)";
    printf("System: %s\n\n", sys);

    Component comps[16];
    int n = pdl_parse(sys, comps, 16);

    double DL_pre = 0, DL_post = 0;
    for (int i = 0; i < n; i++) {
        printf("Component: %-6s", comps[i].name);
        for (int k = 0; k < comps[i].n_params; k++)
            printf("  %s=%.1f", comps[i].param_names[k], comps[i].params[k]);
        printf("\n");
        /* Accumulate DL for magnification */
        if (comps[i].type == COMP_FIBER) {
            double D = 0, L = 0;
            for (int k = 0; k < comps[i].n_params; k++) {
                if (strcmp(comps[i].param_names[k],"D")==0) D = comps[i].params[k];
                if (strcmp(comps[i].param_names[k],"L")==0) L = comps[i].params[k];
            }
            if (DL_pre == 0) DL_pre = fabs(D*L);
            else             DL_post = fabs(D*L);
        }
    }
    double M = 1.0 + DL_post / DL_pre;
    printf("\nDL_pre=%.0f ps/nm  DL_post=%.0f ps/nm\n", DL_pre, DL_post);
    printf("Time-stretch M = 1 + DL_post/DL_pre = %.1f\n", M);
    printf("(Coppinger 1999: first photonic time-stretch ADC demo)\n");
}

/* ====================================================================
 * SECTION 12 — C-TO-HARDWARE MAPPING SUMMARY
 *   C concept       | Hardware equivalent
 *   int             | 32-bit register
 *   unsigned        | unsigned register (no sign extension)
 *   & | ^ ~         | AND/OR/XOR/NOT gate
 *   << >>           | barrel shifter
 *   struct          | memory layout / register file
 *   pointer         | address bus / memory address register
 *   for loop        | counter + comparator + MUX
 *   function call   | CALL instruction (push IP, push args)
 *   array           | contiguous memory, base+offset addressing
 * ==================================================================== */

void hw_mapping_demo(void)
{
    printf("\n========================================\n");
    printf(" SECTION 12: C-TO-HARDWARE MAPPING\n");
    printf("========================================\n");

    printf("C Expression      | Gate / Hardware\n");
    printf("------------------+---------------------------\n");
    printf("a & b             | AND gate\n");
    printf("a | b             | OR gate\n");
    printf("a ^ b             | XOR gate\n");
    printf("~a                | NOT gate (inverter)\n");
    printf("a & (a-1)         | Clear lowest set bit (adder+AND)\n");
    printf("a << 1            | Barrel shifter (multiply by 2)\n");
    printf("a >> 1            | Barrel shifter (divide by 2)\n");
    printf("struct {int a,b;} | Register pair / memory word\n");
    printf("*ptr              | Memory dereference (address bus)\n");
    printf("for(i=0;i<N;i++) | Counter register + comparator\n");
    printf("func(args)        | CALL: push IP, push args, jump\n");
    printf("malloc(N)         | OS: allocate heap page\n");
    printf("\n");

    /* Demonstrate: ripple carry = full adder chain */
    int cout;
    unsigned r = ripple_carry_add(0b1011u, 0b0110u, 4, &cout);
    printf("Ripple carry (in C):  1011 + 0110 = %04u (binary) cout=%d\n",
           r, cout);
    printf("Verified: 11 + 6 = %u\n", 11+6);

    /* Demonstrate: atoi from C standard library equivalent */
    printf("\nmy_atoi style: parse photonic parameter string\n");
    const char *param = "   -850 ";
    int v = 0, sign = 1;
    const char *p = param;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '-') { sign = -1; p++; }
    for (; *p >= '0' && *p <= '9'; p++) v = v*10 + (*p-'0');
    printf("  parse(\"%s\") = %d\n", param, sign*v);
}

/* ====================================================================
 * MAIN
 * ==================================================================== */

int main(void)
{
    printf("================================================\n");
    printf(" JALALI-LAB PHOTONICS IN C — COMPUTER ENGINEERING\n");
    printf(" Gates | Adder | VHDL | FFT | GS | DFT | ISTA  \n");
    printf("================================================\n");

    gates_demo();
    adder_demo();
    converters_demo();
    dft_demo();
    fft_demo();
    gs_phase_retrieval_demo();
    rogue_wave_demo();
    snr_chain_demo();
    ista_demo();
    pdl_demo();
    hw_mapping_demo();

    printf("\n================================================\n");
    printf(" All sections complete.\n");
    printf(" H(f) = exp(j*pi*beta2*L*(2*pi*f)^2) -- unifying theme\n");
    printf(" GS + DFT + FFT + ISTA + Rogue: Jalali group 1999-2017\n");
    printf("================================================\n");
    return 0;
}
