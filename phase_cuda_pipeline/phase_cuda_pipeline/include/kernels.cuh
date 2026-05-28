#pragma once
#include <cufft.h>

void launch_frequency_grid(double* omega, int n, double df_hz);
void launch_init_random_phase(cufftDoubleComplex* field, const double* amp, int n, unsigned long long seed);
void launch_apply_gvd(cufftDoubleComplex* spec, const double* omega, double phi2, int n);
void launch_enforce_magnitude(cufftDoubleComplex* field, const double* amp, int n);
void launch_apply_support(cufftDoubleComplex* spec, const unsigned char* support, int n);
void launch_scale(cufftDoubleComplex* field, double scale, int n);
void launch_intensity(const cufftDoubleComplex* field, double* out, int n);
void launch_make_support(unsigned char* support, const double* freq_hz, double lo_hz, double hi_hz, int n);
void launch_frequency_hz(double* freq_hz, int n, double df_hz);
