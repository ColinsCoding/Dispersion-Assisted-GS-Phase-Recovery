#ifndef DGS_FORWARD_H
#define DGS_FORWARD_H

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_dispersion_kernel(
    cuFloatComplex* d_x_freq,
    const float* d_beta,
    int N
);

void launch_mag_square(
    const cuFloatComplex* d_x,
    float* d_y,
    int N
);

#ifdef __cplusplus
}
#endif

#endif
