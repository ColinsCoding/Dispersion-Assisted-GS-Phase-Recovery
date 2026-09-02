/* mie_kernel.cu -- single-sphere Mie scattering, CUDA device code.
 *
 * Implements the standard Bohren & Huffman ("Absorption and Scattering
 * of Light by Small Particles", 1983) BHMIE recursive algorithm for a
 * homogeneous sphere: given the size parameter x = 2*pi*r/lambda and
 * the complex relative refractive index m = n + i*k, compute the Mie
 * scattering coefficients a_n, b_n and sum them into the extinction
 * and scattering efficiencies Qext, Qsca.
 *
 * ONE THREAD PER PARTICLE (per size parameter x[i]): every particle's
 * Mie series is completely independent of every other's, so this is
 * embarrassingly parallel -- the natural way to accelerate a sweep
 * over many particle sizes/wavelengths, which is exactly what real
 * atmospheric-optics / light-scattering codes need (e.g. computing
 * Qext(lambda) across a whole spectrum, or Qext(r) across a particle
 * size distribution).
 *
 * ALGORITHM (per thread, for its own x):
 *   1. Nmax ~ x + 4*x^(1/3) + 2 (Wiscombe's series-truncation rule) --
 *      terms beyond this contribute negligibly to the sum.
 *   2. The logarithmic derivative D_n(mx) = d/d(mx)[ln(psi_n(mx))] is
 *      computed by DOWNWARD recurrence starting well above Nmax (this
 *      direction is numerically stable; upward is not).
 *   3. The Riccati-Bessel functions psi_n(x), chi_n(x) (both real,
 *      since x itself is real) are computed by UPWARD recurrence
 *      starting from psi_{-1}=cos(x), psi_0=sin(x) (stable in this
 *      direction for x real).
 *   4. a_n, b_n follow directly from D_n, psi_n, psi_{n-1}, xi_n,
 *      xi_{n-1} (xi_n = psi_n - i*chi_n).
 *   5. Qext = (2/x^2) * sum (2n+1)*Re(a_n+b_n)
 *      Qsca = (2/x^2) * sum (2n+1)*(|a_n|^2+|b_n|^2)
 *
 * LIMITATION, stated honestly: MAX_N below caps the series length a
 * single thread can hold in local storage. For x up to ~60 (visible-
 * light Mie scattering off particles up to tens of microns), Nmax
 * stays well under MAX_N=300; a much larger x (very large particles)
 * would need a bigger buffer or a different (asymptotic) method.
 */
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "mie.h"

#define MAX_N 300

__device__ void mie_efficiencies(double x, cuDoubleComplex m,
                                  double *Qext_out, double *Qsca_out) {
    if (x <= 0.0) {
        *Qext_out = 0.0;
        *Qsca_out = 0.0;
        return;
    }

    cuDoubleComplex mx = make_cuDoubleComplex(m.x * x, m.y * x);

    int Nmax = (int)(x + 4.0 * cbrt(x) + 2.0) + 1;
    if (Nmax >= MAX_N) {
        Nmax = MAX_N - 1;   /* clamp -- see the documented MAX_N limitation above */
    }

    /* Bohren & Huffman's own recommendation: start the downward D_n
     * recurrence at least ~15 orders above both Nmax and |mx|, since
     * D_n needs a "settling in" distance to become accurate by the
     * time the recurrence reaches the n values actually used. */
    double abs_mx = sqrt(mx.x * mx.x + mx.y * mx.y);
    int Nstart = Nmax;
    if ((int)abs_mx + 15 > Nstart) {
        Nstart = (int)abs_mx + 15;
    }
    if (Nstart >= MAX_N) {
        Nstart = MAX_N - 1;
    }

    /* downward recurrence for the logarithmic derivative D_n(mx) */
    cuDoubleComplex D[MAX_N];
    D[Nstart] = make_cuDoubleComplex(0.0, 0.0);
    for (int n = Nstart; n >= 1; n--) {
        cuDoubleComplex n_over_mx = cuCdiv(make_cuDoubleComplex((double)n, 0.0), mx);
        cuDoubleComplex denom = cuCadd(D[n], n_over_mx);
        cuDoubleComplex inv_denom = cuCdiv(make_cuDoubleComplex(1.0, 0.0), denom);
        D[n - 1] = cuCsub(n_over_mx, inv_denom);
    }

    /* upward recurrence for psi_n(x), chi_n(x) -- both real, x is real */
    double psi_prev2 = cos(x);   /* psi_{-1} */
    double psi_prev1 = sin(x);   /* psi_0    */
    double chi_prev2 = -sin(x);  /* chi_{-1} */
    double chi_prev1 = cos(x);   /* chi_0    */

    double Qext = 0.0;
    double Qsca = 0.0;

    for (int n = 1; n <= Nmax; n++) {
        double psi_n = (2.0 * n - 1.0) / x * psi_prev1 - psi_prev2;
        double chi_n = (2.0 * n - 1.0) / x * chi_prev1 - chi_prev2;

        cuDoubleComplex xi_n = make_cuDoubleComplex(psi_n, -chi_n);
        cuDoubleComplex xi_prev1 = make_cuDoubleComplex(psi_prev1, -chi_prev1);

        cuDoubleComplex Dn = D[n];
        cuDoubleComplex n_over_x = make_cuDoubleComplex((double)n / x, 0.0);

        /* a_n = [(D_n/m + n/x)*psi_n - psi_{n-1}] / [(D_n/m + n/x)*xi_n - xi_{n-1}] */
        cuDoubleComplex Dn_over_m = cuCdiv(Dn, m);
        cuDoubleComplex term_a = cuCadd(Dn_over_m, n_over_x);
        cuDoubleComplex a_num = cuCsub(cuCmul(term_a, make_cuDoubleComplex(psi_n, 0.0)),
                                        make_cuDoubleComplex(psi_prev1, 0.0));
        cuDoubleComplex a_den = cuCsub(cuCmul(term_a, xi_n), xi_prev1);
        cuDoubleComplex a_n = cuCdiv(a_num, a_den);

        /* b_n = [(D_n*m + n/x)*psi_n - psi_{n-1}] / [(D_n*m + n/x)*xi_n - xi_{n-1}] */
        cuDoubleComplex Dn_times_m = cuCmul(Dn, m);
        cuDoubleComplex term_b = cuCadd(Dn_times_m, n_over_x);
        cuDoubleComplex b_num = cuCsub(cuCmul(term_b, make_cuDoubleComplex(psi_n, 0.0)),
                                        make_cuDoubleComplex(psi_prev1, 0.0));
        cuDoubleComplex b_den = cuCsub(cuCmul(term_b, xi_n), xi_prev1);
        cuDoubleComplex b_n = cuCdiv(b_num, b_den);

        double factor = 2.0 * n + 1.0;
        Qext += factor * (a_n.x + b_n.x);   /* Re(a_n + b_n) */
        Qsca += factor * (cuCabs(a_n) * cuCabs(a_n) + cuCabs(b_n) * cuCabs(b_n));

        psi_prev2 = psi_prev1;
        psi_prev1 = psi_n;
        chi_prev2 = chi_prev1;
        chi_prev1 = chi_n;
    }

    double x2 = x * x;
    *Qext_out = (2.0 / x2) * Qext;
    *Qsca_out = (2.0 / x2) * Qsca;
}

__global__ void mie_kernel(const double *x, cuDoubleComplex m,
                            double *Qext, double *Qsca, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        mie_efficiencies(x[i], m, &Qext[i], &Qsca[i]);
    }
}

extern "C" void run_mie_scattering(const double *x, double m_re, double m_im,
                                    double *Qext, double *Qsca, int n_particles) {
    double *d_x = NULL, *d_Qext = NULL, *d_Qsca = NULL;
    size_t bytes = (size_t)n_particles * sizeof(double);

    cudaMalloc((void **)&d_x, bytes);
    cudaMalloc((void **)&d_Qext, bytes);
    cudaMalloc((void **)&d_Qsca, bytes);

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);

    cuDoubleComplex m = make_cuDoubleComplex(m_re, m_im);
    int threads_per_block = 128;
    int blocks = (n_particles + threads_per_block - 1) / threads_per_block;
    mie_kernel<<<blocks, threads_per_block>>>(d_x, m, d_Qext, d_Qsca, n_particles);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "mie_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaMemcpy(Qext, d_Qext, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Qsca, d_Qsca, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_Qext);
    cudaFree(d_Qsca);
}
