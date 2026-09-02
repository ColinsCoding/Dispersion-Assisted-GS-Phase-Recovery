#ifndef MIE_H
#define MIE_H

/* Shared header between mie_kernel.cu (device Mie physics) and
 * mie_main.cu (host driver). This is what lets two separate .cu
 * translation units call into each other without name-mangling
 * surprises -- mie_kernel.cu defines run_mie_scattering with C
 * linkage, mie_main.cu just declares it here and links against it.
 *
 * Complex refractive index m = n + i*k is passed as two doubles
 * (real/imag) instead of cuDoubleComplex here, so this header has no
 * CUDA-specific types in it and stays includable from plain C++ if
 * ever needed.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the Mie extinction and scattering efficiencies Qext, Qsca
 * for n_particles independent size parameters x[i] = 2*pi*r_i/lambda,
 * all sharing the same complex relative refractive index m = m_re +
 * i*m_im. One CUDA thread handles one size parameter -- embarrassingly
 * parallel, since each particle's Mie series is independent of every
 * other's.
 *
 * x, Qext, Qsca: host arrays of length n_particles (Qext/Qsca are
 * output-only, caller allocates).
 */
void run_mie_scattering(const double *x, double m_re, double m_im,
                         double *Qext, double *Qsca, int n_particles);

#ifdef __cplusplus
}
#endif

#endif /* MIE_H */
