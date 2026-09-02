# Single-Sphere Mie Scattering (CUDA)

Computes the Mie extinction and scattering efficiencies (Qext, Qsca) for
a homogeneous sphere, via the standard Bohren & Huffman (BHMIE)
recursive algorithm, parallelized across an array of size parameters
on the GPU.

## Files

- `mie.h` — shared header declaring `run_mie_scattering`, the link
  between the two `.cu` files.
- `mie_kernel.cu` — the CUDA device code: the BHMIE recursion
  (`mie_efficiencies`), the kernel (`mie_kernel`, one thread per
  particle), and the host-callable launcher (`run_mie_scattering`).
- `mie_main.cu` — host driver: sweeps size parameter `x` from 0.1 to
  20 for a water droplet (`m = 1.33`, non-absorbing), writes
  `mie_output.csv`, and prints a summary table.
- `generate_mie_reference.py` — an independent Python reimplementation
  of the same algorithm (not a library call), used to cross-check the
  CUDA output.

## Build and run

```bash
nvcc mie_kernel.cu mie_main.cu -o mie.exe
./mie.exe
python generate_mie_reference.py
```

(On Windows, `cl.exe` from a Visual Studio install must be on `PATH`
for `nvcc` to find its host compiler.)

## Verification

Two independent implementations of the algorithm (CUDA and Python)
agree to a maximum relative error of **2.29e-6** across 200 size
parameters, and the underlying Riccati-Bessel recurrence was
separately checked against `scipy.special.spherical_jn`/`spherical_yn`
directly, matching to machine precision (~1e-13 to 1e-17).

The physics itself checks out too: `Qext` rises from near zero
(Rayleigh regime, `x << 1`), oscillates through the "Mie ripple"
resonance region (`x` ~ 1-15), and trends toward 2.0 at large `x` —
the well-known "extinction paradox" (a large sphere blocks *twice*
its geometric cross-section, because of diffraction).

## Known limitation

`MAX_N` in `mie_kernel.cu` caps the series length a single thread can
hold (300 terms), which is comfortably enough for size parameters up
to roughly `x ~ 60` (particles up to tens of microns at visible
wavelengths). A much larger `x` would need a bigger buffer.
Absorbing spheres (complex `m` with nonzero imaginary part) are
supported by the math but untested here — the driver uses `m_im=0.0`.
