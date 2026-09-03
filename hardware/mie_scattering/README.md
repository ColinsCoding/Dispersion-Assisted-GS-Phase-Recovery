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
  20 for **five materials** (water, ice, silica — non-absorbing; soot,
  gold — absorbing, complex refractive index), writes `mie_output.csv`
  (columns: `material,x,Qext,Qsca`), and prints a per-material summary.
- `generate_mie_reference.py` — an independent Python reimplementation
  of the same algorithm (not a library call), used to cross-check the
  CUDA output material by material.
- `mie_sympy_formalization.py` — symbolic (not just numeric) proof of
  two identities the recursion depends on: the Riccati-Bessel
  recurrence itself, and the `m=1` "invisible sphere" limit (a sphere
  optically identical to its medium must scatter exactly nothing).
- `Makefile` — `make build` / `make run` / `make verify` (the full
  pipeline) / `make clean`.

## Build and run

```bash
make verify        # compile, run, and cross-check in one step
```

or manually:

```bash
nvcc mie_kernel.cu mie_main.cu -o mie.exe
./mie.exe
python generate_mie_reference.py
```

(On Windows, `cl.exe` from a Visual Studio install must be on `PATH`
for `nvcc` to find its host compiler; see the Makefile's header
comment for the exact `PATH` addition. `make clean` needs `rm`, so run
it from Git Bash/WSL rather than plain PowerShell/cmd.)

## Materials

| Material | m = n + ik | Absorbing? |
|---|---|---|
| water droplet | 1.33 + 0.00i | no |
| ice crystal | 1.31 + 0.00i | no |
| silica/glass | 1.46 + 0.00i | no |
| soot (black carbon) | 1.85 + 0.71i | yes |
| gold nanosphere | 0.47 + 2.40i | yes (plasmonic) |

Illustrative visible-wavelength (~550 nm) values, not tied to one
specific source.

## Verification

Two independent implementations of the algorithm (CUDA and Python)
agree to a **worst-case maximum relative error of 3.44e-6** across all
5 materials x 200 size parameters each. The underlying Riccati-Bessel
recurrence was separately checked against `scipy.special.spherical_jn`/
`spherical_yn` directly (matching to ~1e-13 to 1e-17), and — going
further than a floating-point check — `mie_sympy_formalization.py`
proves the recurrence and the `m=1` limit as exact symbolic identities
using sympy, not just numeric agreement at a handful of test points.

Physical sanity checks, all confirmed rather than assumed:
- `Qext` rises from near zero (Rayleigh regime, `x << 1`), oscillates
  through the "Mie ripple" resonance region (`x` ~ 1-15), and trends
  toward 2.0 at large `x` — the well-known "extinction paradox" (a
  large sphere blocks *twice* its geometric cross-section, from
  diffraction at the edge).
- For non-absorbing materials, `Qext == Qsca` exactly (no energy lost
  to absorption).
- For absorbing materials (soot, gold), `Qabs = Qext - Qsca` is
  genuinely positive — for soot at small `x`, absorption even exceeds
  scattering, as expected for a strongly-absorbing sub-wavelength
  particle.
- At `m=1` (sphere optically identical to its medium), `Qext` and
  `Qsca` both vanish to floating-point-level zero (~1e-16 to 1e-31) —
  confirmed both numerically and via the symbolic proof above.

## Known limitation

`MAX_N` in `mie_kernel.cu` caps the series length a single thread can
hold (300 terms), which is comfortably enough for size parameters up
to roughly `x ~ 60` (particles up to tens of microns at visible
wavelengths). A much larger `x` would need a bigger buffer.
