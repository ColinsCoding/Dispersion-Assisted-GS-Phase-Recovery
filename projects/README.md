# Jalali-lab projects

Self-contained optics/photonics projects from the UCLA Jalali lab, ported to
Python and added alongside the dispersion-assisted GS phase-recovery work. Each is
a faithful port of the original MATLAB (kept in a `matlab/` subfolder for
reference) with the numerical bugs fixed and a test or demo.

| Project | What it is | Run with |
|---|---|---|
| [`seals/`](seals/) | **SEALS** — Spectrally Encoded Angular Light Scattering: a dual-grating element maps scattering angle θ → wavelength λ, read out on a spectrometer. Includes Rayleigh–Debye–Gans and Lorenz–Mie scattering. | `py -3.12` (needs scipy) |
| [`optical_hybrid_90deg/`](optical_hybrid_90deg/) | **90° optical hybrid** — the I/Q mixer at the front of a coherent receiver: signal + LO → four quadrature outputs; balanced detection recovers the complex field. | `py -3.13` (numpy only) |

Both connect to the main repo's theme: SEALS is spectral-angular **encoding**, and
the 90° hybrid is the classic coherent front end whose job — recovering the complex
optical field — is exactly what the dispersion-assisted GS receiver does without a
hybrid or a local oscillator.
