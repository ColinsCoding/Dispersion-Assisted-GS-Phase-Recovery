# Review of the original MATLAB code

Files reviewed: `main.m`, `mie-2.m` (function `mie`), `rayleighdebye.m`,
`SEALS.m`, plus the project description, the 2021 final-report slides and
the paper (Adam et al., *Optics Express* 21, 28960, 2013). `listVariables.m`
was named in the task but was not among the files provided, so it was not
reviewed.

**The Mie model was not replaced.** `matlab/physics/mie_scattering.m` uses
the same formulation line for line (Mätzler's routines after Bohren and
Huffman). `matlab/tests/check_against_original.m` runs the original files and
the cleaned ones side by side with the default parameters:

| Quantity | Largest relative difference, cleaned vs original |
|---|---|
| SEALS displacement, SEALS angle | 0, 1.6e-16 |
| Mie cross section, `a_n`, `b_n` | 1.6e-16, 0, 0 |
| Mie `I_p`, `I_s`, phases | below 1e-15 |
| Rayleigh-Debye in `'legacy'` mode | 0 |

The physics was also checked independently of the original code: the Mie
functions reduce to Rayleigh for small spheres, satisfy the optical theorem,
conserve energy, and integrate to the total cross section (see the tests).

Every behavioural change below is opt-out: `'legacy'` mode in MATLAB and
`legacy=True` in Python reproduce the original numbers.

## Bugs

| # | File | Finding | Effect | Fix |
|---|---|---|---|---|
| B1 | `rayleighdebye.m` | The sphere form factor enters to the **first** power: `P_theta = 3(sin u − u cos u)/u³` is the scattering *amplitude*; intensity needs its square. `abs()` hides the negative lobes. | Wrong angular pattern for any particle that is not tiny. Minima are too shallow and the lobes fall off as 1/u² instead of 1/u⁴. | Square the form amplitude. |
| B2 | `rayleighdebye.m` | `k` is defined with `n_background`, but the `k⁴` prefactor is written `(2π/λ)⁴`, the vacuum value. | Intensity too small by `nmed⁴` in a medium (3.1× in water). No effect in air. | Use the medium wavenumber everywhere. |
| B3 | `rayleighdebye.m` | `0/0` at `theta = 0` (`u = 0`). | `NaN` in the forward direction. | Series limit `1 − u²/10`. |
| B4 | `mie.m` | Hard-coded `zeros([1 500])` and `for a = 1:500`. | Fails or silently truncates unless exactly 500 angles are passed. | Vectorised over any number of angles. |
| B5 | `mie.m` + `main.m` | `E_phi` uses `cos(phi)`; for one incident polarization the perpendicular component goes as `sin(phi)` and would vanish at `phi = π`. As coded, `I_p` and `I_s` are the responses to two **different** incident polarizations, each at full irradiance, and `main.m` plots their sum. | `I_tot = I_p + I_s` is exactly **2×** the irradiance for unpolarized light. The Rayleigh-Debye branch uses the unpolarized formula, so the two branches of `main.m` disagree by a factor of 2 even in the Rayleigh limit. | Components kept as they are and named for what they are; `I_unpolarized = (I_par + I_perp)/2` returned explicitly. |
| B6 | `main.m` | The Lorentzian source spectrum is computed on `linspace(c/lam2, c/lam1, 500)` (increasing frequency) and multiplied element by element with data ordered by increasing wavelength. | The spectrum is applied back to front, on a grid that is not the wavelength grid. Nearly invisible with the centred default; wrong for any off-centre source. | Evaluate the lineshape at `c/lambda` of each sample. |
| B7 | `main.m`, Rayleigh branch | `hold on` is issued before `figure(3)`, so it applies to the previous figure; then the dB curve and the linear curve are plotted on the same axes labelled "dB". | Misleading figure. | One quantity per axis, log scale, labelled. |
| B8 | `main.m`, Rayleigh branch | Figure 5 omits the source lineshape that the Mie branch applies. | The two branches are not comparable. | Same processing for both. |
| B9 | file names | `mie-2.m` defines `function mie`; the task lists `main(4).m`, `SEALS(4).m`. MATLAB needs the file name to equal the function name, and neither `-` nor `(` is legal in one. | The code does not run as delivered. | Files named after their functions. |
| B10 | `main.m` | `while default ~= 'd' & default ~= 'c'` compares character arrays element-wise. | Errors or misbehaves on input longer than one character. | No interactive prompts; a parameter struct. |

## Unit inconsistencies

| # | Finding | Fix |
|---|---|---|
| U1 | `SEALS.m` returns the angle in **degrees** (`atand`), while `mie.m` and `rayleighdebye.m` take **radians**; `main.m` converts back and forth and adds `mangle` in degrees. | Radians everywhere inside; degrees only on plot axes. |
| U2 | Custom prompts ask for nanometres (diameter, groove spacing, wavelengths) and metres (distances) in the same dialogue; the default branch sets SI values directly. | SI units in a struct; units documented per field. |
| U3 | "Intensity (dB)" is `10·log10` of an irradiance in W/m² with no reference stated. | Axes read "W/m², log scale". |
| U4 | `c = 3*10^8` instead of 299 792 458 m/s. | Exact value (0.07 % difference). |
| U5 | The cross section is printed with no unit. | Printed in m². |
| U6 | `NA` is used as `asin(NA)`, valid only for a lens in air. | Documented and range-checked. |

## Unclear names and magic numbers

| Original | Meaning | New name |
|---|---|---|
| `d`, `D`, `a`, `P` | groove spacing, grating distance, grating angle, pupil diameter | `groove_spacing`, `grating_distance`, `grating_angle`, `pupil_diameter` |
| `a` (in `mie.m`) | loop index, clashes with the grating angle | removed (vectorised) |
| `mangle` | detector angle | `measurement_angle` |
| `raymie = 'b'` | sentinel for "not chosen yet" | removed |
| `cs` | scattering cross section | `sigma_sca` |
| `T_p`, `T_s` | phase of the field components | `phase_par`, `phase_perp` |
| `1/6` in `SEALS.m` | telescope magnification `M` (paper, Sec. 3) | `magnification` argument |
| `dcorr = -4.2448e-4`, `P = 0.0058` | fit parameters from the paper's Fig. 3 | documented as fitted values |
| `x`, `I_tot`, `x2` in the functions | computed and never used | removed |

## Hard-coded paths

None of the four files contains a file path. The practical path problem was
the folder layout (files with spaces and version suffixes in their names).
The new code finds everything relative to its own location and writes only
to `results/`.

## Modelling approximations to be aware of (not bugs)

- `main.m` evaluates Mie theory at the **centre** wavelength for all 500
  samples, although each sample has its own wavelength. The paper argues this
  is acceptable for a 1.3 % bandwidth. The Python sensor model uses each
  channel's own wavelength; the reproduction plots keep the original choice.
- `SEALS.m` writes `+M` where the paper's Eq. (1) prints `−M`. After the
  subtraction of `y(end)` the code's sign reproduces the paper's Fig. 2
  (5.23 mm at 1580 nm falling to 0 at 1600 nm), so it was kept.
- With the default parameters the map gives +32° at 1582 nm and −37° at
  1598 nm. The paper's Fig. 3 reads about +40° and −27°: same slope and
  span, offset by roughly 8–10°. `P` and `dcorr` are fit parameters, so the
  defaults may come from a different alignment. This was not resolved.
- The paper quotes a wavelength resolvability of 0.48 nm from
  `λ0·d/(2·m·w0)` with `w0` = 3 mm. With the code's groove spacing
  (909 nm) that formula gives 0.24 nm; it gives 0.48 nm if 3 mm is the beam
  *diameter* (`w0` = 1.5 mm). The channel count of 42 used in the sensor
  model is the paper's stated number.
- The paper gives the bead refractive index as 1.39. Polystyrene is usually
  quoted near 1.57–1.59. The default was left as in the code and the paper.
- Negative scattering angles are fine for a sphere (the pattern is even in
  angle), but channels near 0° would in reality see the unscattered beam.
  The sensor model masks |θ| < 5°.
