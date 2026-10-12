# Variables

SI units throughout. Angles are in radians inside the code and in degrees
only on plot axes. "Datatype" is the Python type; MATLAB uses `double` for
all real quantities and complex `double` for fields and amplitudes.

The Python name is given first; the MATLAB name follows in parentheses when
it differs (`lambda` is a reserved word in Python, hence `lam`).

## Particle and medium

| Variable | Meaning | Unit | Datatype | Source equation |
|---|---|---|---|---|
| `dia` | particle diameter | m | float | input (model parameter) |
| `radius` | particle radius | m | float | `radius = dia/2` |
| `lam` (`lambda`) | vacuum wavelength | m | float | input |
| `npar` | particle refractive index | – | float or complex | input |
| `nmed` | medium refractive index | – | float | input |
| `m` | relative refractive index | – | float or complex | `m = npar/nmed` |
| `k` | wavenumber in the medium | rad/m | float | `k = 2π nmed/lam` |
| `x` | size parameter | – | float | `x = π dia/(lam/nmed)` |
| `theta` | scattering angle, 0 = forward | rad | float array | input, or `theta_illum + measurement_angle` |
| `r` | particle-to-detector distance | m | float | input |
| `K` | polarizability factor | – | float or complex | `K = (m²−1)/(m²+2)` |
| `concentration` | number concentration | particles/m³ | float | input / estimated |

## Scattering model outputs

| Variable | Meaning | Unit | Datatype | Source equation |
|---|---|---|---|---|
| `nmax` | number of multipoles | – | int | `round(2 + x + 4 x^(1/3))` |
| `an`, `bn` | Mie coefficients | – | complex array | boundary conditions at the sphere surface (see `mie_coefficients`) |
| `s1`, `s2` (`S1`, `S2`) | scattering amplitudes, perpendicular and parallel | – | complex array | `S1 = Σ (2n+1)/(n(n+1)) (a_n π_n + b_n τ_n)`, `S2` with π and τ swapped |
| `e_par`, `e_perp` | scattered field per unit incident field | – | complex array | `e^{ikr}/(∓ikr) · cos φ · S2, S1` |
| `i_par`, `i_perp` | irradiance for parallel / perpendicular incident light | W/m² | float array | `I0 \|S2\|²/(kr)²`, `I0 \|S1\|²/(kr)²` |
| `i_unpolarized` | irradiance for unpolarized incident light | W/m² | float array | `(i_par + i_perp)/2` |
| `phase_par`, `phase_perp` | wrapped phase of the field components | rad | float array | `angle(e_par)`, `angle(e_perp)` |
| `qsca` | scattering efficiency | – | float | `(2/x²) Σ (2n+1)(\|a_n\|²+\|b_n\|²)` |
| `qext` | extinction efficiency | – | float | `(2/x²) Σ (2n+1) Re(a_n+b_n)` |
| `sigma_sca` | scattering cross section | m² | float | `qsca · π dia²/4` |
| `dsigma` | differential cross section | m²/sr | float array | `(\|S1\|²+\|S2\|²)/(2k²)` |
| `u` | Rayleigh-Debye phase argument | – | float array | `u = 2 k radius sin(theta/2)` |
| `P(theta)` | Rayleigh-Debye form factor | – | float array | `[3(sin u − u cos u)/u³]²` |

## SEALS optics

| Variable | Meaning | Unit | Datatype | Source equation |
|---|---|---|---|---|
| `groove_spacing` (`d`) | grating groove spacing | m | float | input (1100 lines/mm) |
| `grating_distance` (`D`) | distance between the gratings | m | float | input |
| `grating_angle` (`a`, α) | incidence angle on the grating | rad | float | input |
| `magnification` (M) | telescope magnification | – | float | 1/6 (paper, Sec. 3) |
| `beta` | diffracted angle | rad | float array | `asin(lam/d − sin α)` |
| `y` | beam displacement | m | float array | paper Eq. 1: `M·D·tan(α−β)/(1 + tan(α−β) tan α)` |
| `dcorr` | beam-centre correction | m | float | fit parameter |
| `pupil_diameter` (`P`) | objective entrance pupil | m | float | fit parameter |
| `numerical_aperture` (`NA`) | objective NA (in air) | – | float | input |
| `theta_illum` | illumination angle | rad | float array | paper Eq. 2: `atan(2/P (y − y_c + dcorr) tan(asin NA))` |
| `measurement_angle` (`mangle`) | detector angle | rad | float | input |
| `source_bandwidth` (`band`) | source bandwidth | m | float | input |
| `lineshape` | relative source power | – | float array | Lorentzian in frequency, peak 1 |
| `equalization` | filter transmission per channel | – | tuple of float | `design_equalization` |

## Sensor chain

| Variable | Meaning | Unit | Datatype | Source equation |
|---|---|---|---|---|
| `incident_irradiance` (I0) | irradiance at the sample | W/m² | float | input |
| `detector_area`, `detector_distance` | detector geometry | m², m | float | input |
| `solid_angle` | detector solid angle | sr | float | `area / distance²` |
| `probe_volume` | illuminated and viewed volume | m³ | float | input |
| `power` | optical power per channel | W | float array | `N · I0 · L · dσ/dΩ · solid_angle` |
| `responsivity` | photodiode responsivity | A/W | float | input |
| `dark_current` | photodiode dark current | A | float | input |
| `tia_gain` | transimpedance gain | V/A | float | input |
| `gain` | programmable gain | – | int | largest setting below 80 % of full scale |
| `antialias_cutoff` | filter corner | Hz | float | input, `< sampling_rate/2` |
| `noise_bandwidth` | equivalent noise bandwidth | Hz | float | `(π/2) · antialias_cutoff` |
| `adc_bits`, `adc_vref` | ADC resolution and full scale | bit, V | int, float | input |
| `lsb` | ADC step | V | float | `adc_vref / 2^adc_bits` |
| `sampling_rate` | sampling frequency | Hz | float | input |
| `codes` | averaged ADC code per channel | counts | float array | `round(v · gain / lsb)`, averaged |
| `code_sums` | integer sum of the samples | counts | int64 array | hardware accumulator |
| `sigma_volts` | RMS noise of the averaged voltage | V | float array | shot, thermal, amplifier, quantization in quadrature, `/√n` |
| shot noise | photon/electron counting noise | A | float | `sqrt(2 q (i + i_dark) B)` |
| thermal noise | feedback-resistor noise | A | float | `sqrt(4 k_B T B / tia_gain)` |

## Estimator and motion

| Variable | Meaning | Unit | Datatype | Source equation |
|---|---|---|---|---|
| `templates` | predicted voltage per particle, per candidate | V | float array (K, J) | `responsivity · tia_gain · power_per_particle` |
| `n_particles` | fitted particle count | – | float | `Σ w v T / Σ w T²`, `w = 1/σ²` |
| `chi2` | weighted misfit | – | float | `Σ w (v − N T)²` |
| `dia_ci`, `npar_ci` | confidence intervals | m, – | tuple of float | `χ² − χ²_min ≤ 3.84` (95 %) |
| `fit_probability` | goodness of fit | – | float | upper tail of χ² with `dof` degrees of freedom |
| `g2` | intensity autocorrelation | – | float array | `⟨I(t) I(t+τ)⟩/⟨I⟩²` |
| `q` | scattering vector | rad/m | float | `(4π nmed/lam) sin(theta/2)` |
| `decay_rate` (Γ) | autocorrelation decay rate | 1/s | float | `g2 − 1 = β e^{−2Γτ}`, `Γ = D q²` |
| `diffusion` (D) | diffusion coefficient | m²/s | float | `Γ/q²` |
| `hydrodynamic_dia` | diameter from motion | m | float | Stokes-Einstein: `k_B T/(3π η D)` |
| `temperature`, `viscosity` | of the liquid | K, Pa·s | float | input |

## ASIC model and telemetry

| Variable | Meaning | Unit | Datatype | Source equation |
|---|---|---|---|---|
| `templates_q15` | stored templates | – | int16 array | `round(T/\|T\| · 32767)` |
| `meas` | dark-subtracted measurement | counts | int64 array | `code_sums − dark_code_sums` |
| `corr` | correlator outputs | counts | int64 array | `Σ_j meas_j · T_kj` |
| `template_energy` | energy of each stored template | – | int64 array | `Σ_j T_kj²` |
| `noise_sigma_counts` | noise of one element of `meas` | counts | float | from `sigma_volts` |
| `tag_id` | sample-container identifier | – | uint32 | assigned |
| `timestamp_ms` | time of the estimate | ms | uint64 | instrument clock |
| `confidence_level` | level of the interval | – | float | chosen (0.95) |
| `flags` | status bits | – | uint8 | see `telemetry.py` |
| CRC | packet check | – | uint16 | CRC-16/CCITT, polynomial 0x1021 |
| `path_loss_db` | free-space loss | dB | float | `20 log10(4π d f/c)` |
| `bit_error_rate` | probability of a flipped bit | – | float | `½ exp(−Eb/N0 / 2)` (non-coherent FSK) |
