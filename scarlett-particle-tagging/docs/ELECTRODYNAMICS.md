# Electrodynamics behind the model

The model is **classical**: Maxwell's equations, a plane wave, a homogeneous
sphere. This page connects the quantities the code computes. Symbols match
`docs/VARIABLES.md`.

## 1. Electric and magnetic field of the incident wave

A monochromatic plane wave travelling along `z` in a medium of refractive
index `nmed`:

$$
\mathbf E(z,t)=\mathrm{Re}\!\left[\mathbf E_0\,e^{i(kz-\omega t)}\right],\qquad
\mathbf B=\frac{1}{\omega}\,\mathbf k\times\mathbf E,\qquad
|\mathbf B|=\frac{n_{\text{med}}}{c}\,|\mathbf E| .
$$

`E`, `B` and the direction of travel are mutually perpendicular. The magnetic
field is not an independent unknown: once `E` is known, `B` follows.

## 2. Refractive index and wavelength

The refractive index is the ratio of the vacuum speed of light to the phase
speed in the material, `n = c/v`. The frequency does not change when light
enters a medium; the wavelength does:

$$
\lambda_{\text{med}}=\frac{\lambda}{n_{\text{med}}},\qquad
k=\frac{2\pi}{\lambda_{\text{med}}}=\frac{2\pi n_{\text{med}}}{\lambda}.
$$

That is why the size parameter is `x = pi*dia/(lambda/nmed)`: the sphere is
measured against the wavelength *in the medium*. Scattering exists only
because the particle and the medium differ, which is why every formula
contains the relative index `m = npar/nmed`. If `m = 1` nothing scatters.

## 3. Poynting vector and irradiance

Energy flows along

$$
\mathbf S=\frac{1}{\mu_0}\,\mathbf E\times\mathbf B ,\qquad
I=\langle|\mathbf S|\rangle=\tfrac12\,n_{\text{med}}\,\varepsilon_0\,c\,|\mathbf E_0|^2 .
$$

The time average `I` is the **irradiance** in W/m², and it is what a
photodiode responds to. Detectors see `|E|^2`, never `E` itself, so the
optical phase is lost unless an interference measurement is made. The code
normalizes the incident irradiance to 1 W/m² by default and reports
scattered irradiance in the same unit.

## 4. Scattering amplitude

Far from the particle (`k r >> 1`) the scattered field is an outgoing
spherical wave. For a sphere it is fully described by two complex functions
of the scattering angle `theta`:

$$
E_{\parallel}^{\text{sca}}=\frac{e^{ikr}}{-ikr}\,S_2(\theta)\,E_{\parallel}^{\text{inc}},\qquad
E_{\perp}^{\text{sca}}=\frac{e^{ikr}}{-ikr}\,S_1(\theta)\,E_{\perp}^{\text{inc}} .
$$

(The code keeps the overall sign convention of the original `mie.m` for the
perpendicular component; a global sign does not affect any irradiance.)

`S1` and `S2` are the **scattering amplitudes**. Mie theory gives them as
sums over multipoles with coefficients `a_n` (electric type) and `b_n`
(magnetic type), found by requiring the tangential `E` and `H` fields to be
continuous at the sphere's surface. The `1/r` field gives a `1/r²` irradiance:

$$
I_{\parallel}=I_0\frac{|S_2|^2}{k^2r^2},\qquad
I_{\perp}=I_0\frac{|S_1|^2}{k^2r^2},\qquad
I_{\text{unpolarized}}=I_0\frac{|S_1|^2+|S_2|^2}{2k^2r^2}.
$$

Dividing out the geometry gives the differential cross section
`dσ/dΩ = r² I / I0` in m²/sr, a property of the particle alone.

## 5. Polarization

"Parallel" and "perpendicular" refer to the **scattering plane**, the plane
containing the incident and the scattered directions. A small particle
radiates like a dipole: the perpendicular component is the same at every
angle, while the parallel component goes as `cos² theta` and vanishes at 90°.
Large particles mix many multipoles and the two components develop different
lobe structures. The ratio of the two is a measurement in its own right.

## 6. Cross sections and conservation of energy

The scattered power divided by the incident irradiance is the scattering
cross section `σ_sca` (m²); divided again by the geometric area it is the
efficiency `Q_sca`. The **optical theorem** ties the power removed from the
beam to the forward amplitude:

$$
Q_{\text{ext}}=\frac{4}{x^2}\,\mathrm{Re}\,S(0),\qquad Q_{\text{ext}}=Q_{\text{sca}}+Q_{\text{abs}} .
$$

For a real refractive index nothing is absorbed, so `Q_ext = Q_sca`. The
unit tests check this identity and check that `dσ/dΩ` integrated over all
directions equals `σ_sca`.

## 7. The three regimes

| Regime | Condition | Behaviour |
|---|---|---|
| Rayleigh | `x << 1`, `\|m\| x << 1` | one dipole; `σ ∝ dia⁶/λ⁴`; smooth pattern |
| Rayleigh-Debye | `\|m−1\| << 1`, `2x\|m−1\| << 1` | dipoles throughout the volume interfere; form factor `P(θ)`; no internal field distortion |
| Mie | any `x`, any `m` (homogeneous sphere) | full multipole sum; lobes, resonances; `Q_ext → 2` for very large spheres |

## 8. Where quantum mechanics enters, and where it does not

The Mie field is a **classical field**: a solution of Maxwell's equations. It
has an amplitude and a phase, it interferes, and its square is an energy flux.

It is **not a quantum wavefunction**, and the code never treats it as one.
The mathematical differences:

| | Classical field `E(r,t)` | Quantum wavefunction `ψ` |
|---|---|---|
| What it is | a measurable physical field (V/m) | a probability amplitude |
| What its square means | energy flux (W/m²), a definite number | probability density of a detection |
| Obeys | Maxwell's equations | the Schrödinger equation (or, for light, the quantized field) |
| Normalization | set by the laser power | must integrate to 1 |
| A weak measurement gives | a proportionally weaker signal | individual, random clicks |

The connection is that the classical irradiance fixes the **mean** rate at
which photons arrive; it says nothing about the fluctuations around that
mean. Only four quantum facts are used in this project:

1. **Photon energy.** `E = h f = h c / λ`. At 1590 nm that is 0.78 eV, so 1 nW
   is about 8 × 10⁹ photons per second.
2. **Poisson photon-counting statistics.** For laser light the number of
   photons detected in a fixed time is Poisson distributed: mean `N`,
   standard deviation `√N`.
3. **Shot noise.** The same statistics seen as a current:
   `i_shot = sqrt(2 q i B)`. It is the floor no amplifier can remove, and it
   is one of three terms in the sensor noise budget.
4. **Coherent versus incoherent illumination.** With coherent light, fields
   from different particles add *before* squaring, which produces speckle;
   as the particles move, the speckle fluctuates, and that fluctuation is
   what the motion estimator measures. When many particles at random
   positions are averaged over time (or the light is incoherent), the cross
   terms average out and the *powers* add. The concentration estimate relies
   on that incoherent sum; the motion estimate relies on the coherent
   fluctuations.
