# Modern physics vocabulary (Knight Ch. 36-42)

Each equation is checked for dimensional consistency in `tests/test_modern_physics_vocab.py`.

## Ch. 36: Relativity

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| inertial reference frame | A viewpoint that is not accelerating, where Newton's first law holds. |  | All of relativity compares measurements made in two such frames. | First step of any relativity problem: name the two frames. |
| principle of relativity | The laws of physics are the same in every inertial frame. |  | No experiment can tell which inertial frame is 'really' at rest. | Use to argue both observers are equally right. |
| speed of light postulate | Every inertial observer measures light at the same speed c. | `c = d/t` | Forces time and space to depend on the observer. | If a result gives a speed above c, something is wrong. |
| event | Something that happens at one place at one instant: (x, t). |  | Relativity compares the coordinates of events, not of objects. | List the events (start, end) before computing anything. |
| simultaneity | Two events happening at the same time in a given frame. |  | Simultaneity is frame-dependent: events simultaneous in one frame are not in another. | Question asks 'simultaneous for the moving observer?' -> use the Lorentz transform of dt, not the interval. |
| relativity of simultaneity | Separated events that are simultaneous in one frame are not simultaneous in a frame moving along their separation. | `dtp = gamma*(dt - v*dx/c**2` | The event ahead (in the direction of motion) happens first for the moving observer. | dt = 0 and dx != 0 in one frame -> dt' = -gamma v dx / c^2. |
| Lorentz factor | The stretch factor gamma, always >= 1. | `gamma = 1/sqrt(1 - v**2/c**2` | gamma = 1 at rest and grows without bound as v -> c. | Compute it first in every relativity problem; remember the SQUARE in v^2/c^2. |
| beta | Speed as a fraction of the speed of light. | `beta = v/c` | Keeps relativity formulas dimensionless. | Write speeds as beta*c to keep the algebra clean. |
| proper time | Time between two events measured by a clock present at both events (same place in that frame). |  | Proper time is the shortest time any observer measures between those events. | Ask: which frame sees both events at the same place? That clock reads proper time. |
| time dilation | Moving clocks run slow as seen by an observer they move past. | `dt = gamma*dtau` | Explains muons reaching the ground (the comic on the exam cover). | Given proper time, find another frame's time -> multiply by gamma. |
| proper length | Length of an object measured in the object's rest frame. |  | The longest length any observer measures. | Distance between things at rest relative to each other (mountains, a ship) in their frame -> L0. |
| length contraction | A moving object is shorter along its direction of motion. | `L = L0/gamma` | Only the length along the motion shrinks. | Given proper length, the moving observer's length is L0/gamma, always SHORTER (sanity check). |
| Lorentz transformation | The equations converting event coordinates between frames. | `xp = gamma*(x - v*t` | Replaces Galileo's x' = x - vt at high speed. | Need coordinates or times of specific events in another frame -> use it (it handles simultaneity, dilation, contraction). |
| spacetime interval | A combination of time and distance between two events that every observer agrees on. | `s2 = c**2*dt**2 - dx**2` | Invariant: same in every frame, so it can classify event pairs but cannot decide simultaneity. | Use to classify (timelike, spacelike, lightlike) or to check a transformation, not to answer 'simultaneous?'. |
| timelike / spacelike / lightlike | s^2 > 0, s^2 < 0, or s^2 = 0. |  | Only timelike-separated events can be causally connected; spacelike ones can change order between frames. | s^2 = 0 means a light signal connects the events, NOT that they are simultaneous. |
| relativistic velocity addition | How a velocity measured in one frame looks in another frame. | `up = (u - v)/(1 - u*v/c**2` | Keeps every speed below c. | Plug in velocities WITH signs (left = negative); check /u'/ < c and that it makes sense. |
| relativistic momentum | Momentum corrected for high speed. | `p = gamma*m*v` | Momentum is conserved in every frame only with the gamma factor. | Collisions or forces at high speed. |
| rest energy | Energy a mass has just by existing. | `E0 = m*c**2` | Mass and energy are interchangeable (fission, fusion, pair production). | Mass changes in reactions -> energy released = (mass lost) c^2. |
| total relativistic energy | Rest energy plus kinetic energy. | `E = gamma*m*c**2` | Grows without bound as v -> c. | Given gamma, E = gamma m c^2. |
| relativistic kinetic energy | Energy of motion at any speed. | `K = (gamma - 1)*m*c**2` | Reduces to (1/2) m v^2 when v << c. | If K is comparable to m c^2 (electrons at ~100 keV and up), use this, not (1/2) m v^2. |
| energy-momentum relation | Links total energy, momentum and mass without needing the speed. | `E**2 = (p*c)**2 + (m*c**2)**2` | For light (m = 0) it gives E = pc. | Given E and m, find p (or vice versa) without computing v. |
| causality | Causes come before effects in every frame. |  | Why nothing (no signal) can travel faster than light. | Spacelike events cannot cause one another. |

## Ch. 37: Foundations of modern physics

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| cathode rays | Beams from the negative electrode of an evacuated tube, later identified as electrons. |  | Led to the discovery of the electron. | History question context. |
| electron | The light, negatively charged particle in every atom. |  | Charge -e, mass 9.11e-31 kg. | Use e = 1.60e-19 C and m_e = 9.11e-31 kg. |
| charge-to-mass ratio | e/m, measured by bending electrons in fields. | `qm = v/(B*r` | Showed electrons are ~2000x lighter than hydrogen ions. | Crossed or deflecting fields with unknown particle -> ratio q/m. |
| velocity selector | Crossed E and B fields that let only one speed through undeflected. | `v = Ef/B` | Balancing forces, not adding fields. | Undeflected charge in crossed fields -> qE = qvB, so v = E/B. Never add E and B (different units). |
| elementary charge | The smallest free charge, e = 1.60e-19 C. | `q = N*e` | Millikan's oil drops showed charge comes in whole multiples of e. | Charge = N e with N an integer. |
| electron volt | Energy one electron gains crossing 1 V: 1.60e-19 J. | `E = q*DV` | The natural energy unit for atoms. | Charge e through potential V -> energy in eV equals V numerically. |
| nucleus | The tiny, dense, positive center of an atom. |  | Rutherford's large-angle scattering proved it exists. | Atomic size ~1e-10 m; nucleus ~1e-15 m. |
| atomic number | Z, the number of protons. |  | Determines the element and its chemistry. | Z identifies the element; changes in beta and alpha decay. |
| mass number | A, the number of protons plus neutrons. |  | Sets nuclear size and approximate mass in u. | A = Z + N; conserved in decays. |
| isotope | Atoms with the same Z but different numbers of neutrons. |  | Same chemistry, different nuclear properties. | Same element symbol, different A. |
| emission and absorption spectra | The specific wavelengths a gas gives off or absorbs. |  | Fingerprints of the atom's energy levels. | Spectral lines -> photon energies hc/lambda -> energy differences between levels. |
| Balmer formula | Wavelengths of hydrogen's visible lines. | `1/lam = R*(Rational(1,4) - 1/n**2` | Explained later by the Bohr model. | Hydrogen visible lines ending on n = 2. |
| blackbody radiation | Light emitted by a hot object because of its temperature. | `lam_peak*Temp = b` | Planck's explanation started quantum theory. | Peak wavelength from temperature -> Wien's law. |

## Ch. 38: Quantization

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| photon | A quantum (smallest packet) of light energy. |  | Light is absorbed and emitted one photon at a time. | Counting light energy -> number of photons x energy per photon. |
| photon energy | Energy of one photon, set by its frequency. | `E = h*f` | Higher frequency (bluer) photons carry more energy. | Given wavelength, use E = hc/lambda: convert to frequency first, NEVER h times lambda. |
| photon energy from wavelength | Same energy written with wavelength. | `E = h*c/lam` | hc = 1240 eV*nm is the fast shortcut. | Given lambda in nm, E(eV) = 1240 / lambda(nm). |
| Planck's constant | h = 6.63e-34 J*s, the scale of quantum effects. |  | Sets how big a quantum of energy is. | Units J*s; hbar = h / (2 pi). |
| photoelectric effect | Light knocking electrons out of a metal. |  | Showed light comes in photons: one photon frees one electron. | Light + metal + electrons -> Einstein's equation, not the hydrogen formula. |
| work function | Minimum energy needed to free an electron from a metal. | `phi = h*f - e*Vs` | A property of the metal, always positive. | phi = hf - e*V_stop. A negative phi means a mistake (often lambda used in place of f). |
| stopping potential | Reverse voltage that just stops the fastest photoelectrons. | `Kmax = e*Vs` | Measures the maximum kinetic energy directly. | e*V_stop = K_max. Make sure the voltage given is the STOPPING one. |
| Einstein's photoelectric equation | Photon energy = work function + maximum kinetic energy. | `Kmax = h*f - phi` | Energy conservation, one photon at a time. | Given any two of f (or lambda), phi, K_max -> the third. |
| threshold frequency | Lowest frequency that can eject electrons. | `f0 = phi/h` | Below it, no electrons no matter how bright the light. | f0 = phi/h; light with f < f0 does nothing. |
| photon momentum | Momentum carried by one photon. | `p = h/lam` | Light pushes (radiation pressure, Compton scattering). | Photon recoil or radiation pressure. |
| de Broglie wavelength | Wavelength of a moving particle. | `lam = h/(m*v` | Matter behaves as a wave (electron diffraction). | Given K (non-relativistic) -> p = sqrt(2 m K), then lambda = h/p. |
| matter wave | The wave associated with a moving particle. |  | Explains quantized orbits and diffraction of electrons. | Wave behavior matters when lambda is comparable to the object size. |
| quantization | A quantity can only take certain discrete values. |  | Energy levels, charge, angular momentum are quantized. | Look for 'only certain values' -> integer quantum numbers. |
| stationary state | An allowed state with a definite energy that does not radiate. |  | Bohr's fix for why atoms don't collapse. | Energy diagram lines are stationary states. |
| quantum jump | A transition between stationary states with emission or absorption of a photon. | `Ephoton = Ei - Ef` | Photon energy = energy difference. | Use the energy diagram: E_photon = /E_i - E_f/, lambda = hc / E_photon. |
| Bohr model | Hydrogen atom with quantized circular orbits. |  | Gets hydrogen's energies right; wrong picture of orbits. | Use only for hydrogen (or one-electron ions). |
| hydrogen energy levels | Allowed energies of hydrogen's electron. | `En = -ER/n**2` | Negative = bound; 13.6 eV frees it from the ground state. | ONLY for hydrogen. For other atoms use the given energy diagram; for a free photon use E = hc/lambda. |
| Bohr radius | Radius of hydrogen's ground-state orbit, 0.0529 nm. | `rn = n**2*aB` | Sets the size of atoms. | Orbit radius r_n = n^2 a_B. |
| ground state | The lowest-energy state. |  | Where systems end up after emitting all they can. | n = 1. |
| excited state | Any state above the ground state. |  | Decays by emitting photons. | n > 1; collisions or photons put atoms there. |
| ionization energy | Energy needed to remove an electron completely. |  | 13.6 eV for hydrogen's ground state. | Ionization = going from E_n to E = 0. |
| Rydberg constant | R = 1.097e7 1/m, the constant in hydrogen's wavelength formula. |  | Combines fundamental constants. | Units 1/m: it gives 1/lambda. |

## Ch. 39: Wave functions and uncertainty

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| wave function | psi(x): the quantum description of a particle. |  | Its square gives probabilities; psi itself is not directly measured. | Asked 'where is the particle likely?' -> look at /psi/^2, not psi. |
| probability density | /psi/^2: probability per unit length. | `Prob = psi**2*dx` | Large where the particle is likely to be found. | Probability in a small interval = /psi/^2 times the width. |
| normalization | The total probability of finding the particle somewhere is 1. |  | Fixes the constant in front of psi. | Unknown amplitude A -> set the integral of /psi/^2 equal to 1. |
| wave packet | A short burst of wave made by adding many frequencies. |  | Models a localized particle. | Short in time or space means broad in frequency or wavelength. |
| bandwidth relation | Duration times frequency spread is about 1. | `df*dt = 1` | Classical root of the uncertainty principle. | Pulse of duration dt -> needs frequencies spread over about 1/dt. |
| Heisenberg uncertainty principle | Position and momentum can't both be known exactly. | `dx*dp >= hbar/2` | A property of waves, not of clumsy measurement. | Confined to size dx -> momentum spread at least hbar/(2 dx). (Some texts use a rougher form; check your book's.) |

## Ch. 40: One-dimensional quantum mechanics

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| Schrodinger equation | The equation that determines psi for a given potential energy. | `d2psi = -(2*m/hbar**2)*(E - Upot)*psi` | Plays the role Newton's law plays classically. | Given U(x), the allowed E and psi come from solving it with boundary conditions. |
| potential-energy function | U(x): how potential energy depends on position. |  | It defines the quantum model (Knight 40.2). | Identify U(x) first: box, well, parabola, barrier. |
| energy diagram | Graph of U(x) with each allowed energy drawn as a horizontal line. |  | Most questions can be read off it. | Spectra, turning points, tunneling and node counts all come from the diagram. |
| classically forbidden region | Where E < U: impossible classically. |  | Quantum psi decays there but is not zero. | Use the penetration distance eta. |
| turning point | Where E = U: a classical particle stops and turns around. | `E = Upot` | Boundary between allowed and forbidden regions. | Set E = U(x) and solve for x. |
| boundary conditions | Requirements psi must meet at walls and edges. |  | They are what quantize the energy. | Rigid wall -> psi = 0 there; finite step -> psi and psi' continuous. |
| quantum number | The integer n labeling allowed states. |  | n = 1 is the ground state in Knight's convention. | Knight starts at n = 1 even for the oscillator. |
| particle in a rigid box | A particle trapped between infinitely high walls. | `En = n**2*h**2/(8*m*L**2` | Simplest quantized system; models quantum dots. | Infinite walls -> E_n grows as n^2 and shrinks as 1/L^2. |
| zero-point energy | The lowest allowed energy is not zero. |  | A confined particle can never be at rest (uncertainty). | Ground-state energy E_1 > 0. |
| node | A point where psi = 0, so the particle is never found there. |  | psi_n has n - 1 nodes. | Count nodes to identify n. |
| correspondence principle | Quantum results approach classical ones for large quantum numbers. |  | Checks that QM agrees with everyday physics. | Large n -> probability spreads out evenly like a classical particle. |
| finite potential well | A well with walls of finite height U0. |  | Finitely many bound states, each below the rigid-box value. | psi leaks into the walls; levels lower than the box. |
| penetration distance | How far psi reaches into a forbidden region. | `eta = hbar/sqrt(2*m*(U0 - E` | Larger when E is close to the top of the barrier. | E < U -> eta = hbar / sqrt(2 m (U - E)). |
| quantum harmonic oscillator | A particle in a parabolic potential well. | `En = (n - Rational(1,2))*hbar*omega` | Equally spaced levels; models vibrating molecules and resonators. | Parabola U = (1/2) k x^2 -> E_n = (n - 1/2) hbar omega (Knight, n from 1). |
| quantum-mechanical tunneling | A particle passing through a barrier higher than its energy. | `Ptun = exp(-2*w/eta` | Used in STMs, tunnel diodes, flash memory; causes alpha decay. | E < U0 and finite width w -> P ~ exp(-2 w / eta). Exponentially sensitive to w. |
| scanning tunneling microscope | Microscope that maps surfaces by tunneling current. |  | Exponential sensitivity resolves single atoms. | Tunneling application example. |
| bound state | A state with E below the potential at infinity, so the particle stays localized. |  | Bound states have discrete energies. | Discrete lines on the energy diagram. |

## Ch. 41: Atomic physics

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| principal quantum number | n = 1, 2, 3, ...: sets the shell and mostly the energy. |  | Shell size grows with n. | Energy level label. |
| orbital quantum number | l = 0 ... n - 1: sets the orbital angular momentum. | `Lmag = sqrt(l*(l + 1))*hbar` | s, p, d, f = l of 0, 1, 2, 3. | Angular momentum magnitude L = sqrt(l(l+1)) hbar. |
| magnetic quantum number | m = -l ... +l: the z-component of angular momentum. | `Lz = m*hbar` | Gives 2l + 1 orientations. | L_z = m hbar. |
| electron spin | Intrinsic angular momentum of the electron: spin up or down. |  | Stern-Gerlach experiment; doubles the states per orbital. | m_s = +1/2 or -1/2. |
| Pauli exclusion principle | No two electrons in an atom share all four quantum numbers. |  | Builds the periodic table and makes matter rigid. | Count electrons per subshell: 2(2l + 1). |
| shell and subshell | Shell = same n; subshell = same n and l. |  | Electron configurations fill subshells in order of energy. | 1s 2s 2p 3s 3p 4s 3d ... |
| electron configuration | List of occupied subshells, e.g. 1s2 2s2 2p6. |  | Explains chemical behavior. | Fill lowest energy first, max 2(2l + 1) per subshell. |
| selection rule | Photon transitions require delta l = +/- 1. |  | A photon carries one unit of angular momentum. | Allowed vs forbidden spectral lines. |
| excited-state lifetime | Average time before an excited atom decays. | `Nt = N0*exp(-t/tau` | Typical ~1e-8 s for allowed transitions. | Number still excited falls exponentially. |
| stimulated emission | A photon triggers an excited atom to emit an identical photon. |  | The physics behind the laser. | Same frequency, phase and direction as the trigger photon. |
| population inversion | More atoms in the upper level than the lower. |  | Needed for laser gain. | Requires pumping energy in. |
| laser | Light amplification by stimulated emission of radiation. |  | Coherent, single-color, directional light. | Safety: compute the eyewear optical density for the beam. |

## Ch. 42: Nuclear physics

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| nucleon | A proton or neutron. |  | The building blocks of nuclei. | A counts nucleons. |
| nuclide | A specific nucleus with given Z and A. |  | Written as a symbol with A and Z. | Balance A and Z in every reaction. |
| atomic mass unit | u = 1/12 the mass of carbon-12, 1.66e-27 kg (931.5 MeV/c^2). |  | Convenient unit for nuclear masses. | 931.5 MeV per u of mass converts to energy. |
| nuclear radius | Nuclei have radius ~ r0 A^(1/3), r0 ~ 1.2 fm. | `r = r0*A**Rational(1,3` | Constant density: volume proportional to A. | Size from mass number. |
| strong force | Short-range attraction between nucleons. |  | Overcomes proton repulsion inside the nucleus. | Range ~1e-15 m. |
| binding energy | Energy needed to pull a nucleus apart into its nucleons. | `B = (Z*mH + N*mn - matom)*c**2` | Comes from the mass defect: the nucleus weighs less than its parts. | Use atomic masses (hydrogen-atom mass for protons) so electrons cancel. |
| binding energy per nucleon | Binding energy divided by A. |  | Peaks near iron: fusion of light and fission of heavy nuclei both release energy. | Compare stability across nuclei. |
| radioactivity | Spontaneous decay of unstable nuclei. |  | Random for each nucleus, predictable on average. | Decay laws are statistical. |
| alpha decay | Emission of a helium-4 nucleus. |  | A drops by 4, Z by 2; happens by tunneling. | Balance A and Z. |
| beta decay | A neutron becomes a proton (beta-minus) or the reverse (beta-plus), emitting an electron or positron. |  | A unchanged, Z changes by 1. | Balance charge and A. |
| gamma decay | An excited nucleus emits a high-energy photon. |  | A and Z unchanged. | Photon energy = nuclear level difference. |
| half-life | Time for half of the nuclei to decay. | `thalf = tau*log(2` | Independent of how many you start with. | N = N0 (1/2)^(t / t_half); t_half = tau ln 2. |
| activity | Decays per second. | `Ract = Nn/tau` | Measured in becquerel (1 Bq = 1 decay/s). | R = N / tau. |
| fission and fusion | Splitting heavy nuclei / joining light ones. |  | Both move toward the binding-energy peak and release energy. | Energy = mass lost times c^2. |

## Math words

| Term | English | Math | Science | Decide |
|---|---|---|---|---|
| exponential decay | A quantity that falls by the same fraction in each equal time. | `y = y0*exp(-t/tau` | Decay, RC circuits, tunneling all follow it. | Take ln of both sides to solve for time. |
| natural logarithm | ln, the inverse of e^x. |  | Turns exponential equations into linear ones. | ln(e^a) = a; ln(1/q) = -ln q. |
| complex number | a + i b with i^2 = -1. |  | Wave functions are complex; /psi/^2 = psi* psi is real. | Probability uses the magnitude squared. |
| dimensionless | A number with no units. |  | Arguments of exp, sin, log must be dimensionless. | Check units of every exponent: a unit inside exp is a mistake. |
| binomial approximation | (1 + x)^n ~ 1 + n x for small x. |  | Shows relativity reduces to Newton at low speed. | Small parameter (like v/c) -> expand. |
| eigenvalue | A special value (like an allowed energy) for which an equation has a solution. |  | Allowed energies are eigenvalues of the Schrodinger equation. | Computers find them with a matrix solver. |

## Decisions: given -> asked -> use -> avoid

| Given | Asked | Use | Avoid |
|---|---|---|---|
| photon wavelength | photon energy or frequency | f = c/lambda; E = h f = hc/lambda (E[eV] = 1240/lambda[nm]) | h times lambda (wrong units); the hydrogen formula |
| light on a metal, stopping potential | work function, K_max, or v_max | K_max = e V_stop; phi = hf - K_max; v = sqrt(2 K_max / m) | negative work function; the hydrogen formula; using a voltage that is not the stopping potential |
| an atom's energy levels or a diagram | emitted or absorbed wavelengths | lambda = hc / |E_i - E_f| for each allowed pair | -13.6/n^2 unless the atom is hydrogen |
| events simultaneous in one frame | are they simultaneous for a moving observer? | dt' = gamma (dt - v dx / c^2) | the spacetime interval (it is the same in all frames) |
| a length measured in the object's rest frame | length seen by a moving observer | L = L0 / gamma (shorter) | dropping the square in gamma; getting a longer length |
| a clock's own (proper) time | time in another frame | dt = gamma dtau (longer) | dividing by gamma |
| two velocities in different frames | velocity in the other frame | u' = (u - v)/(1 - u v / c^2) with signs | Galilean u - v; ignoring direction |
| a fast particle (K comparable to m c^2) | energy, momentum, or speed | gamma, E = gamma m c^2, E^2 = (pc)^2 + (mc^2)^2 | (1/2) m v^2 at high speed |
| a particle's kinetic energy or speed | de Broglie wavelength | p = sqrt(2 m K) (slow) or relativistic p; lambda = h / p | using the photon formula E = hc/lambda for a massive particle |
| a potential-energy shape | allowed energies or wave functions | box: n^2; finite well: lower, finite count; parabola: (n - 1/2) hbar omega | mixing up the box and oscillator formulas |
| E below a barrier of width w | tunneling probability | eta = hbar / sqrt(2 m (U0 - E)); P ~ exp(-2 w / eta) | using E instead of U0 - E |
| half-life and elapsed time | amount left or activity | N = N0 (1/2)^(t / t_half); R = N / tau | subtracting a fixed amount per half-life |
| nuclear or atomic masses | binding energy or energy released | mass difference times c^2 (931.5 MeV per u) | mixing nuclear and atomic masses |
| a charge in crossed E and B fields | speed that passes undeflected | qE = qvB -> v = E/B | adding E and B (different units) |
