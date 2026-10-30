"""
Modern-physics vocabulary (Knight Ch. 36-42), each word in four forms, plus a decision guide.

In plain words
--------------
Every entry gives a term:
  * **english**   what it means in plain words
  * **equation**  the math, when the term has one (SymPy syntax), with SI **units** for each symbol
  * **science**   why it matters / what it tells you about nature
  * **decide**    when to reach for it on a problem (and the common trap)

``check_dimensions`` checks every equation for dimensional consistency: both sides must have
the same SI dimensions, sums must add like units, and the argument of exp/sin/cos/log must be
dimensionless. (SymPy alone accepts exp(meters), so that last check is done here.) The midterm
mistake E = h * lambda fails this check, which is the point.

``DECISIONS`` is a "given -> asked for -> use -> avoid" table built from common PHYS 106
mistakes, and ``decide(question)`` ranks the rules for a question by keyword overlap.

The definitions are written for this repo, in Knight's notation. Chapter numbers follow Knight,
Physics for Scientists and Engineers, 5th ed. (36 relativity, 37 foundations, 38 quantization,
39 wave functions, 40 one-dimensional QM, 41 atomic physics, 42 nuclear physics). Your course
syllabus decides which chapters are tested.
"""
import random
import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.physics import units as U
from sympy.physics.units.systems.si import SI

UNITS = {"1": sp.Integer(1), "m": U.meter, "s": U.second, "kg": U.kilogram, "J": U.joule, "eV": U.electronvolt,
         "Hz": U.hertz, "C": U.coulomb, "V": U.volt, "T": U.tesla, "K": U.kelvin, "N": U.newton, "W": U.watt,
         "Bq": U.becquerel, "u": U.atomic_mass_unit, "A": U.ampere}
CHAPTERS = {36: "Relativity", 37: "Foundations of modern physics", 38: "Quantization",
            39: "Wave functions and uncertainty", 40: "One-dimensional quantum mechanics",
            41: "Atomic physics", 42: "Nuclear physics", 0: "Math words"}


@dataclass
class Term:
    term: str
    chapter: int
    english: str
    science: str
    decide: str
    equation: str = ""                     # SymPy syntax, e.g. "Eq(E, h*f)"
    units: dict = field(default_factory=dict)


T = Term
VOCAB = [
    # ---- 36 Relativity -------------------------------------------------------------------
    T("inertial reference frame", 36, "A viewpoint that is not accelerating, where Newton's first law holds.",
      "All of relativity compares measurements made in two such frames.", "First step of any relativity problem: name the two frames."),
    T("principle of relativity", 36, "The laws of physics are the same in every inertial frame.",
      "No experiment can tell which inertial frame is 'really' at rest.", "Use to argue both observers are equally right."),
    T("speed of light postulate", 36, "Every inertial observer measures light at the same speed c.",
      "Forces time and space to depend on the observer.", "If a result gives a speed above c, something is wrong.",
      "Eq(c, d/t)", {"c": "m/s", "d": "m", "t": "s"}),
    T("event", 36, "Something that happens at one place at one instant: (x, t).",
      "Relativity compares the coordinates of events, not of objects.", "List the events (start, end) before computing anything."),
    T("simultaneity", 36, "Two events happening at the same time in a given frame.",
      "Simultaneity is frame-dependent: events simultaneous in one frame are not in another.",
      "Question asks 'simultaneous for the moving observer?' -> use the Lorentz transform of dt, not the interval."),
    T("relativity of simultaneity", 36, "Separated events that are simultaneous in one frame are not simultaneous in a frame moving along their separation.",
      "The event ahead (in the direction of motion) happens first for the moving observer.",
      "dt = 0 and dx != 0 in one frame -> dt' = -gamma v dx / c^2.",
      "Eq(dtp, gamma*(dt - v*dx/c**2))", {"dtp": "s", "gamma": "1", "dt": "s", "v": "m/s", "dx": "m", "c": "m/s"}),
    T("Lorentz factor", 36, "The stretch factor gamma, always >= 1.", "gamma = 1 at rest and grows without bound as v -> c.",
      "Compute it first in every relativity problem; remember the SQUARE in v^2/c^2.",
      "Eq(gamma, 1/sqrt(1 - v**2/c**2))", {"gamma": "1", "v": "m/s", "c": "m/s"}),
    T("beta", 36, "Speed as a fraction of the speed of light.", "Keeps relativity formulas dimensionless.", "Write speeds as beta*c to keep the algebra clean.",
      "Eq(beta, v/c)", {"beta": "1", "v": "m/s", "c": "m/s"}),
    T("proper time", 36, "Time between two events measured by a clock present at both events (same place in that frame).",
      "Proper time is the shortest time any observer measures between those events.", "Ask: which frame sees both events at the same place? That clock reads proper time."),
    T("time dilation", 36, "Moving clocks run slow as seen by an observer they move past.", "Explains muons reaching the ground (the comic on the exam cover).",
      "Given proper time, find another frame's time -> multiply by gamma.",
      "Eq(dt, gamma*dtau)", {"dt": "s", "gamma": "1", "dtau": "s"}),
    T("proper length", 36, "Length of an object measured in the object's rest frame.", "The longest length any observer measures.",
      "Distance between things at rest relative to each other (mountains, a ship) in their frame -> L0."),
    T("length contraction", 36, "A moving object is shorter along its direction of motion.", "Only the length along the motion shrinks.",
      "Given proper length, the moving observer's length is L0/gamma, always SHORTER (sanity check).",
      "Eq(L, L0/gamma)", {"L": "m", "L0": "m", "gamma": "1"}),
    T("Lorentz transformation", 36, "The equations converting event coordinates between frames.", "Replaces Galileo's x' = x - vt at high speed.",
      "Need coordinates or times of specific events in another frame -> use it (it handles simultaneity, dilation, contraction).",
      "Eq(xp, gamma*(x - v*t))", {"xp": "m", "gamma": "1", "x": "m", "v": "m/s", "t": "s"}),
    T("spacetime interval", 36, "A combination of time and distance between two events that every observer agrees on.",
      "Invariant: same in every frame, so it can classify event pairs but cannot decide simultaneity.",
      "Use to classify (timelike, spacelike, lightlike) or to check a transformation, not to answer 'simultaneous?'.",
      "Eq(s2, c**2*dt**2 - dx**2)", {"s2": "m**2", "c": "m/s", "dt": "s", "dx": "m"}),
    T("timelike / spacelike / lightlike", 36, "s^2 > 0, s^2 < 0, or s^2 = 0.", "Only timelike-separated events can be causally connected; spacelike ones can change order between frames.",
      "s^2 = 0 means a light signal connects the events, NOT that they are simultaneous."),
    T("relativistic velocity addition", 36, "How a velocity measured in one frame looks in another frame.", "Keeps every speed below c.",
      "Plug in velocities WITH signs (left = negative); check |u'| < c and that it makes sense.",
      "Eq(up, (u - v)/(1 - u*v/c**2))", {"up": "m/s", "u": "m/s", "v": "m/s", "c": "m/s"}),
    T("relativistic momentum", 36, "Momentum corrected for high speed.", "Momentum is conserved in every frame only with the gamma factor.",
      "Collisions or forces at high speed.", "Eq(p, gamma*m*v)", {"p": "kg*m/s", "gamma": "1", "m": "kg", "v": "m/s"}),
    T("rest energy", 36, "Energy a mass has just by existing.", "Mass and energy are interchangeable (fission, fusion, pair production).",
      "Mass changes in reactions -> energy released = (mass lost) c^2.", "Eq(E0, m*c**2)", {"E0": "J", "m": "kg", "c": "m/s"}),
    T("total relativistic energy", 36, "Rest energy plus kinetic energy.", "Grows without bound as v -> c.", "Given gamma, E = gamma m c^2.",
      "Eq(E, gamma*m*c**2)", {"E": "J", "gamma": "1", "m": "kg", "c": "m/s"}),
    T("relativistic kinetic energy", 36, "Energy of motion at any speed.", "Reduces to (1/2) m v^2 when v << c.",
      "If K is comparable to m c^2 (electrons at ~100 keV and up), use this, not (1/2) m v^2.",
      "Eq(K, (gamma - 1)*m*c**2)", {"K": "J", "gamma": "1", "m": "kg", "c": "m/s"}),
    T("energy-momentum relation", 36, "Links total energy, momentum and mass without needing the speed.", "For light (m = 0) it gives E = pc.",
      "Given E and m, find p (or vice versa) without computing v.",
      "Eq(E**2, (p*c)**2 + (m*c**2)**2)", {"E": "J", "p": "kg*m/s", "c": "m/s", "m": "kg"}),
    T("causality", 36, "Causes come before effects in every frame.", "Why nothing (no signal) can travel faster than light.", "Spacelike events cannot cause one another."),
    # ---- 37 Foundations ------------------------------------------------------------------
    T("cathode rays", 37, "Beams from the negative electrode of an evacuated tube, later identified as electrons.", "Led to the discovery of the electron.", "History question context."),
    T("electron", 37, "The light, negatively charged particle in every atom.", "Charge -e, mass 9.11e-31 kg.", "Use e = 1.60e-19 C and m_e = 9.11e-31 kg."),
    T("charge-to-mass ratio", 37, "e/m, measured by bending electrons in fields.", "Showed electrons are ~2000x lighter than hydrogen ions.",
      "Crossed or deflecting fields with unknown particle -> ratio q/m.", "Eq(qm, v/(B*r))", {"qm": "C/kg", "v": "m/s", "B": "T", "r": "m"}),
    T("velocity selector", 37, "Crossed E and B fields that let only one speed through undeflected.", "Balancing forces, not adding fields.",
      "Undeflected charge in crossed fields -> qE = qvB, so v = E/B. Never add E and B (different units).",
      "Eq(v, Ef/B)", {"v": "m/s", "Ef": "V/m", "B": "T"}),
    T("elementary charge", 37, "The smallest free charge, e = 1.60e-19 C.", "Millikan's oil drops showed charge comes in whole multiples of e.",
      "Charge = N e with N an integer.", "Eq(q, N*e)", {"q": "C", "N": "1", "e": "C"}),
    T("electron volt", 37, "Energy one electron gains crossing 1 V: 1.60e-19 J.", "The natural energy unit for atoms.",
      "Charge e through potential V -> energy in eV equals V numerically.", "Eq(E, q*DV)", {"E": "J", "q": "C", "DV": "V"}),
    T("nucleus", 37, "The tiny, dense, positive center of an atom.", "Rutherford's large-angle scattering proved it exists.", "Atomic size ~1e-10 m; nucleus ~1e-15 m."),
    T("atomic number", 37, "Z, the number of protons.", "Determines the element and its chemistry.", "Z identifies the element; changes in beta and alpha decay."),
    T("mass number", 37, "A, the number of protons plus neutrons.", "Sets nuclear size and approximate mass in u.", "A = Z + N; conserved in decays."),
    T("isotope", 37, "Atoms with the same Z but different numbers of neutrons.", "Same chemistry, different nuclear properties.", "Same element symbol, different A."),
    T("emission and absorption spectra", 37, "The specific wavelengths a gas gives off or absorbs.", "Fingerprints of the atom's energy levels.",
      "Spectral lines -> photon energies hc/lambda -> energy differences between levels."),
    T("Balmer formula", 37, "Wavelengths of hydrogen's visible lines.", "Explained later by the Bohr model.",
      "Hydrogen visible lines ending on n = 2.", "Eq(1/lam, R*(Rational(1,4) - 1/n**2))", {"lam": "m", "R": "1/m", "n": "1"}),
    T("blackbody radiation", 37, "Light emitted by a hot object because of its temperature.", "Planck's explanation started quantum theory.",
      "Peak wavelength from temperature -> Wien's law.", "Eq(lam_peak*Temp, b)", {"lam_peak": "m", "Temp": "K", "b": "m*K"}),
    # ---- 38 Quantization -----------------------------------------------------------------
    T("photon", 38, "A quantum (smallest packet) of light energy.", "Light is absorbed and emitted one photon at a time.",
      "Counting light energy -> number of photons x energy per photon."),
    T("photon energy", 38, "Energy of one photon, set by its frequency.", "Higher frequency (bluer) photons carry more energy.",
      "Given wavelength, use E = hc/lambda: convert to frequency first, NEVER h times lambda.",
      "Eq(E, h*f)", {"E": "J", "h": "J*s", "f": "Hz"}),
    T("photon energy from wavelength", 38, "Same energy written with wavelength.", "hc = 1240 eV*nm is the fast shortcut.",
      "Given lambda in nm, E(eV) = 1240 / lambda(nm).", "Eq(E, h*c/lam)", {"E": "J", "h": "J*s", "c": "m/s", "lam": "m"}),
    T("Planck's constant", 38, "h = 6.63e-34 J*s, the scale of quantum effects.", "Sets how big a quantum of energy is.", "Units J*s; hbar = h / (2 pi)."),
    T("photoelectric effect", 38, "Light knocking electrons out of a metal.", "Showed light comes in photons: one photon frees one electron.",
      "Light + metal + electrons -> Einstein's equation, not the hydrogen formula."),
    T("work function", 38, "Minimum energy needed to free an electron from a metal.", "A property of the metal, always positive.",
      "phi = hf - e*V_stop. A negative phi means a mistake (often lambda used in place of f).",
      "Eq(phi, h*f - e*Vs)", {"phi": "J", "h": "J*s", "f": "Hz", "e": "C", "Vs": "V"}),
    T("stopping potential", 38, "Reverse voltage that just stops the fastest photoelectrons.", "Measures the maximum kinetic energy directly.",
      "e*V_stop = K_max. Make sure the voltage given is the STOPPING one.", "Eq(Kmax, e*Vs)", {"Kmax": "J", "e": "C", "Vs": "V"}),
    T("Einstein's photoelectric equation", 38, "Photon energy = work function + maximum kinetic energy.", "Energy conservation, one photon at a time.",
      "Given any two of f (or lambda), phi, K_max -> the third.", "Eq(Kmax, h*f - phi)", {"Kmax": "J", "h": "J*s", "f": "Hz", "phi": "J"}),
    T("threshold frequency", 38, "Lowest frequency that can eject electrons.", "Below it, no electrons no matter how bright the light.",
      "f0 = phi/h; light with f < f0 does nothing.", "Eq(f0, phi/h)", {"f0": "Hz", "phi": "J", "h": "J*s"}),
    T("photon momentum", 38, "Momentum carried by one photon.", "Light pushes (radiation pressure, Compton scattering).",
      "Photon recoil or radiation pressure.", "Eq(p, h/lam)", {"p": "kg*m/s", "h": "J*s", "lam": "m"}),
    T("de Broglie wavelength", 38, "Wavelength of a moving particle.", "Matter behaves as a wave (electron diffraction).",
      "Given K (non-relativistic) -> p = sqrt(2 m K), then lambda = h/p.", "Eq(lam, h/(m*v))", {"lam": "m", "h": "J*s", "m": "kg", "v": "m/s"}),
    T("matter wave", 38, "The wave associated with a moving particle.", "Explains quantized orbits and diffraction of electrons.", "Wave behavior matters when lambda is comparable to the object size."),
    T("quantization", 38, "A quantity can only take certain discrete values.", "Energy levels, charge, angular momentum are quantized.", "Look for 'only certain values' -> integer quantum numbers."),
    T("stationary state", 38, "An allowed state with a definite energy that does not radiate.", "Bohr's fix for why atoms don't collapse.", "Energy diagram lines are stationary states."),
    T("quantum jump", 38, "A transition between stationary states with emission or absorption of a photon.", "Photon energy = energy difference.",
      "Use the energy diagram: E_photon = |E_i - E_f|, lambda = hc / E_photon.", "Eq(Ephoton, Ei - Ef)", {"Ephoton": "J", "Ei": "J", "Ef": "J"}),
    T("Bohr model", 38, "Hydrogen atom with quantized circular orbits.", "Gets hydrogen's energies right; wrong picture of orbits.", "Use only for hydrogen (or one-electron ions)."),
    T("hydrogen energy levels", 38, "Allowed energies of hydrogen's electron.", "Negative = bound; 13.6 eV frees it from the ground state.",
      "ONLY for hydrogen. For other atoms use the given energy diagram; for a free photon use E = hc/lambda.",
      "Eq(En, -ER/n**2)", {"En": "eV", "ER": "eV", "n": "1"}),
    T("Bohr radius", 38, "Radius of hydrogen's ground-state orbit, 0.0529 nm.", "Sets the size of atoms.", "Orbit radius r_n = n^2 a_B.",
      "Eq(rn, n**2*aB)", {"rn": "m", "n": "1", "aB": "m"}),
    T("ground state", 38, "The lowest-energy state.", "Where systems end up after emitting all they can.", "n = 1."),
    T("excited state", 38, "Any state above the ground state.", "Decays by emitting photons.", "n > 1; collisions or photons put atoms there."),
    T("ionization energy", 38, "Energy needed to remove an electron completely.", "13.6 eV for hydrogen's ground state.", "Ionization = going from E_n to E = 0."),
    T("Rydberg constant", 38, "R = 1.097e7 1/m, the constant in hydrogen's wavelength formula.", "Combines fundamental constants.", "Units 1/m: it gives 1/lambda."),
    # ---- 39 Wave functions and uncertainty -----------------------------------------------
    T("wave function", 39, "psi(x): the quantum description of a particle.", "Its square gives probabilities; psi itself is not directly measured.",
      "Asked 'where is the particle likely?' -> look at |psi|^2, not psi."),
    T("probability density", 39, "|psi|^2: probability per unit length.", "Large where the particle is likely to be found.",
      "Probability in a small interval = |psi|^2 times the width.", "Eq(Prob, psi**2*dx)", {"Prob": "1", "psi": "m**(-1/2)", "dx": "m"}),
    T("normalization", 39, "The total probability of finding the particle somewhere is 1.", "Fixes the constant in front of psi.",
      "Unknown amplitude A -> set the integral of |psi|^2 equal to 1."),
    T("wave packet", 39, "A short burst of wave made by adding many frequencies.", "Models a localized particle.", "Short in time or space means broad in frequency or wavelength."),
    T("bandwidth relation", 39, "Duration times frequency spread is about 1.", "Classical root of the uncertainty principle.",
      "Pulse of duration dt -> needs frequencies spread over about 1/dt.", "Eq(df*dt, 1)", {"df": "Hz", "dt": "s"}),
    T("Heisenberg uncertainty principle", 39, "Position and momentum can't both be known exactly.", "A property of waves, not of clumsy measurement.",
      "Confined to size dx -> momentum spread at least hbar/(2 dx). (Some texts use a rougher form; check your book's.)",
      "Ge(dx*dp, hbar/2)", {"dx": "m", "dp": "kg*m/s", "hbar": "J*s"}),
    # ---- 40 One-dimensional QM -----------------------------------------------------------
    T("Schrodinger equation", 40, "The equation that determines psi for a given potential energy.", "Plays the role Newton's law plays classically.",
      "Given U(x), the allowed E and psi come from solving it with boundary conditions.",
      "Eq(d2psi, -(2*m/hbar**2)*(E - Upot)*psi)", {"d2psi": "m**(-5/2)", "m": "kg", "hbar": "J*s", "E": "J", "Upot": "J", "psi": "m**(-1/2)"}),
    T("potential-energy function", 40, "U(x): how potential energy depends on position.", "It defines the quantum model (Knight 40.2).",
      "Identify U(x) first: box, well, parabola, barrier."),
    T("energy diagram", 40, "Graph of U(x) with each allowed energy drawn as a horizontal line.", "Most questions can be read off it.",
      "Spectra, turning points, tunneling and node counts all come from the diagram."),
    T("classically forbidden region", 40, "Where E < U: impossible classically.", "Quantum psi decays there but is not zero.", "Use the penetration distance eta."),
    T("turning point", 40, "Where E = U: a classical particle stops and turns around.", "Boundary between allowed and forbidden regions.", "Set E = U(x) and solve for x.",
      "Eq(E, Upot)", {"E": "J", "Upot": "J"}),
    T("boundary conditions", 40, "Requirements psi must meet at walls and edges.", "They are what quantize the energy.", "Rigid wall -> psi = 0 there; finite step -> psi and psi' continuous."),
    T("quantum number", 40, "The integer n labeling allowed states.", "n = 1 is the ground state in Knight's convention.", "Knight starts at n = 1 even for the oscillator."),
    T("particle in a rigid box", 40, "A particle trapped between infinitely high walls.", "Simplest quantized system; models quantum dots.",
      "Infinite walls -> E_n grows as n^2 and shrinks as 1/L^2.", "Eq(En, n**2*h**2/(8*m*L**2))", {"En": "J", "n": "1", "h": "J*s", "m": "kg", "L": "m"}),
    T("zero-point energy", 40, "The lowest allowed energy is not zero.", "A confined particle can never be at rest (uncertainty).", "Ground-state energy E_1 > 0."),
    T("node", 40, "A point where psi = 0, so the particle is never found there.", "psi_n has n - 1 nodes.", "Count nodes to identify n."),
    T("correspondence principle", 40, "Quantum results approach classical ones for large quantum numbers.", "Checks that QM agrees with everyday physics.", "Large n -> probability spreads out evenly like a classical particle."),
    T("finite potential well", 40, "A well with walls of finite height U0.", "Finitely many bound states, each below the rigid-box value.", "psi leaks into the walls; levels lower than the box."),
    T("penetration distance", 40, "How far psi reaches into a forbidden region.", "Larger when E is close to the top of the barrier.",
      "E < U -> eta = hbar / sqrt(2 m (U - E)).", "Eq(eta, hbar/sqrt(2*m*(U0 - E)))", {"eta": "m", "hbar": "J*s", "m": "kg", "U0": "J", "E": "J"}),
    T("quantum harmonic oscillator", 40, "A particle in a parabolic potential well.", "Equally spaced levels; models vibrating molecules and resonators.",
      "Parabola U = (1/2) k x^2 -> E_n = (n - 1/2) hbar omega (Knight, n from 1).",
      "Eq(En, (n - Rational(1,2))*hbar*omega)", {"En": "J", "n": "1", "hbar": "J*s", "omega": "1/s"}),
    T("quantum-mechanical tunneling", 40, "A particle passing through a barrier higher than its energy.", "Used in STMs, tunnel diodes, flash memory; causes alpha decay.",
      "E < U0 and finite width w -> P ~ exp(-2 w / eta). Exponentially sensitive to w.",
      "Eq(Ptun, exp(-2*w/eta))", {"Ptun": "1", "w": "m", "eta": "m"}),
    T("scanning tunneling microscope", 40, "Microscope that maps surfaces by tunneling current.", "Exponential sensitivity resolves single atoms.", "Tunneling application example."),
    T("bound state", 40, "A state with E below the potential at infinity, so the particle stays localized.", "Bound states have discrete energies.", "Discrete lines on the energy diagram."),
    # ---- 41 Atomic physics ---------------------------------------------------------------
    T("principal quantum number", 41, "n = 1, 2, 3, ...: sets the shell and mostly the energy.", "Shell size grows with n.", "Energy level label."),
    T("orbital quantum number", 41, "l = 0 ... n - 1: sets the orbital angular momentum.", "s, p, d, f = l of 0, 1, 2, 3.",
      "Angular momentum magnitude L = sqrt(l(l+1)) hbar.", "Eq(Lmag, sqrt(l*(l + 1))*hbar)", {"Lmag": "J*s", "l": "1", "hbar": "J*s"}),
    T("magnetic quantum number", 41, "m = -l ... +l: the z-component of angular momentum.", "Gives 2l + 1 orientations.",
      "L_z = m hbar.", "Eq(Lz, m*hbar)", {"Lz": "J*s", "m": "1", "hbar": "J*s"}),
    T("electron spin", 41, "Intrinsic angular momentum of the electron: spin up or down.", "Stern-Gerlach experiment; doubles the states per orbital.", "m_s = +1/2 or -1/2."),
    T("Pauli exclusion principle", 41, "No two electrons in an atom share all four quantum numbers.", "Builds the periodic table and makes matter rigid.",
      "Count electrons per subshell: 2(2l + 1)."),
    T("shell and subshell", 41, "Shell = same n; subshell = same n and l.", "Electron configurations fill subshells in order of energy.", "1s 2s 2p 3s 3p 4s 3d ..."),
    T("electron configuration", 41, "List of occupied subshells, e.g. 1s2 2s2 2p6.", "Explains chemical behavior.", "Fill lowest energy first, max 2(2l + 1) per subshell."),
    T("selection rule", 41, "Photon transitions require delta l = +/- 1.", "A photon carries one unit of angular momentum.", "Allowed vs forbidden spectral lines."),
    T("excited-state lifetime", 41, "Average time before an excited atom decays.", "Typical ~1e-8 s for allowed transitions.",
      "Number still excited falls exponentially.", "Eq(Nt, N0*exp(-t/tau))", {"Nt": "1", "N0": "1", "t": "s", "tau": "s"}),
    T("stimulated emission", 41, "A photon triggers an excited atom to emit an identical photon.", "The physics behind the laser.", "Same frequency, phase and direction as the trigger photon."),
    T("population inversion", 41, "More atoms in the upper level than the lower.", "Needed for laser gain.", "Requires pumping energy in."),
    T("laser", 41, "Light amplification by stimulated emission of radiation.", "Coherent, single-color, directional light.", "Safety: compute the eyewear optical density for the beam."),
    # ---- 42 Nuclear physics --------------------------------------------------------------
    T("nucleon", 42, "A proton or neutron.", "The building blocks of nuclei.", "A counts nucleons."),
    T("nuclide", 42, "A specific nucleus with given Z and A.", "Written as a symbol with A and Z.", "Balance A and Z in every reaction."),
    T("atomic mass unit", 42, "u = 1/12 the mass of carbon-12, 1.66e-27 kg (931.5 MeV/c^2).", "Convenient unit for nuclear masses.", "931.5 MeV per u of mass converts to energy."),
    T("nuclear radius", 42, "Nuclei have radius ~ r0 A^(1/3), r0 ~ 1.2 fm.", "Constant density: volume proportional to A.",
      "Size from mass number.", "Eq(r, r0*A**Rational(1,3))", {"r": "m", "r0": "m", "A": "1"}),
    T("strong force", 42, "Short-range attraction between nucleons.", "Overcomes proton repulsion inside the nucleus.", "Range ~1e-15 m."),
    T("binding energy", 42, "Energy needed to pull a nucleus apart into its nucleons.", "Comes from the mass defect: the nucleus weighs less than its parts.",
      "Use atomic masses (hydrogen-atom mass for protons) so electrons cancel.",
      "Eq(B, (Z*mH + N*mn - matom)*c**2)", {"B": "J", "Z": "1", "mH": "kg", "N": "1", "mn": "kg", "matom": "kg", "c": "m/s"}),
    T("binding energy per nucleon", 42, "Binding energy divided by A.", "Peaks near iron: fusion of light and fission of heavy nuclei both release energy.", "Compare stability across nuclei."),
    T("radioactivity", 42, "Spontaneous decay of unstable nuclei.", "Random for each nucleus, predictable on average.", "Decay laws are statistical."),
    T("alpha decay", 42, "Emission of a helium-4 nucleus.", "A drops by 4, Z by 2; happens by tunneling.", "Balance A and Z."),
    T("beta decay", 42, "A neutron becomes a proton (beta-minus) or the reverse (beta-plus), emitting an electron or positron.", "A unchanged, Z changes by 1.", "Balance charge and A."),
    T("gamma decay", 42, "An excited nucleus emits a high-energy photon.", "A and Z unchanged.", "Photon energy = nuclear level difference."),
    T("half-life", 42, "Time for half of the nuclei to decay.", "Independent of how many you start with.",
      "N = N0 (1/2)^(t / t_half); t_half = tau ln 2.", "Eq(thalf, tau*log(2))", {"thalf": "s", "tau": "s"}),
    T("activity", 42, "Decays per second.", "Measured in becquerel (1 Bq = 1 decay/s).", "R = N / tau.", "Eq(Ract, Nn/tau)", {"Ract": "Bq", "Nn": "1", "tau": "s"}),
    T("fission and fusion", 42, "Splitting heavy nuclei / joining light ones.", "Both move toward the binding-energy peak and release energy.", "Energy = mass lost times c^2."),
    # ---- math words ----------------------------------------------------------------------
    T("exponential decay", 0, "A quantity that falls by the same fraction in each equal time.", "Decay, RC circuits, tunneling all follow it.",
      "Take ln of both sides to solve for time.", "Eq(y, y0*exp(-t/tau))", {"y": "1", "y0": "1", "t": "s", "tau": "s"}),
    T("natural logarithm", 0, "ln, the inverse of e^x.", "Turns exponential equations into linear ones.", "ln(e^a) = a; ln(1/q) = -ln q."),
    T("complex number", 0, "a + i b with i^2 = -1.", "Wave functions are complex; |psi|^2 = psi* psi is real.", "Probability uses the magnitude squared."),
    T("dimensionless", 0, "A number with no units.", "Arguments of exp, sin, log must be dimensionless.", "Check units of every exponent: a unit inside exp is a mistake."),
    T("binomial approximation", 0, "(1 + x)^n ~ 1 + n x for small x.", "Shows relativity reduces to Newton at low speed.", "Small parameter (like v/c) -> expand."),
    T("eigenvalue", 0, "A special value (like an allowed energy) for which an equation has a solution.", "Allowed energies are eigenvalues of the Schrodinger equation.", "Computers find them with a matrix solver."),
]

# ---- decision guide: given -> asked -> use -> avoid ----------------------------------------
DECISIONS = [
    {"given": "photon wavelength", "asked": "photon energy or frequency", "use": "f = c/lambda; E = h f = hc/lambda (E[eV] = 1240/lambda[nm])",
     "avoid": "h times lambda (wrong units); the hydrogen formula", "keywords": {"photon", "wavelength", "energy", "frequency", "nm", "light"}},
    {"given": "light on a metal, stopping potential", "asked": "work function, K_max, or v_max", "use": "K_max = e V_stop; phi = hf - K_max; v = sqrt(2 K_max / m)",
     "avoid": "negative work function; the hydrogen formula; using a voltage that is not the stopping potential",
     "keywords": {"metal", "work", "function", "stopping", "potential", "photoelectric", "electron", "ejected", "kmax"}},
    {"given": "an atom's energy levels or a diagram", "asked": "emitted or absorbed wavelengths", "use": "lambda = hc / |E_i - E_f| for each allowed pair",
     "avoid": "-13.6/n^2 unless the atom is hydrogen", "keywords": {"levels", "diagram", "emitted", "absorbed", "transition", "wavelengths", "spectrum", "atom"}},
    {"given": "events simultaneous in one frame", "asked": "are they simultaneous for a moving observer?", "use": "dt' = gamma (dt - v dx / c^2)",
     "avoid": "the spacetime interval (it is the same in all frames)", "keywords": {"simultaneous", "events", "frame", "observer", "same", "time"}},
    {"given": "a length measured in the object's rest frame", "asked": "length seen by a moving observer", "use": "L = L0 / gamma (shorter)",
     "avoid": "dropping the square in gamma; getting a longer length", "keywords": {"length", "distance", "contraction", "moving", "frame", "ship", "rest"}},
    {"given": "a clock's own (proper) time", "asked": "time in another frame", "use": "dt = gamma dtau (longer)",
     "avoid": "dividing by gamma", "keywords": {"time", "clock", "dilation", "lifetime", "muon", "proper"}},
    {"given": "two velocities in different frames", "asked": "velocity in the other frame", "use": "u' = (u - v)/(1 - u v / c^2) with signs",
     "avoid": "Galilean u - v; ignoring direction", "keywords": {"velocity", "relative", "speed", "rock", "moving", "toward", "away", "addition"}},
    {"given": "a fast particle (K comparable to m c^2)", "asked": "energy, momentum, or speed", "use": "gamma, E = gamma m c^2, E^2 = (pc)^2 + (mc^2)^2",
     "avoid": "(1/2) m v^2 at high speed", "keywords": {"relativistic", "kinetic", "momentum", "mev", "fast", "accelerated", "rest", "energy"}},
    {"given": "a particle's kinetic energy or speed", "asked": "de Broglie wavelength", "use": "p = sqrt(2 m K) (slow) or relativistic p; lambda = h / p",
     "avoid": "using the photon formula E = hc/lambda for a massive particle", "keywords": {"de", "broglie", "wavelength", "electron", "particle", "diffraction", "matter"}},
    {"given": "a potential-energy shape", "asked": "allowed energies or wave functions", "use": "box: n^2; finite well: lower, finite count; parabola: (n - 1/2) hbar omega",
     "avoid": "mixing up the box and oscillator formulas", "keywords": {"box", "well", "potential", "allowed", "energies", "oscillator", "ground", "state"}},
    {"given": "E below a barrier of width w", "asked": "tunneling probability", "use": "eta = hbar / sqrt(2 m (U0 - E)); P ~ exp(-2 w / eta)",
     "avoid": "using E instead of U0 - E", "keywords": {"barrier", "tunneling", "tunnel", "width", "probability", "through"}},
    {"given": "half-life and elapsed time", "asked": "amount left or activity", "use": "N = N0 (1/2)^(t / t_half); R = N / tau",
     "avoid": "subtracting a fixed amount per half-life", "keywords": {"half", "life", "decay", "remaining", "activity", "sample", "radioactive"}},
    {"given": "nuclear or atomic masses", "asked": "binding energy or energy released", "use": "mass difference times c^2 (931.5 MeV per u)",
     "avoid": "mixing nuclear and atomic masses", "keywords": {"binding", "mass", "nucleus", "released", "fission", "fusion", "defect", "u"}},
    {"given": "a charge in crossed E and B fields", "asked": "speed that passes undeflected", "use": "qE = qvB -> v = E/B",
     "avoid": "adding E and B (different units)", "keywords": {"crossed", "fields", "undeflected", "selector", "magnetic", "electric", "charge"}},
]


# ---- dimension checking --------------------------------------------------------------------
def _dims(expr):
    return SI.get_dimension_system().get_dimensional_dependencies(SI._collect_factor_and_dimension(expr)[1])


def _check_functions(expr):
    for f in expr.atoms(sp.Function):
        if isinstance(f, sp.Abs):
            continue
        for arg in f.args:
            if _dims(arg):
                raise ValueError(f"argument of {f.func.__name__} has units {_dims(arg)}: must be dimensionless")


def check_dimensions(term):
    """True if the term's equation is dimensionally consistent; raises ValueError with the reason otherwise."""
    if not term.equation:
        return True
    syms = {name: sp.Symbol(name) for name in term.units}
    expr = sp.sympify(term.equation, locals={**syms, "Rational": sp.Rational})
    missing = {s.name for s in expr.free_symbols} - set(term.units)
    if missing:
        raise ValueError(f"{term.term}: no units given for {sorted(missing)}")
    # magnitude symbol x unit, so that x - v*t can't cancel to a bare 0 (1 m - 1 m) and hide its units
    sub = {syms[k]: sp.Symbol(f"{k}_val", positive=True) * sp.sympify(v, locals=UNITS) for k, v in term.units.items()}
    # per side (substituting into Eq would evaluate it), and xreplace (one pass) so the mass symbol `m`
    # is never confused with the meter unit, whose abbreviation is also "m"
    lhs, rhs = expr.lhs.xreplace(sub), expr.rhs.xreplace(sub)
    try:
        for side in (lhs, rhs):
            _check_functions(side)
        dl, dr = _dims(lhs), _dims(rhs)          # SymPy itself raises if a sum mixes units
    except ValueError as err:
        raise ValueError(f"{term.term}: {err}") from None
    if dl != dr:
        raise ValueError(f"{term.term}: left side {dl} != right side {dr}")
    return True


# ---- lookup, decisions, flashcards --------------------------------------------------------
def by_chapter(ch):
    if ch not in CHAPTERS:
        raise ValueError(f"chapter must be one of {sorted(CHAPTERS)}, got {ch}")
    return [t for t in VOCAB if t.chapter == ch]


def lookup(word):
    w = word.lower().strip()
    hits = [t for t in VOCAB if w in t.term.lower()]
    if not hits:
        raise KeyError(f"no vocabulary entry matching {word!r}")
    return hits


def decide(question, top=3):
    """Rank decision rules for a word problem by keyword overlap. Returns [(score, rule)]."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    scored = sorted(((len(words & r["keywords"]), i, r) for i, r in enumerate(DECISIONS)), key=lambda s: (-s[0], s[1]))
    return [(s, r) for s, _, r in scored[:top] if s > 0]


def flashcards(chapter=None, n=10, seed=0):
    pool = VOCAB if chapter is None else by_chapter(chapter)
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def to_markdown():
    """A printable study sheet: one table per chapter, then the decision guide."""
    out = ["# Modern physics vocabulary (Knight Ch. 36-42)", "",
           "Each equation is checked for dimensional consistency in `tests/test_modern_physics_vocab.py`.", ""]
    for ch, name in CHAPTERS.items():
        terms = by_chapter(ch)
        if not terms:
            continue
        out += [f"## {'Ch. ' + str(ch) + ': ' if ch else ''}{name}", "", "| Term | English | Math | Science | Decide |", "|---|---|---|---|---|"]
        for t in terms:
            eq = t.equation.replace("Eq(", "").replace("Ge(", "").rstrip(")").replace(", ", " = ", 1) if t.equation else ""
            if t.equation.startswith("Ge("):
                eq = eq.replace(" = ", " >= ", 1)
            cells = [t.term, t.english, f"`{eq}`" if eq else "", t.science, t.decide]
            out.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
        out.append("")
    out += ["## Decisions: given -> asked -> use -> avoid", "", "| Given | Asked | Use | Avoid |", "|---|---|---|---|"]
    out += [f"| {r['given']} | {r['asked']} | {r['use']} | {r['avoid']} |" for r in DECISIONS]
    return "\n".join(out) + "\n"
