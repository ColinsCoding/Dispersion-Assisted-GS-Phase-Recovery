"""PHYS 106 free-response practice set (Knight Ch. 36-42 topics) with a unit-aware grader.

The problems are original (written for this repo, not taken from Knight or any exam). Each one
has numeric parts, which ``grade`` checks with units, and conceptual parts, which come with a
model answer and the key ideas a grader looks for.

    from dgs import phys106_frq as F
    F.grade("photoelectric", "b", "1.7 eV")
    F.grade("velocity_addition", "a", "1.3 c")      # flags the Galilean mistake

Answers are strings: a number then a unit, such as "0.10 nm", "3.2e-8 s", "0.92 c", "1.70 V",
"640 MeV/c". Dimensionless answers need no unit. The tolerance is 2 percent unless a part says
otherwise, because exam answers are usually rounded to 2-3 significant figures.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.physics import units as u

# ---- constants (CODATA 2018 exact or recommended) --------------------------------------------
C = 2.99792458e8            # m/s
H = 6.62607015e-34          # J s
HBAR = H / (2 * math.pi)
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31      # kg
ME_C2_EV = 0.51099895e6     # eV
MP_C2_MEV = 938.27208816    # MeV
HC_EV_NM = H * C / E_CHARGE * 1e9        # 1239.84 eV nm
HBARC_EV_NM = HC_EV_NM / (2 * math.pi)  # 197.327 eV nm
RYD_EV = 13.605693122994

TEMPLATE = """\
1. Given    - list every number with its unit and symbol.
2. Find     - what is asked, with the unit it should come out in.
3. Physics  - name the principle (time dilation, energy conservation, ...) and why it applies.
4. Equation - write it symbolically first.
5. Solve    - algebra in symbols, then put numbers in WITH units.
6. Check    - units come out right? size reasonable? v < c? probability between 0 and 1?
"""

_UNITS = {
    "eV": u.electronvolt, "keV": 1e3 * u.electronvolt, "MeV": 1e6 * u.electronvolt,
    "GeV": 1e9 * u.electronvolt, "J": u.joule, "V": u.volt, "W": u.watt,
    "m": u.meter, "cm": u.centimeter, "mm": u.millimeter, "um": u.micrometer,
    "nm": u.nanometer, "pm": u.picometer, "fm": 1e-15 * u.meter, "km": u.kilometer,
    "s": u.second, "ms": u.millisecond, "us": u.microsecond, "ns": u.nanosecond,
    "h": u.hour, "hr": u.hour, "d": u.day, "days": u.day, "day": u.day, "yr": u.year,
    "kg": u.kilogram, "c": u.speed_of_light, "Hz": u.hertz,
}
_BASE = [u.kilogram, u.meter, u.second, u.ampere]


def quantity(text):
    """Parse '1.70 eV', '0.92 c', '640 MeV/c', '3e-8 s', or a bare number into a SymPy quantity."""
    text = re.sub(r"\s*[x×*]\s*10\^", "e", text.strip()).replace("μ", "u")
    m = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$", text)
    if not m:
        raise ValueError(f"could not read a number at the start of {text!r}")
    value, unit_text = float(m.group(1)), m.group(2).strip()
    if not unit_text:
        return sp.Float(value)
    names = re.findall(r"[A-Za-z]+", unit_text)
    unknown = [n for n in names if n not in _UNITS]
    if unknown:
        raise ValueError(f"unknown unit(s) {unknown}; known: {sorted(_UNITS)}")
    expr = sp.sympify(unit_text.replace("^", "**"), locals={k: v for k, v in _UNITS.items()})
    return value * expr


def _ratio(a, b):
    """a / b as a plain float, or None when the dimensions differ."""
    r = u.convert_to(a / b, _BASE)
    r = sp.nsimplify(r) if r.is_number else sp.simplify(r)
    return float(r) if r.is_number else None


@dataclass
class Part:
    label: str
    prompt: str
    answer: str = ""                     # numeric parts: e.g. "1.70 eV"; empty for conceptual
    solution: str = ""
    points: int = 2
    tol: float = 0.02
    mistakes: list = field(default_factory=list)   # [(wrong answer string, feedback)]
    key_ideas: list = field(default_factory=list)  # conceptual parts: what earns credit

    @property
    def numeric(self):
        return bool(self.answer)


@dataclass
class Problem:
    pid: str
    chapter: int
    title: str
    setup: str
    parts: list

    def part(self, label):
        for p in self.parts:
            if p.label == label:
                return p
        raise KeyError(f"{self.pid} has no part {label!r}; parts: {[p.label for p in self.parts]}")

    @property
    def points(self):
        return sum(p.points for p in self.parts)


def _g(x, sig=4):
    return f"{x:.{sig}g}"


def _build():
    P = []

    # 1. time dilation (Ch 36)
    tau, beta = 2.6e-8, 0.95
    gam = 1 / math.sqrt(1 - beta**2)
    t_lab = gam * tau
    d_lab = beta * C * t_lab
    P.append(Problem("time_dilation", 36, "An unstable particle in a beam",
        f"A particle with proper lifetime {tau:g} s moves through the lab at {beta}c.",
        [Part("a", "Find the Lorentz factor gamma.", _g(gam),
              f"gamma = 1/sqrt(1 - beta^2) = 1/sqrt(1 - {beta}^2) = {gam:.3f}",
              mistakes=[(_g(math.sqrt(1 - beta**2)), "That is 1/gamma. gamma is always >= 1.")]),
         Part("b", "Find its lifetime measured in the lab.", f"{_g(t_lab)} s",
              f"The proper time is in the particle's frame, so the lab sees it dilated: "
              f"dt = gamma dtau = {gam:.3f} x {tau:g} s = {t_lab:.3e} s",
              mistakes=[(f"{_g(tau / gam)} s", "You divided by gamma. Moving clocks run SLOW, so the lab time is longer.")]),
         Part("c", "How far does it travel in the lab before decaying?", f"{_g(d_lab)} m",
              f"d = v dt = {beta} x c x {t_lab:.3e} s = {d_lab:.2f} m",
              mistakes=[(f"{_g(beta * C * tau)} m", "You used the proper lifetime with the lab speed. Mixing frames.")]),
         Part("d", "In the particle's own frame it only lives 2.6e-8 s. Explain how it still covers the lab distance.",
              solution="In its frame the lab moves past at 0.95c and the lab distance is length-contracted by "
                       "1/gamma. The contracted distance divided by 0.95c equals the proper lifetime. Both frames "
                       "agree it reaches the same mark; they disagree on why (dilation vs contraction).",
              key_ideas=["length contraction", "lab distance is shorter in the particle frame",
                         "both frames agree on the event (decay at the same mark)"])]))

    # 2. velocity addition (Ch 36)
    v, up = 0.60, 0.70
    w = (v + up) / (1 + v * up)
    P.append(Problem("velocity_addition", 36, "A probe fired from a ship",
        f"A ship moves at {v}c relative to a station and fires a probe forward at {up}c relative to the ship.",
        [Part("a", "Find the probe's speed relative to the station, as a fraction of c.", f"{_g(w)} c",
              f"u = (u' + v)/(1 + u'v/c^2) = ({up} + {v})/(1 + {up}x{v}) = {w:.4f}c",
              mistakes=[(f"{v + up} c", "Galilean addition. The answer exceeds c, which is a sign it's wrong.")]),
         Part("b", "Show your answer is less than c, and say what happens if the probe were a light pulse.",
              solution="With u' = c: (c + v)/(1 + v/c) = c exactly. Light moves at c in every frame (2nd postulate).",
              key_ideas=["u' = c gives u = c", "second postulate"])]))

    # 3. relativistic energy (Ch 36)
    K = 500.0
    gam3 = 1 + K / MP_C2_MEV
    b3 = math.sqrt(1 - 1 / gam3**2)
    E3 = gam3 * MP_C2_MEV
    p3 = math.sqrt(E3**2 - MP_C2_MEV**2)
    P.append(Problem("relativistic_energy", 36, "A fast proton",
        f"A proton (mc^2 = {MP_C2_MEV:.2f} MeV) has kinetic energy {K:g} MeV.",
        [Part("a", "Find gamma.", _g(gam3), f"K = (gamma - 1) mc^2, so gamma = 1 + K/mc^2 = {gam3:.4f}"),
         Part("b", "Find its speed as a fraction of c.", f"{_g(b3)} c", f"beta = sqrt(1 - 1/gamma^2) = {b3:.4f}",
              mistakes=[(f"{_g(math.sqrt(2 * K / MP_C2_MEV))} c", "That is K = mv^2/2, the Newtonian formula. At K ~ mc^2 it fails.")]),
         Part("c", "Find its momentum in MeV/c.", f"{_g(p3)} MeV/c",
              f"E = K + mc^2 = {E3:.1f} MeV; (pc)^2 = E^2 - (mc^2)^2, so p = {p3:.1f} MeV/c",
              mistakes=[(f"{_g(math.sqrt(2 * MP_C2_MEV * K))} MeV/c", "Newtonian p = sqrt(2mK). Use E^2 = (pc)^2 + (mc^2)^2.")])]))

    # 4. photoelectric (Ch 38)
    W0, lam = 2.30, 310.0
    Eph = HC_EV_NM / lam
    Kmax = Eph - W0
    P.append(Problem("photoelectric", 38, "Light on a metal",
        f"Light of wavelength {lam:g} nm shines on a metal with work function {W0} eV.",
        [Part("a", "Find the photon energy in eV.", f"{_g(Eph)} eV", f"E = hc/lambda = 1240 eV nm / {lam:g} nm = {Eph:.3f} eV",
              mistakes=[(f"{_g(H * lam * 1e-9 / E_CHARGE)} eV", "E = h lambda has the wrong units (J s m). It is E = hc/lambda.")]),
         Part("b", "Find the maximum kinetic energy of the electrons.", f"{_g(Kmax)} eV",
              f"Kmax = E - W0 = {Eph:.3f} - {W0} = {Kmax:.3f} eV",
              mistakes=[(f"{_g(Eph + W0)} eV", "The work function is energy the electron LOSES escaping; subtract it.")]),
         Part("c", "Find the stopping potential.", f"{_g(Kmax)} V", f"e V_stop = Kmax, so V_stop = {Kmax:.3f} V"),
         Part("d", "Find the longest wavelength that ejects electrons.", f"{_g(HC_EV_NM / W0)} nm",
              f"Threshold: hc/lambda0 = W0, so lambda0 = 1240/{W0} = {HC_EV_NM / W0:.0f} nm"),
         Part("e", "The intensity is doubled at the same wavelength. What happens to Kmax and to the current?",
              solution="Kmax is unchanged (it depends only on photon energy). The current doubles, since twice as "
                       "many photons per second eject twice as many electrons.",
              key_ideas=["Kmax unchanged", "current increases", "intensity = number of photons"])]))

    # 5. de Broglie (Ch 39)
    V5 = 150.0
    lam5 = HC_EV_NM / math.sqrt(2 * ME_C2_EV * V5)
    P.append(Problem("de_broglie", 39, "An electron beam",
        f"An electron starts from rest and is accelerated through {V5:g} V.",
        [Part("a", "Find its kinetic energy in eV.", f"{V5:g} eV", f"K = eV = {V5:g} eV"),
         Part("b", "Find its de Broglie wavelength.", f"{_g(lam5)} nm",
              f"p = sqrt(2mK), lambda = h/p = hc/sqrt(2 mc^2 K) = 1240/sqrt(2 x 511000 x {V5:g}) = {lam5:.4f} nm",
              mistakes=[(f"{_g(HC_EV_NM / V5)} nm", "lambda = hc/E is for photons only. Electrons use lambda = h/p.")]),
         Part("c", "Would this beam diffract from a crystal with 0.2 nm atomic spacing? Why?",
              solution="Yes: the wavelength (~0.1 nm) is comparable to the spacing, which is the condition for "
                       "strong diffraction (Davisson-Germer).",
              key_ideas=["wavelength comparable to spacing", "Davisson-Germer / electron diffraction"])]))

    # 6. Bohr hydrogen (Ch 41/42)
    n_hi, n_lo = 4, 2
    E6 = RYD_EV * (1 / n_lo**2 - 1 / n_hi**2)
    P.append(Problem("hydrogen", 38,"A hydrogen emission line",
        f"A hydrogen atom drops from n = {n_hi} to n = {n_lo}. Energies are E_n = -13.6 eV / n^2.",
        [Part("a", "Find the photon energy.", f"{_g(E6)} eV", f"E = 13.6 (1/{n_lo**2} - 1/{n_hi**2}) = {E6:.3f} eV",
              mistakes=[(f"{_g(RYD_EV * (1 / n_lo - 1 / n_hi))} eV", "Energy goes as 1/n^2, not 1/n.")]),
         Part("b", "Find the wavelength.", f"{_g(HC_EV_NM / E6)} nm", f"lambda = 1240/{E6:.3f} = {HC_EV_NM / E6:.1f} nm"),
         Part("c", "How much energy ionizes the atom from n = 2?", f"{_g(RYD_EV / 4)} eV", "E = 0 - (-13.6/4) = 3.40 eV"),
         Part("d", "What color is the line, and is it in the Balmer series? Why?",
              solution="About 486 nm, blue-green. Yes: Balmer lines all end on n = 2.",
              key_ideas=["blue-green / visible", "ends on n = 2"])]))

    # 7. particle in a box (Ch 40)
    L7 = 0.50
    E1 = (HC_EV_NM / (2 * L7))**2 / (2 * ME_C2_EV)
    p_mid = 1 / 3 + (math.sin(2 * math.pi / 3) - math.sin(4 * math.pi / 3)) / (2 * math.pi)
    P.append(Problem("particle_in_box", 40, "An electron in a box",
        f"An electron is in a rigid box of length {L7} nm.",
        [Part("a", "Find the ground-state energy.", f"{_g(E1)} eV",
              f"E_n = n^2 h^2/(8mL^2) = n^2 (hc)^2/(8 mc^2 L^2); E1 = 1240^2/(8 x 511000 x {L7}^2) = {E1:.3f} eV"),
         Part("b", "Find the wavelength of the photon emitted from n = 2 to n = 1.", f"{_g(HC_EV_NM / (3 * E1))} nm",
              f"dE = (4 - 1)E1 = {3 * E1:.3f} eV; lambda = 1240/dE = {HC_EV_NM / (3 * E1):.0f} nm",
              mistakes=[(f"{_g(HC_EV_NM / E1)} nm", "You used E1. The photon carries the DIFFERENCE E2 - E1 = 3E1.")]),
         Part("c", "In the ground state, what is the probability of finding the electron in the middle third of the box?",
              _g(p_mid), "Integrate (2/L) sin^2(pi x/L) from L/3 to 2L/3: 1/3 + sqrt(3)/(2 pi) = 0.609",
              mistakes=[("0.3333", "1/3 is the classical answer. The n = 1 wave function peaks in the middle.")]),
         Part("d", "Sketch psi_2. Where is the electron never found, and how does that differ from a classical particle?",
              solution="psi_2 is one full sine hump up and one down, with a node at x = L/2. The probability density is "
                       "zero at the walls and at the center. A classical particle is equally likely anywhere.",
              key_ideas=["node at L/2", "zero at the walls", "classical is uniform"])]))

    # 8. tunneling (Ch 40)
    E8, U8, w8 = 3.0, 5.0, 0.40
    eta = HBARC_EV_NM / math.sqrt(2 * ME_C2_EV * (U8 - E8))
    Pt = math.exp(-2 * w8 / eta)
    P.append(Problem("tunneling", 40, "An electron at a barrier",
        f"A {E8:g} eV electron meets a {U8:g} eV barrier {w8} nm wide.",
        [Part("a", "Find the penetration distance eta.", f"{_g(eta)} nm",
              f"eta = hbar/sqrt(2m(U - E)) = 197.3 eV nm / sqrt(2 x 511000 x {U8 - E8:g}) = {eta:.4f} nm",
              mistakes=[(f"{_g(HBARC_EV_NM / math.sqrt(2 * ME_C2_EV * U8))} nm", "Use U - E, the energy deficit, not U.")]),
         Part("b", "Estimate the tunneling probability.", _g(Pt), f"P = exp(-2w/eta) = exp(-{2 * w8 / eta:.3f}) = {Pt:.2e}",
              tol=0.05, mistakes=[(_g(math.exp(-w8 / eta)), "The exponent is -2w/eta: |psi|^2 doubles the exponent.")]),
         Part("c", "If the width doubles, by what factor does P change?", _g(Pt), "P' = exp(-4w/eta) = P^2, so it changes by a factor of P.",
              tol=0.05),
         Part("d", "Why does tunneling make no sense classically?",
              solution="Classically K = E - U would be negative inside the barrier, which is impossible. In QM psi "
                       "decays exponentially instead of vanishing, so it has a nonzero value on the far side.",
              key_ideas=["negative kinetic energy classically", "exponential decay inside", "nonzero on the far side"])]))

    # 9. uncertainty (Ch 39)
    dx = 0.10e-9
    dv = HBAR / (2 * M_E * dx)
    P.append(Problem("uncertainty", 39, "A confined electron",
        f"An electron is confined to a region {dx * 1e9:.2f} nm wide (about an atom).",
        [Part("a", "Find the minimum uncertainty in its velocity.", f"{_g(dv)} m/s",
              f"dx dp >= hbar/2, so dv >= hbar/(2 m dx) = {dv:.2e} m/s",
              mistakes=[(f"{_g(H / (M_E * dx))} m/s", "That uses h instead of hbar/2, a factor 4 pi too big. Check which form your course uses.")]),
         Part("b", "Is a 'stationary electron at a point' possible? Explain.",
              solution="No: dx -> 0 forces dp -> infinity, so an electron cannot be both localized and at rest. "
                       "This is also why the ground-state energy of a box is not zero.",
              key_ideas=["dx -> 0 means dp -> infinity", "zero-point energy"])]))

    # 10. half-life (Ch 42)
    th, t10 = 8.02, 3.0
    P.append(Problem("half_life", 42, "A medical isotope",
        f"An isotope has a half-life of {th} days.",
        [Part("a", f"What fraction remains after {t10:g} days?", _g(2 ** (-t10 / th)), f"N/N0 = (1/2)^(t/t1/2) = 2^(-{t10:g}/{th}) = {2 ** (-t10 / th):.3f}",
              mistakes=[(_g(1 - t10 / (2 * th)), "Decay is exponential, not linear.")]),
         Part("b", "How long until 10 percent remains?", f"{_g(th * math.log2(10))} days",
              f"t = t1/2 x log2(10) = {th} x 3.32 = {th * math.log2(10):.1f} days"),
         Part("c", "Find the decay constant in s^-1.", f"{_g(math.log(2) / (th * 86400))} s^-1",
              f"lambda = ln 2 / t1/2 = 0.693/({th} x 86400 s) = {math.log(2) / (th * 86400):.3e} /s")]))
    return {p.pid: p for p in P}


PROBLEMS = _build()


def grade(pid, label, response):
    """Grade one numeric part. Returns dict(correct, points, of, rel_error, feedback)."""
    part = PROBLEMS[pid].part(label) if pid in PROBLEMS else None
    if part is None:
        raise KeyError(f"unknown problem {pid!r}; choose from {list(PROBLEMS)}")
    out = {"correct": False, "points": 0, "of": part.points, "rel_error": None}
    if not part.numeric:
        out["feedback"] = "Conceptual part: compare with the model answer. Credit for: " + "; ".join(part.key_ideas)
        return out
    if not response or not response.strip():
        out["feedback"] = "No answer yet."
        return out
    try:
        got = quantity(response)
    except ValueError as err:
        out["feedback"] = str(err)
        return out
    want = quantity(part.answer)
    r = _ratio(got, want)
    if r is None:
        out["feedback"] = f"Wrong units: expected something like {part.answer}."
        return out
    out["rel_error"] = abs(r - 1)
    if out["rel_error"] <= part.tol:
        out.update(correct=True, points=part.points, feedback="Correct.")
        return out
    for wrong, why in part.mistakes:
        rw = _ratio(got, quantity(wrong))
        if rw is not None and abs(rw - 1) <= part.tol:
            out["feedback"] = "Common mistake: " + why
            return out
    out["points"] = 1 if out["rel_error"] <= 0.10 else 0      # setup likely right, arithmetic off
    out["feedback"] = ("Close: check rounding and arithmetic." if out["points"]
                       else "Not matching. Recheck the physics in the setup step.")
    return out


def grade_problem(pid, responses):
    """Grade all numeric parts of a problem from {label: answer}. Returns (score, max, rows)."""
    prob = PROBLEMS[pid]
    rows = [(p.label, responses.get(p.label, ""), grade(pid, p.label, responses.get(p.label, "")))
            for p in prob.parts if p.numeric]
    return sum(r[2]["points"] for r in rows), sum(p.points for p in prob.parts if p.numeric), rows


def practice_exam(n=4, seed=None, chapters=None):
    """Pick n problems, optionally only from the given chapters."""
    pool = [p for p in PROBLEMS.values() if chapters is None or p.chapter in chapters]
    if n > len(pool):
        raise ValueError(f"only {len(pool)} problems available for chapters {chapters}")
    return random.Random(seed).sample(pool, n)
