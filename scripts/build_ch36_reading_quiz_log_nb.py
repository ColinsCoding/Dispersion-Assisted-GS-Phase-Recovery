"""Build notebooks/ch36_reading_quiz_log.ipynb -- "Chapter 36 Reading
Quizzes: a Study Log with Computations".

Documentation notebook: each reading-quiz question is recorded with the
attempts made, the status (confirmed / unresolved), the reasoning, and a
short SymPy or NumPy computation that illustrates the idea being tested.
New questions are added by appending to the QUIZ_LOG list in the first
code cell and rebuilding.

Needs: sympy, numpy.

Build:   py -3.13 scripts/build_ch36_reading_quiz_log_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/ch36_reading_quiz_log.ipynb
(or: make notebook-ch36-quiz-log)
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md(r"""# Chapter 36 Reading Quizzes — Study Log

A record of each reading-quiz question: what was asked, what was tried,
what is known for sure, and a small computation that shows the idea the
question is testing. It complements the worked chapter notebook
[`ch36_relativity.ipynb`](ch36_relativity.ipynb).

**Status key**

| Status | Meaning |
|---|---|
| confirmed | accepted as correct by the quiz system |
| expected | follows directly from the textbook's own definition; not yet submitted |
| unresolved | attempts so far were marked incorrect; the answer is still open |

To add a question, append an entry to `QUIZ_LOG` below, then rebuild the
notebook (`make notebook-ch36-quiz-log`)."""))

cells.append(co(r"""import numpy as np
import sympy as sp

QUIZ_LOG = [
    {
        "quiz": "36.02",
        "question": "Special relativity requires us to change our ideas about ____. (select all that apply)",
        "options": ["space", "time", "mass", "energy", "conservation laws"],
        "attempts_marked_incorrect": [
            ["space", "time", "mass", "energy"],
            ["space", "time", "mass", "energy", "conservation laws"],
            ["space", "time"],
        ],
        "status": "unresolved",
        "answer": None,
        "note": "The quiz checks one sentence in the chapter introduction / Section 36.1. Search the eText "
                "for 'change our ideas' before using another attempt.",
    },
    {
        "quiz": "36.03",
        "question": "The fundamental element of relativity is ____.",
        "options": ["a reference frame", "spacetime coordinates", "an event", "time dilation", "a clock"],
        "attempts_marked_incorrect": [],
        "status": "confirmed",
        "answer": ["an event"],
        "note": "Section 36.3 (Events and Measurements) names the event as the fundamental entity of relativity.",
    },
    {
        "quiz": "36.04",
        "question": "It's essential that all experimenters in a reference frame ____.",
        "options": ["synchronize their clocks", "use the same clock", "calibrate their meter sticks",
                    "communicate only with light waves", "be equally spaced"],
        "attempts_marked_incorrect": [],
        "status": "confirmed",
        "answer": ["synchronize their clocks"],
        "note": "An event is timed by the clock at the event, so every clock in the frame must agree. "
                "One shared clock cannot be at distant events.",
    },
    {
        "quiz": "36.05",
        "question": "The story of Peggy and Ryan, illustrated in Figure 36.15, illustrates ____.",
        "options": ["length contraction", "time dilation", "the Lorentz transformations",
                    "the relativity of space", "the relativity of simultaneity"],
        "attempts_marked_incorrect": [],
        "status": "confirmed",
        "answer": ["the relativity of simultaneity"],
        "note": "Two firecrackers are simultaneous in Ryan's (ground) frame but not in Peggy's (train) frame.",
    },
    {
        "quiz": "36.06",
        "question": "Time dilation is discovered using ____.",
        "options": ["a swinging pendulum", "an oscillating atom", "a light clock", "a pulsar", "traveling twins"],
        "attempts_marked_incorrect": [],
        "status": "expected",
        "answer": ["a light clock"],
        "note": "A light pulse bouncing between mirrors: seen moving, it takes a longer diagonal path at speed c.",
    },
]

for entry in QUIZ_LOG:
    print(f"Quiz {entry['quiz']}  [{entry['status']}]")
    print(f"  {entry['question']}")
    print(f"  answer: {', '.join(entry['answer']) if entry['answer'] else '(not yet known)'}")
    for attempt in entry["attempts_marked_incorrect"]:
        print(f"  marked incorrect: {', '.join(attempt)}")
    print(f"  note: {entry['note']}\n")"""))

cells.append(co(r"""# what is still possible for 36.02: every combination not already ruled out
from itertools import combinations

entry = QUIZ_LOG[0]
ruled_out = [set(attempt) for attempt in entry["attempts_marked_incorrect"]]
remaining = [set(combo) for size in range(1, 6) for combo in combinations(entry["options"], size)
             if set(combo) not in ruled_out]
with_space_and_time = [combo for combo in remaining if {"space", "time"} <= combo]
print(f"{len(remaining)} of 31 possible selections remain; "
      f"{len(with_space_and_time)} of them include both space and time:")
for combo in sorted(with_space_and_time, key=len):
    print("  ", ", ".join(option for option in entry["options"] if option in combo))
assert len(remaining) == 31 - len(ruled_out)"""))

# ================================================================ 36.02
cells.append(md(r"""# Quiz 36.02 — what changes, computed

The physics behind each option, at $v = 0.8c$ where
$\gamma = 1/\sqrt{1 - 0.8^2} = 5/3$. Physically all five are affected in
some way — which is why the quiz must be checking the book's specific
wording rather than the physics alone."""))

cells.append(co(r"""beta = sp.Rational(4, 5)
gamma = 1 / sp.sqrt(1 - beta**2)
print("gamma at 0.8c =", gamma)

# space: a 1 m stick moving along its length
stick = 1 / gamma
# time: round trip to a star 4.0 light-years away
earth_years, ship_years = 2 * 4 / beta, 2 * 4 / beta / gamma
# mass: two lumps of mass m at +/-0.6c stick together (c = m = 1)
u = sp.Rational(3, 5)
lump_mass = 2 / sp.sqrt(1 - u**2)
# energy: kinetic energy at 0.8c, relativistic vs classical (units of m c^2)
K_rel, K_classical = gamma - 1, beta**2 / 2
# conservation laws: momentum of that collision seen from a frame moving at 0.8c
add = lambda a, b: (a - b) / (1 - a * b)
G = lambda w: 1 / sp.sqrt(1 - w**2)
pA, pB = add(u, beta), add(-u, beta)
p_before = G(pA) * pA + G(pB) * pB
p_after = G(-beta) * lump_mass * (-beta)

print(f"space : moving 1 m stick measures {stick} m = {float(stick):.2f} m")
print(f"time  : Earth {earth_years} yr, traveller {ship_years} yr")
print(f"mass  : merged lump has mass {lump_mass} m, not 2 m")
print(f"energy: K = {K_rel} m c^2 relativistic vs {K_classical} m c^2 classical")
print(f"conservation laws: p_before = {sp.simplify(p_before)}, p_after = {sp.simplify(p_after)} "
      f"(conserved only with p = gamma m u)")
assert stick == sp.Rational(3, 5) and (earth_years, ship_years) == (10, 6) and lump_mass == sp.Rational(5, 2)
assert sp.simplify(p_before - p_after) == 0"""))

# ================================================================ 36.03
cells.append(md(r"""# Quiz 36.03 — the event

**Answer: an event.** An event is something that happens at one place
and one instant. It is physical: a firecracker exploding is an event
whether or not anyone is watching. Everything else in the option list
describes or follows from events:

| Option | Its role |
|---|---|
| reference frame | a coordinate system, with clocks, for *labelling* events |
| spacetime coordinates | the labels $(x, y, z, t)$ a frame gives an event |
| a clock | the instrument that reads the $t$ label |
| time dilation | a *result* about time intervals between two events |
| **an event** | **the thing being labelled — the fundamental element** |

The computation shows why the event is the fundamental thing: two
frames give the same event **different coordinates**, yet agree that it
happened, and agree on the spacetime interval between any two events."""))

cells.append(co(r"""c, v = sp.symbols("c v", positive=True)
x, t = sp.symbols("x t", real=True)
g = 1 / sp.sqrt(1 - v**2 / c**2)


def coordinates_in_moving_frame(x_event, t_event):
    return g * (x_event - v * t_event), g * (t_event - v * x_event / c**2)


# one event: a firecracker at x = 600 m, t = 1 microsecond, seen from a train at 0.6c
C = 299_792_458
values = {v: sp.Rational(3, 5) * C, c: C}
x_event, t_event = 600, sp.Rational(1, 10**6)
x_train, t_train = [sp.N(q.subs(values)) for q in coordinates_in_moving_frame(x_event, t_event)]
print(f"ground frame : x = {x_event} m,  t = {float(t_event) * 1e6:.3f} us")
print(f"train frame  : x' = {float(x_train):.1f} m,  t' = {float(t_train) * 1e6:.3f} us")
print("different labels -- the same event")

# both frames agree on the interval between two events
x1, t1, x2, t2 = sp.symbols("x1 t1 x2 t2", real=True)
xp1, tp1 = coordinates_in_moving_frame(x1, t1)
xp2, tp2 = coordinates_in_moving_frame(x2, t2)
interval = (c * (t2 - t1))**2 - (x2 - x1)**2
interval_moving = (c * (tp2 - tp1))**2 - (xp2 - xp1)**2
assert sp.simplify(interval_moving - interval) == 0
print("\nthe spacetime interval between any two events is the same in both frames (verified symbolically)")"""))

cells.append(md(r"""# Quiz 36.04 — synchronizing clocks

**Answer: synchronize their clocks.** A reference frame is imagined as
a lattice of meter sticks with a clock at every point, and an event is
timed by the clock **at the event** — no light-travel delay to correct
for. That only works if all the clocks agree.

They cannot be synchronized by carrying them around (moving clocks run
slow) or by reading one clock from afar (light takes time). The
standard method: a light flash leaves the origin at $t = 0$; each clock
at distance $d$ is preset to read $d/c$ and starts when the flash
arrives.

The computation checks the method: after presetting, a second flash
sent *back* from any clock to the origin is timed consistently — the
round trip divides evenly, which is Einstein's definition of
synchronized."""))

cells.append(co(r"""distances = np.array([300.0, 1500.0, 6000.0])       # m, clocks along the x axis
preset = distances / C                              # each clock is preset to d/c

# the flash from the origin (t = 0) reaches clock k at true time d_k / c; clock k starts there,
# showing preset_k, so from then on it reads true time exactly
arrival_true = distances / C
reading_at_arrival = preset
print(" clock at   preset      reads at flash arrival   true time")
for d, p, r_, a in zip(distances, preset, reading_at_arrival, arrival_true):
    print(f"  {d:6.0f} m  {p * 1e6:7.3f} us        {r_ * 1e6:7.3f} us            {a * 1e6:7.3f} us")
assert np.allclose(reading_at_arrival, arrival_true)

# Einstein's check: a flash leaves the origin at t_A, reflects at clock k (reading t_B), returns at t_A2.
# Synchronized means t_B = (t_A + t_A2) / 2.
t_A = 2e-6
t_B = t_A + distances / C                            # clock k's reading on reflection
t_A2 = t_A + 2 * distances / C                       # origin clock on return
assert np.allclose(t_B, (t_A + t_A2) / 2)
print("\nEinstein's round-trip test passes for every clock: reflection time = midpoint of send and return")"""))

cells.append(md(r"""# Quiz 36.05 — Peggy, Ryan, and simultaneity

**Answer: the relativity of simultaneity.** Ryan stands on the ground
midway between two firecrackers that explode at the same instant *in
his frame*; the flashes reach him together. Peggy rides a train past
him. In her frame the two explosions are **not** simultaneous: the one
toward the front of the train happens first.

Applying the Lorentz transformation to two events with $\Delta t = 0$
and separation $\Delta x$:

$$\Delta t' = -\frac{\gamma\, v\, \Delta x}{c^2}.$$

The illustrative numbers below (firecrackers 600 m apart, train at
$0.6c$) are not the textbook's figure values."""))

cells.append(co(r"""dx, dt = sp.symbols("Delta_x Delta_t", real=True)
_, dt_peggy = coordinates_in_moving_frame(dx, dt)
dt_peggy_simultaneous = sp.simplify(dt_peggy.subs(dt, 0))
print("Delta t' (events simultaneous for Ryan) =", dt_peggy_simultaneous)

# rear firecracker at x = -300 m, front at x = +300 m, both at t = 0 in Ryan's frame
_, t_rear = coordinates_in_moving_frame(-300, 0)
_, t_front = coordinates_in_moving_frame(300, 0)
t_rear, t_front = [float(q.subs(values)) for q in (t_rear, t_front)]
print(f"Ryan : both at t = 0")
print(f"Peggy: front at t' = {t_front * 1e6:+.3f} us, rear at t' = {t_rear * 1e6:+.3f} us"
      f" -> front first by {(t_rear - t_front) * 1e6:.3f} us")
assert t_front < 0 < t_rear and abs((t_rear - t_front) * 1e6 - 1.501) < 0.001"""))

cells.append(md(r"""Neither is wrong. Simultaneity is not absolute: whether two separated
events happen "at the same time" depends on the frame. Events at the
**same place** ($\Delta x = 0$) stay simultaneous in every frame, which
is why the effect needs $\Delta x \neq 0$.

# Quiz 36.06 — the light clock, and time dilation at work

**Answer: a light clock.** Two mirrors a distance $h$ apart; a light
pulse bounces between them. In the clock's own frame one round trip
takes $\Delta\tau = 2h/c$. Seen from a frame where the clock moves at
$v$, the pulse follows a longer, diagonal path — still at speed $c$ —
so each tick takes longer: $\Delta t = \gamma\,\Delta\tau$. A light
clock is used because it relies only on the one thing every frame
agrees on, the speed of light."""))

cells.append(co(r"""h, dt2 = sp.symbols("h Delta_t2", positive=True)       # dt2 stands for (Delta t)^2
tick_sq = sp.solve(sp.Eq(c**2 * dt2 / 4, h**2 + v**2 * dt2 / 4), dt2)[0]
ratio_sq = sp.simplify(tick_sq / (2 * h / c)**2)
print("(Delta t / Delta tau)^2 =", ratio_sq)
assert sp.simplify(ratio_sq - g**2) == 0"""))

cells.append(md(r"""## Time dilation and radioactive decay

Radioactive decay is a clock too: after a time $t$ measured in the
sample's own frame, a fraction $2^{-t/T_{1/2}}$ survives. A sample on a
ship moving at $0.8c$ decays according to **ship time**, which is the
proper time.

**Example.** Half-life 3 years. Earth clocks record 5 years; the ship's
clock records $5/\gamma = 3$ years — exactly one half-life."""))

cells.append(co(r"""t_earth, half_life, b = sp.symbols("t_E T_half beta", positive=True)
ship_time = t_earth * sp.sqrt(1 - b**2)
surviving = 2 ** (-ship_time / half_life)
print("surviving fraction =", surviving)

numbers = {t_earth: 5, half_life: 3, b: sp.Rational(4, 5)}
print(f"ship time = {ship_time.subs(numbers)} years")
print(f"surviving fraction = {surviving.subs(numbers)}  (ignoring time dilation: "
      f"{float(2 ** sp.Rational(-5, 3)):.3f})")
assert ship_time.subs(numbers) == 3 and surviving.subs(numbers) == sp.Rational(1, 2)

# Boolean -> polynomial: two nuclei, each surviving with probability p.
# "the sample still has at least one undecayed nucleus" = a OR b  ->  a + b - ab  ->  2p - p^2
p, atom_a, atom_b = sp.symbols("p a b")
at_least_one = sp.Or(atom_a, atom_b)
polynomial = atom_a + atom_b - atom_a * atom_b
for A_, B_ in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    assert bool(at_least_one.subs({atom_a: bool(A_), atom_b: bool(B_)})) == bool(polynomial.subs({atom_a: A_, atom_b: B_}))
p_any = sp.expand(polynomial.subs({atom_a: p, atom_b: p}))
print(f"\nP(at least one of two survives) = {p_any}  ->  at one half-life: {p_any.subs(p, sp.Rational(1, 2))}")
assert p_any.subs(p, sp.Rational(1, 2)) == sp.Rational(3, 4)"""))

cells.append(md(r"""## The same calculation in ANSI C, reporting through `$?`

The program below is strict ANSI C (C89: `/* */` comments, declarations
at the top of each block). It computes $\gamma$, the ship time and the
surviving fraction, and reports success through its **exit status** —
the number a shell stores in `$?`:

| exit status | meaning |
|---|---|
| 0 | computed, self-check passed |
| 1 | bad input (for example $\beta \ge 1$) |
| 2 | self-check failed: $\gamma^2(1 - \beta^2) \ne 1$ |

The cell compiles it with `gcc -std=c89 -pedantic` and runs it twice:
once with valid input and once with $\beta = 1.2$, which must be
rejected."""))

cells.append(co(r"""C_SOURCE = r'''/* gamma_decay.c -- Lorentz factor and time-dilated decay, in ANSI C (C89).
   usage:   gamma_decay beta half_life_years earth_years
   exit status ($?): 0 = ok, 1 = bad input, 2 = self-check failed */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double lorentz_gamma(double beta)
{
    return 1.0 / sqrt(1.0 - beta * beta);
}

int main(int argc, char *argv[])
{
    double beta, half_life, earth_years, gamma, ship_years, remaining, naive;

    if (argc != 4) {
        fprintf(stderr, "usage: %s beta half_life_years earth_years\n", argv[0]);
        return 1;
    }
    beta = atof(argv[1]);
    half_life = atof(argv[2]);
    earth_years = atof(argv[3]);
    if (beta < 0.0 || beta >= 1.0 || half_life <= 0.0 || earth_years < 0.0) {
        fprintf(stderr, "need 0 <= beta < 1, half-life > 0, time >= 0\n");
        return 1;
    }

    gamma = lorentz_gamma(beta);
    ship_years = earth_years / gamma;
    remaining = pow(2.0, -ship_years / half_life);
    naive = pow(2.0, -earth_years / half_life);

    printf("gamma              = %.6f\n", gamma);
    printf("ship (proper) time = %.4f years\n", ship_years);
    printf("fraction remaining = %.4f  (ignoring time dilation: %.4f)\n", remaining, naive);

    if (fabs(gamma * gamma * (1.0 - beta * beta) - 1.0) > 1e-12) {
        return 2;
    }
    return 0;
}
'''

import os, shutil, subprocess, tempfile, pathlib

gcc = shutil.which("gcc") or next((path for path in [r"C:\msys64\mingw64\bin\gcc.exe"] if os.path.exists(path)), None)
if gcc is None:
    print("no gcc found in this kernel; the source above can be compiled by hand with:")
    print("  gcc -std=c89 -pedantic -Wall -Wextra gamma_decay.c -o gamma_decay -lm")
else:
    env = dict(os.environ, PATH=str(pathlib.Path(gcc).parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory() as folder:
        source, program = pathlib.Path(folder) / "gamma_decay.c", pathlib.Path(folder) / "gamma_decay"
        source.write_text(C_SOURCE, encoding="utf-8", newline="\n")
        build = subprocess.run([gcc, "-std=c89", "-pedantic", "-Wall", "-Wextra", str(source), "-o", str(program), "-lm"],
                               capture_output=True, text=True, env=env)
        print("compiler warnings:", build.stderr.strip() or "none")
        assert build.returncode == 0
        for arguments in (["0.8", "3", "5"], ["1.2", "3", "5"]):
            run = subprocess.run([str(program)] + arguments, capture_output=True, text=True, env=env)
            print(f"\n$ ./gamma_decay {' '.join(arguments)}; echo $?")
            print((run.stdout + run.stderr).rstrip())
            print(run.returncode)
            assert run.returncode == (0 if arguments[0] == "0.8" else 1)"""))

cells.append(md(r"""# Lessons for future reading quizzes

- Reading quizzes check the **textbook's wording**. When the physics
  allows several defensible answers (as in 36.02), find the sentence
  the question was written from before spending attempts.
- Definitional questions (as in 36.03) are answered by the section
  title and its first bolded term.
- Record every attempt here, so a later rebuild shows what has been
  ruled out."""))

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "ch36_reading_quiz_log.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}, {len(cells)} cells")
