"""Build notebooks/jalali_workbook_batch01.ipynb (run from the repo root)."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'projects' / 'jalali_workbook'))
from problems_batch01 import P, HDR, S   # noqa: E402

OUT = ROOT / 'notebooks' / 'jalali_workbook_batch01.ipynb'
LEAN_REL = 'projects/jalali_workbook/lean/JalaliWorkbookBatch01.lean'


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip('\n').splitlines(True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip('\n').splitlines(True)}


CATS = {'A': 'Algebra, units, dimensional analysis, scientific notation', 'B': 'Trigonometry, complex numbers, phasors, polar coordinates',
        'C': 'Single-variable calculus', 'D': 'Multivariable calculus', 'E': 'Differential equations and dynamical systems',
        'F': 'Linear algebra', 'G': 'Fourier analysis, sampling, filtering', 'H': 'Probability and statistics',
        'I': 'Rare-event and extreme-value statistics', 'J': 'Photonics', 'K': 'Time-stretch imaging and DFT models',
        'L': 'RF and microwave photonics', 'M': 'Biomedical imaging and cell counting (synthetic data)', 'N': 'Numerical methods, reproducibility',
        'O': 'Research writing, sources, uncertainty, design', 'P': 'Quantum-information connections (only when justified)'}
SECONDARY = {12: 'L', 25: 'L', 30: 'M', 33: 'M', 34: 'M', 38: 'M', 40: 'M', 28: 'K', 31: 'K', 41: 'I', 13: 'D', 21: 'N', 45: 'O'}


def problem_md(p):
    pre = ', '.join(f'JW-{q:03d}' for q in p['pre']) or 'none (entry point)'
    cats = p['cat'] + (f' (also {SECONDARY[p["id"]]})' if p['id'] in SECONDARY else '')
    src = S[p['src']] if p['src'] != 'SYN' else 'synthetic educational model'
    return f"""
#### JW-{p['id']:03d}  ·  category {cats}

- **Problem ID:** JW-{p['id']:03d}
- **Research theme:** {p['theme']}
- **Prerequisites:** {pre}
- **Difficulty:** {p['diff']}
- **Learning objective:** {p['obj']}
- **Physical or mathematical context:** {p['ctx']}
- **Given variables and units:** {p['given']}
- **Unknown variables:** {p['unknown']}
- **Assumptions:** {p['assume']}
- **Mathematical model:** {p['model']}
- **Question:** {p['q']}
- **Expected answer type:** {p['atype']}
- **Common mistake:** {p['mistake']}
- **Verification method:** {p['verify']}
- **Research source or "synthetic educational model":** {src}
"""


cells = [
md(r"""
# Undergraduate Research Workbook: Mathematics and Computation through Time-Stretch Photonics
## Batch 1 of 20: research map, taxonomy, and Problems JW-001 to JW-050

**Prepared by Colin Casey (student-made study material), 2026-10-12.**

> **Not endorsed.** This workbook is an independent educational project. It is **not** produced, reviewed, or endorsed by Prof. Bahram Jalali, the Jalali Lab, or UCLA. The laboratory's published research is used only as motivation; every number not taken from a cited source is labeled **synthetic**.

**How to use it.** Sections 1 to 3 are the research map: who and what the laboratory is, with every claim tied to a source and labeled verified, historical, inferred, or unknown. Section 4 is the plan for 1000 problems. Section 5 has the 50 problems; Section 6 is the answer key, which is **computed** (every answer comes from executing the SymPy code, not typed by hand). Section 7 holds Lean 4 + Mathlib proofs that were **actually compiled**. Sections 8 to 12 give the dependency graph, project ideas, limitations, what this does not prove, and the plan for Batches 2 to 20.

**Status labels used throughout**

| label | meaning |
|---|---|
| **verified** | confirmed in a primary or authoritative source listed in Section 2 (link given) |
| **historical** | verified, but describes something that has ended (a closed award, a past position) |
| **inferred** | suggested by sources (or by secondary summaries) but not stated by an authoritative one |
| **unknown / not publicly verified** | no authoritative source found |
"""),

md(r"""
---
## 1. Verified identity and research status of the laboratory

All sources were consulted on **2026-10-12**.

| claim | status | source |
|---|---|---|
| Bahram Jalali is a **professor emeritus** of Electrical and Computer Engineering at UCLA, with joint appointments in Bioengineering and the California NanoSystems Institute (CNSI) | **verified** | UCLA Samueli profile [U1] |
| The laboratory is the **Jalali Lab** (UCLA Photonics Laboratory), Department of Electrical and Computer Engineering, UCLA Samueli School of Engineering, website photonics.ucla.edu | **verified** | lab website [L1] |
| Ph.D. in applied physics, Columbia University, 1989 | **verified** | [U1], [L2] |
| Physics Research Division, Bell Laboratories, Murray Hill, NJ, **1988 to 1992**, before joining UCLA | **verified** ([U1]); the lab biography [L2] says "until 2002", a **discrepancy**; the UCLA profile is treated as authoritative | [U1], [L2] |
| Elected to the National Academy of Engineering in 2022 | **verified** (UCLA profile; the NAE's own directory was not checked) | [U1] |
| Named to the **Northrop Grumman Endowed Opto-Electronic Chair** (announced June 28, 2011, funded by a $1.16 million Northrop Grumman gift) | **historical** (verified announcement; whether he still holds it is not verified) | [U2] |
| Holder of the **Fang Lu Endowed Chair in Engineering** (UCLA announcement Nov. 10, 2020; chair established 2019) | **historical** (verified 2020 announcement). A secondary search summary indicates another professor was named to this chair in 2025: **inferred**, not checked against a primary source | [U3] |
| Research areas listed on the lab website: radio-frequency-tagged fluorescence imaging (FIRE), generation and control of optical rogue waves, high-throughput optical microscopy for cancer detection, all-dielectric photonic-assisted radio front-end technology | **verified** (as listed on the site) | [L1] |
| Most recent news item on the lab website: Aug. 21, 2023; site copyright 2021 | **verified** | [L1] |
| Whether the laboratory is **currently** running funded research projects | **unknown / not publicly verified** (emeritus status, no active award found in NIH RePORTER or NSF Award Search, and the newest news item is from 2023) | [U1], [L1], Section 2 |

**What "research status" means here.** The person and the laboratory are verified. Current, active funded research is **not publicly verified**. The research themes below are documented by publications, the lab website, and past awards, which is enough for an educational workbook, and is all this workbook relies on.
"""),

md(r"""
---
## 2. Source tables: research themes and funding

### 2a. Research themes

| # | theme | evidence (what the source shows) | source | status |
|---|---|---|---|---|
| 1 | Silicon photonics | first silicon laser credited to Jalali; silicon Raman laser paper | [U1]; [BJ04] | verified ([U1]); bibliographic details of [BJ04] entered from memory, confirm before citing |
| 2 | Microwave photonics and photonic RF receivers | "All-Dielectric Photonic-Assisted Radio Front-End Technology" listed as a research area; 2023 news post on RF communication and sensing | [L1] | verified (website listing) |
| 3 | Time-stretch instruments and dispersive Fourier transformation | UCLA profile credits him with inventing "time stretch"; review articles | [U1]; [GJ13], [MAH17] | verified ([U1]) |
| 4 | High-throughput optical microscopy and biomedical diagnostics | "High-Throughput Optical Microscopy for Cancer Detection" and FIRE listed; two NIH R21 awards (Section 2b) | [L1]; [N1], [N2] | verified |
| 5 | Rare events, extreme-value statistics, optical rogue waves | "Generation and Control of Optical Rogue Waves" listed; NSF award 0925638; *Nature* 2007 paper | [L1]; [F1]; [SRKJ07] | verified |
| 6 | Physics-inspired optical computing and image processing | lab news: PhyCV lecture (June 2023), VEViD low-light enhancement algorithm (Nov. 2022) | [L1] | verified (website news) |
| 7 | Optical communication and analog-to-digital conversion | photonic time-stretch ADC paper; "Best Paper Award, First ADC Forum (IMTC) 2003" on the lab biography; a fiber-optic company founded by Jalali and acquired by Intel appears only in secondary summaries | [HJ03]; [L2] | ADC work verified via [L2] honors list and [HJ03]; the company history is **inferred** (secondary sources only) |

### 2b. Funding

Queries: **NIH RePORTER API v2**, principal-investigator name "Bahram Jalali", all fiscal years; **NSF Award Search API**, PI name "Jalali". Both run 2026-10-12.

| sponsor | award number | dates | project title (as recorded) | investigators | source | status | confidence |
|---|---|---|---|---|---|---|---|
| NIH / NIGMS | 1R21GM107924-01A1; 5R21GM107924-02; 5R21GM107924-03 | 2014-08-01 to 2018-05-31 (FY2014 to FY2016) | "Kilohertz frame rate two-photon and confocal fluorescence microscope enabled by r..." (title truncated in the RePORTER field returned) | B. Jalali (PI) | [N1] | **historical** | high |
| NIH / NIBIB | 1R21EB019645-01A1; 5R21EB019645-02 | 2015-08-01 to 2017-12-31 (FY2015 to FY2016) | "A Frequency-multiplexed flow cytometer for high throughput screening and drug discovery" | D. Di Carlo, B. Jalali (multiple PIs) | [N2] | **historical** | high |
| NSF | 0925638 | 2009-10-01 to 2013-09-30; $300,000 obligated | "Enhanced White Light Sources Using Stimulated Rogue Wave Phenomenon" | B. Jalali (PI) | [F1] | **historical** | high |
| NSF | 9422538 | 1995-06-01 to 1999-05-31; $240,043 obligated | "Transferred-Electron Induced Current Instabilities in Heterojunction Bipolar Transistors" | B. Jalali (PI) | [F2] | **historical** | high |
| Northrop Grumman (industry gift) | none (endowed chair) | announced 2011-06-28; $1.16 million | Northrop Grumman Endowed Opto-Electronic Chair in Electrical Engineering | B. Jalali (chair holder) | [U2] | **historical** (endowment, not a research grant) | high for the gift; current holding unknown |
| Fang Lu (alumnus gift) | none (endowed chair) | chair established 2019; announced 2020-11-10 | Fang Lu Endowed Chair in Engineering | B. Jalali (chair holder, 2020) | [U3] | **historical**; later reassignment **inferred** | high for 2020; low for current status |
| DARPA, ONR, AFOSR, other DoD | not found | not found | not found | not found | no DoD database searched; would require paper acknowledgments or agency records | **unknown / not publicly verified** | none |
| Any **current** federal grant | not found | not found | not found | not found | NIH RePORTER (no records after FY2016), NSF (no active awards) | **not publicly verified** | n/a |

**Name collisions in the NSF results (excluded).** The NSF query also returned awards to *Shirin Jalali* (CAREER 2237538), *Bahar Jalali-Farahani* (0802006, ADC noise cancellation) and *F. Jalali* (8750539, 1987). These are not attributed to Bahram Jalali. The ADC-themed award is a reminder that a matching topic does not establish identity.

### Sources

- [U1] UCLA Samueli School of Engineering, "Bahram Jalali" (profile). https://samueli.ucla.edu/people/bahram-jalali/
- [U2] UCLA Samueli, "Bahram Jalali named to Northrop Grumman Endowed Opto-Electronic Chair in Electrical Engineering," June 28, 2011. https://samueli.ucla.edu/bahram-jalali-named-to-northrop-grumman-endowed-opto-electronic-chair-in-electrical-engineering/
- [U3] UCLA Samueli, "UCLA Samueli announces new endowed faculty chairs," Nov. 10, 2020. https://samueli.ucla.edu/ucla-samueli-announces-new-endowed-faculty-chairs/
- [L1] Jalali Lab website. https://photonics.ucla.edu/
- [L2] Jalali Lab, "Professor Bahram Jalali." https://photonics.ucla.edu/professor-bahram-jalali
- [N1] NIH RePORTER, project R21GM107924. https://reporter.nih.gov/ (search the project number)
- [N2] NIH RePORTER, project R21EB019645. https://reporter.nih.gov/
- [F1] NSF Award Search, award 0925638. https://www.nsf.gov/awardsearch/showAward?AWD_ID=0925638
- [F2] NSF Award Search, award 9422538. https://www.nsf.gov/awardsearch/showAward?AWD_ID=9422538
- Publications [BJ04], [GTJ09], [GJ13], [HJ03], [MAH17], [SRKJ07], [DBGJ13] and textbooks: see Section 13. **Bibliographic details of the publications were entered from memory and should be confirmed before citing.**
"""),

md(r"""
---
## 3. Which funding is verified, historical, inferred, or unknown

- **Verified and historical (high confidence):** the four federal awards in Section 2b. Each was found in the agency's own database, with an award number, dates, title and investigators. All have ended, so they are *historical*. They document that themes 4 (high-throughput biomedical imaging) and 5 (rogue waves) were federally funded research.
- **Verified institutional gifts (historical):** the Northrop Grumman (2011) and Fang Lu (2020) endowed chairs. These are endowments for a faculty position, **not** research grants for a project, and should not be described as funding a specific study.
- **Inferred:** that the Fang Lu chair has since been reassigned (secondary search summary only), and company history in optical communications (secondary sources only).
- **Unknown / not publicly verified:** any Department of Defense funding (DARPA, ONR, AFOSR), any industry research contracts, and **any current grant**. These may exist; this workbook does not claim they do.
- **Search limitations:** PI-name queries miss co-investigator roles, subawards, foundation grants, and non-US funding. NIH RePORTER covers NIH (and some HHS) awards only; NSF Award Search covers NSF only. Absence from these databases is not proof that no funding exists.
"""),

md(r"""
---
## 4. Taxonomy for 1000 problems

### 4a. By research progression (20 batches of 50)

| problems | batches | progression stage | main categories |
|---|---|---|---|
| 1–50 | 1 | algebra, units, calculus, probability, introductory photonics | A B C D E F H I J (+ G K L M N O) |
| 51–150 | 2–3 | Fourier methods, sampling, linear systems, signal processing | G F C N |
| 151–250 | 4–5 | dispersion, time-stretch measurement, pulse propagation | J K E G |
| 251–350 | 6–7 | RF photonics, modulation, detection, noise | L G H B |
| 351–450 | 8–9 | biomedical imaging and synthetic cell-counting data | M H F N |
| 451–550 | 10–11 | rare events and extreme-value statistics | I H N |
| 551–650 | 12–13 | nonlinear optics and optical rogue-wave models | J E I N |
| 651–750 | 14–15 | numerical linear algebra, optimization, inverse problems | F D N |
| 751–850 | 16–17 | physics-inspired image processing and optical computing | G D F N |
| 851–950 | 18–19 | research design, uncertainty, reproducibility, literature review | O H N |
| 951–1000 | 20 | integrated undergraduate research projects | all |

### 4b. By category (primary category of each problem; 1000 total)

| cat. | topic | planned | batch 1 |
|---|---|---|---|
""" + '\n'.join(f'| {k} | {v} | {n} | {sum(1 for p in P if p["cat"] == k)} |' for k, v, n in zip(
    CATS, CATS.values(), [50, 60, 70, 60, 60, 90, 90, 90, 70, 80, 70, 60, 50, 60, 30, 10])) + r"""
| | **total** | **1000** | **50** |

Category P (quantum information) is deliberately small and appears only where the mathematics is shared (for example, a 2×2 unitary in a directional coupler is the same matrix as a qubit gate). **Batch 1 has no P problems**, because none of its classical problems needs quantum mechanics.

### 4c. Notation used in every batch

| symbol | type | meaning | SI unit |
|---|---|---|---|
| λ | scalar | vacuum wavelength | m |
| f, ω | scalar | frequency, angular frequency | Hz, rad/s |
| c | constant | 299 792 458 (exact) | m/s |
| n, n_g | scalar function of λ | phase index, group index | 1 |
| β₂ | scalar | group-velocity dispersion | s²/m (quoted ps²/km) |
| D | scalar | dispersion parameter, D = −2πcβ₂/λ² | s/m² (quoted ps/(nm·km)) |
| L, z | scalar | fiber length, propagation distance | m |
| T₀, T(z) | scalar | 1/e intensity half-width, width at z | s |
| L_D | scalar | dispersion length T₀²/\|β₂\| | m |
| M | scalar | time-stretch factor | 1 |
| P, I, E | scalar | power, intensity, field (E complex) | W, W/m², V/m |
| X, Z, K, T_k | random variables | (capital letters) | as stated |
| **A**, **M** (bold or matrix) | matrices | as stated | as stated |
| *data* | measured or synthetic data | always labeled "synthetic" when simulated | as stated |
"""),

md(r"""
---
## 5. Problems JW-001 to JW-050

Answers are **not** printed here; they are computed in Section 6. Every problem uses exact rational inputs where possible, SI units internally, and states whether its numbers are taken from a source or are synthetic.
"""),
]

for i in range(0, len(P), 5):
    cells.append(md('\n---\n'.join(problem_md(p) for p in P[i:i + 5])))

cells += [
md(r"""
---
## 6. Answer key (computed)

Each cell starts with `import sympy as sp` and `sp.init_printing()`, defines the exact SI constants, then solves one problem. Exact results are printed first, decimals second. Simulations use fixed seeds and are labeled simulated.
"""),
]
for p in P:
    cells.append(md(f"**JW-{p['id']:03d}**: {p['q']}"))
    cells.append(code(HDR + p['code']))

lean_src = (ROOT / LEAN_REL).read_text(encoding='utf-8')
cells += [
md(r"""
---
## 7. Lean 4 + Mathlib formalizations

**Status: verified.** All 12 theorems below were compiled on 2026-10-12 with Lean `v4.34.0-rc2` and Mathlib (tag `v4.34.0-rc2`), using `lake env lean` inside a local Mathlib project, with no errors and no `sorry`. The cell below recompiles them on your machine, if that project exists; if it doesn't, it says so rather than claiming success.

Each theorem is a **mathematical abstraction** of a problem. Lean proves the mathematics. It does not prove that the physical model is right; the model's assumptions enter as hypotheses.

| theorem | problem | claim | hypotheses needed, and why |
|---|---|---|---|
| `freq_of_wavelength`, `double_wavelength_halves_freq` | JW-001 | c = λf ⇒ f = c/λ; doubling λ halves f | λ ≠ 0 (division) |
| `dB_mul` | JW-008 | 10 log₁₀(g₁g₂) = 10 log₁₀ g₁ + 10 log₁₀ g₂ | g₁, g₂ > 0 (log of a non-positive number is meaningless as a gain) |
| `broadening_ge` | JW-020 | T₀ ≤ T₀√(1 + (z/L_D)²) | T₀ ≥ 0; holds for every z and L_D (even L_D = 0, where Lean's z/0 = 0 makes it trivially true) |
| `broadening_doubles` | JW-020 | at z = √3 L_D the width is 2T₀ | L_D ≠ 0 |
| `attenuation_solves_ode` | JW-024 | d/dz [P₀e^{−αz}] = −α·P₀e^{−αz} | none: true for all real P₀, α, z |
| `stretch_factor_ge_one` | JW-032 | M = 1 + d₂/d₁ ≥ 1 | d₁ > 0, d₂ ≥ 0 (same-sign dispersion; fails for opposite signs) |
| `ppv_le_one` | JW-033 | PPV ≤ 1 | 0 ≤ p ≤ 1, specificity ≤ 1, and a positive denominator |
| `at_least_one_le_np` | JW-034 | 1 − (1−p)ⁿ ≤ np (union bound) | 0 ≤ p ≤ 1 (via Bernoulli's inequality, which needs −p ≥ −2) |
| `exp_memoryless` | JW-038 | e^{−r(s+t)}/e^{−rs} = e^{−rt} | none |
| `exceed_8_mean_small` | JW-043 | e^{−8} ≤ 1/256 | none (uses e ≥ 2 from 1 + x ≤ eˣ) |
| `na_pos` | JW-047 | √(n₁² − n₂²) > 0 | 0 < n₂ < n₁ (core index above cladding) |

**SymPy is not proof.** SymPy's `simplify(...) == 0` in Section 6 is a computer-algebra check. It can fail to simplify true identities, and it makes branch assumptions (for example about square roots) silently. The Lean theorems are checked by a proof kernel.
"""),
code(r"""
import sympy as sp
sp.init_printing()
import pathlib, shutil, subprocess, time
ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / 'projects').exists() else pathlib.Path.cwd().parent
LEAN_FILE = ROOT / '""" + LEAN_REL + r"""'
MATHLIB_PROJECT = pathlib.Path.home() / 'lean' / 'NavierStokesAndEuler'     # any Lake project with Mathlib v4.34.0-rc2 built
print(LEAN_FILE.read_text(encoding='utf-8'))
if MATHLIB_PROJECT.exists() and shutil.which('lake'):
    t0 = time.time()
    res = subprocess.run(['lake', 'env', 'lean', str(LEAN_FILE)], cwd=MATHLIB_PROJECT, capture_output=True, text=True, timeout=1800)
    if res.returncode == 0 and 'sorry' not in res.stdout + res.stderr:
        print(f'STATUS: verified. {sum(1 for ln in LEAN_FILE.read_text(encoding="utf-8").splitlines() if ln.startswith("theorem "))} theorems compiled with Mathlib in {time.time() - t0:.0f} s, no errors, no sorry.')
    else:
        print('STATUS: NOT verified on this run.\n', res.stdout[-3000:], res.stderr[-3000:])
else:
    print('STATUS: not checked on this machine (no Mathlib project found). Status from 2026-10-12: verified.')
"""),

md(r"""
---
## 8. Dependency graph

An arrow from A to B means problem B uses a result or skill from problem A. Problems with no incoming arrow are entry points. Columns are the depth of the longest prerequisite chain.
"""),
code(r"""
import sympy as sp
sp.init_printing()
import sys, pathlib
import matplotlib.pyplot as plt
ROOT = pathlib.Path.cwd() if (pathlib.Path.cwd() / 'projects').exists() else pathlib.Path.cwd().parent
sys.path.insert(0, str(ROOT / 'projects' / 'jalali_workbook'))
from problems_batch01 import P
pre = {p['id']: p['pre'] for p in P}; cat = {p['id']: p['cat'] for p in P}
depth = {}
def d(i):
    if i not in depth: depth[i] = 0 if not pre[i] else 1 + max(d(j) for j in pre[i])
    return depth[i]
for i in pre: d(i)
cols = {}
for i in sorted(pre): cols.setdefault(depth[i], []).append(i)
pos = {i: (dd, -k) for dd, ids in cols.items() for k, i in enumerate(ids)}
OKABE_ITO = {'A': '#E69F00', 'B': '#56B4E9', 'C': '#009E73', 'D': '#F0E442', 'E': '#0072B2', 'F': '#D55E00', 'G': '#CC79A7',
             'H': '#999999', 'I': '#000000', 'J': '#E69F00', 'K': '#56B4E9', 'O': '#009E73', 'N': '#D55E00'}
MARK = {'A': 'o', 'B': 's', 'C': '^', 'D': 'v', 'E': 'D', 'F': 'P', 'G': 'X', 'H': 'o', 'I': 's', 'J': '^', 'K': 'v', 'O': 'D', 'N': 'P'}
fig, ax = plt.subplots(figsize=(10, 11))
for i, js in pre.items():
    for j in js:
        ax.annotate('', xy=pos[i], xytext=pos[j], arrowprops=dict(arrowstyle='->', color='0.55', lw=0.8, shrinkA=7, shrinkB=7))
for i, (x, y) in pos.items():
    ax.plot(x, y, MARK[cat[i]], color=OKABE_ITO[cat[i]], ms=11, mec='k', mew=0.6)
    ax.text(x + 0.08, y, f'{i:02d} ({cat[i]})', va='center', fontsize=7.5)
ax.set_xticks(range(max(cols) + 1)); ax.set_xlabel('depth (longest prerequisite chain)'); ax.set_yticks([])
ax.set_title('Batch 1 dependency graph (label: problem number and category)')
plt.tight_layout(); plt.show()
print(f'Figure: {len(pre)} problems as markers, one column per depth level 0 to {max(cols)}; arrows point from prerequisite to dependent problem.')
print(f'{len(cols[0])} problems are entry points (depth 0); the longest chain has {max(cols) + 1} problems:', end=' ')
deep = max(pre, key=lambda i: depth[i]); chain = [deep]
while pre[chain[-1]]: chain.append(max(pre[chain[-1]], key=lambda j: depth[j]))
print(' <- '.join(f'JW-{i:03d}' for i in chain))
for i in sorted(pre):
    if pre[i]: print(f'JW-{i:03d} <- ' + ', '.join(f'JW-{j:03d}' for j in pre[i]))
"""),

md(r"""
---
## 9. Three problems suitable for a semester research project

1. **JW-028: wavelength-to-time calibration of a dispersive Fourier transform.** Extend the 4-point least-squares fit to a full calibration: include second-order dispersion (a quadratic map), estimate parameter uncertainties from residuals, compare against a Sellmeier-based prediction from `dgs/materials.py`, and test how calibration error propagates into a recovered spectrum. Uses only synthetic or openly published data.
2. **JW-041 → JW-045: estimating rare-event probabilities.** Start from naive Monte Carlo (JW-041), add importance sampling to estimate P(Z > 6) efficiently, then compare exponential, Pareto and other heavy-tailed fits on synthetic data with known ground truth, including how the threshold convention (JW-045) changes the count of "rogue" events.
3. **JW-030: unmixing frequency-multiplexed measurements.** Generalize the 2×2 crosstalk model to N pixels, study how the condition number grows with crosstalk, add noise (seeded), and compare direct inversion, least squares, and regularized (Tikhonov) solutions. A simplified model; it does not reproduce the published FIRE processing.

---
## 10. Likely experimental and modeling limitations

- **Second-order dispersion only.** JW-005, -019, -020, -023 and -032 ignore third-order dispersion, loss along the fiber, and nonlinearity. Real time-stretch systems can require all three.
- **Bulk material vs. fiber.** JW-046 uses bulk fused silica; real fiber adds waveguide dispersion, so its group index and D differ.
- **Ideal components.** Lossless polarizers (JW-027), identical couplers (JW-029), single-pole detectors (JW-012, -025), ideal stretch (JW-031) and noiseless unmixing (JW-030).
- **Independence assumptions.** The probability problems (JW-033 to -040) assume independent cells or photons; real flow and real light can be correlated.
- **Synthetic numbers.** Frame rates, data rates, prevalence, sensitivity, cell sizes, and the datasets in JW-028 and JW-045 are synthetic. They teach the method and are not measurements from any laboratory.
- **Approximate material values.** The silicon and silica indices in JW-048 are rounded textbook values; the Cauchy coefficients in JW-014 are synthetic.

---
## 11. What this does not prove

- It does **not** show that any equation is the one used in a particular Jalali Lab instrument. The equations are standard (cited textbooks and review papers), and simplified models are labeled.
- The Lean theorems prove **mathematical statements under hypotheses**. They do not prove that fibers, detectors or cells satisfy those hypotheses.
- SymPy results are **computations**, not proofs; a `True` from `simplify` is evidence, not certainty.
- The Monte Carlo result (JW-041) is **simulated data**. It estimates a known probability; it is not an observation of rogue waves or of anything physical.
- The funding tables show what is in two public databases and three institutional announcements. They do **not** show the laboratory's complete or current funding.
- Nothing here is endorsed by, or speaks for, Prof. Jalali, the Jalali Lab, or UCLA.

---
## 12. Proposed plan for Batches 2 to 20

| batch | problems | focus | example problem types | Lean targets (Mathlib) |
|---|---|---|---|---|
| 2 | 51–100 | Fourier transform basics, convolution | Gaussian ↔ Gaussian transforms; time-bandwidth product; convolution of rect pulses | linearity of convolution on finite sequences; Parseval for the DFT (finite sums) |
| 3 | 101–150 | sampling, aliasing, filtering, linear systems | Nyquist with time stretch; aliasing frequency formula; FIR filters; LTI impulse response | aliasing formula as modular arithmetic; causality of FIR filters |
| 4 | 151–200 | dispersion and pulse propagation | Sellmeier → β₂; third-order dispersion; chirp; split-step propagation (numerical) | positivity and monotonicity of broadening factors |
| 5 | 201–250 | time-stretch measurement and DFT | far-field (stationary-phase) condition; wavelength-to-time resolution; stretch-factor design | stretch-factor algebra; resolution bounds as inequalities |
| 6 | 251–300 | modulation and mixing | Mach-Zehnder transfer function; small-signal modulation; Bessel sidebands; mixing products | trig product-to-sum identities; MZM transfer bounds |
| 7 | 301–350 | detection and noise | shot, thermal, RIN noise; SNR budgets; ENOB of a photonic ADC | SNR monotonicity; noise variance addition |
| 8 | 351–400 | imaging models | line-scan imaging geometry; resolution; motion blur in flow; pixel counting | counting identities; area and sampling bounds |
| 9 | 401–450 | synthetic cell-counting statistics | Poisson counts; Bayes classification; ROC curves; confusion matrices (synthetic) | Bayes rule; Markov and Chebyshev bounds |
| 10 | 451–500 | rare events I | exceedance probabilities; return periods; importance sampling | Markov/Chernoff-type bounds |
| 11 | 501–550 | extreme-value statistics | maxima of samples; Gumbel/Fréchet/Weibull families; threshold choice | max of i.i.d. CDF = Fⁿ; tail monotonicity |
| 12 | 551–600 | nonlinear optics | Kerr effect; nonlinear length; self-phase modulation; soliton number | algebra of nonlinear/dispersion lengths |
| 13 | 601–650 | rogue-wave models | modulation instability gain; long-tailed statistics from simulated NLSE (labeled simulated) | MI gain maximum (calculus) |
| 14 | 651–700 | numerical linear algebra | SVD, conditioning, least squares, regularization | matrix identities; norm inequalities |
| 15 | 701–750 | optimization and inverse problems | phase retrieval (Gerchberg-Saxton), gradient descent, convexity | convexity of least squares; descent lemma (simple form) |
| 16 | 751–800 | physics-inspired image processing | phase-stretch transform as an all-pass filter; edge detection | unit-modulus (all-pass) filters preserve energy |
| 17 | 801–850 | optical computing | linear optical transforms as matrices; unitary meshes; analog computing error | unitary matrices preserve norm |
| 18 | 851–900 | uncertainty and reproducibility | propagation of uncertainty; seeds; version control of notebooks | linear error propagation |
| 19 | 901–950 | research design and literature review | source evaluation; preregistration; power analysis | sample-size inequalities |
| 20 | 951–1000 | integrated projects | 50 multi-part capstones combining batches 1 to 19 | selected from earlier batches |

**Each batch will include:** the 15-field problem format, a computed SymPy answer key for every problem, at least 3 compiled Lean theorems (status reported honestly), a dependency graph, three project candidates, limitations, and a "what this does not prove" section.

---
## 13. Publication and textbook references

""" + '\n'.join(f'- [{k}] {v}' for k, v in S.items() if k not in ('SYN',)) + r"""

*The publication details above were entered from memory and are to be confirmed against the original papers before being cited in a report.*

**Awaiting approval before Batch 2.**
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3.12 (py312)", "language": "python", "name": "py312"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(OUT, len(cells), 'cells')
