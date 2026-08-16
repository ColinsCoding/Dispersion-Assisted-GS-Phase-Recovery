# Physics 17 (Rudnick) — Source and Scope

**Source:** *Vibration, Wave Motion, Sound, Heat and Kinetic Theory* — prepared by
Joseph Heiserman from notes by Isadore Rudnick, revised by Steve Baker, September
1980. A UCLA course reader. Rudnick was a UCLA physics professor known for
low-temperature/acoustic physics; this reader is the classical-mechanics/waves/
thermo backbone that sits underneath the RF, dispersion, and photonics work this
repo already does.

**Original scans:** `D:\Spring2026\Physics17_part_*.pdf` (10 files, 181 phone-scanned
pages, no text layer). OCR'd with pymupdf (render) + tesseract 5.4 (recognize);
scan quality is uneven (shadows/glare from the phone-photo capture), so treat
every transcription here as a **first draft to check against the original scan**,
not a verbatim source.

## Table of contents (from the reader's own TOC, page `i`)

| Chapter | Topic | Reader pages |
|---|---|---|
| — | General References | 1 |
| 1 | Oscillations | 1–46 |
| 2 | Physics of Fluids | 47–66 |
| 3 | Elasticity | 67–75 |
| 4 | Waves | 76–102 |
| 5 | Sound Waves | 103–136 |
| 6 | Thermodynamics | 137–146 |
| 7 | Kinetic Theory | 147–171 |
| App. I | The Quality Factor, Q | 172–174 |
| App. II | Energetics of a Sound Field | 175–181 |

## What's already covered elsewhere in this repo

Chapters 1 (Oscillations) and 4 (Waves) overlap heavily with existing modules —
[`dgs/vibration_modes.py`](../dgs/vibration_modes.py),
[`dgs/pierce_oscillator.py`](../dgs/pierce_oscillator.py),
[`dgs/eigen_modes.py`](../dgs/eigen_modes.py),
[`dgs/pde_separation.py`](../dgs/pde_separation.py),
[`dgs/dispersive_fourier.py`](../dgs/dispersive_fourier.py) — so those chapters are
lower priority to re-derive from scratch; skim for anything Rudnick covers that
those modules don't (driven/damped oscillator Q-factor derivation, normal-mode
coupled oscillators).

Chapter 7 (Kinetic Theory) now has a companion notebook —
[`notebooks/physics17_kinetic_theory.ipynb`](../notebooks/physics17_kinetic_theory.ipynb) —
that reproduces Rudnick's own pressure derivation symbolically, cross-checks it
against `dgs/maxwell_boltzmann.py`, and adds a torch-based inverse-problem demo.

Chapters 2 (Fluids), 3 (Elasticity), 5 (Sound Waves proper), and 6
(Thermodynamics) remain **genuine gaps** — nothing in `dgs/` derives
elasticity/stress-strain or classical thermodynamics from scratch yet
(`dgs/microfluidics.py` touches an adjacent corner of fluids but not the
reader's own hydrostatics derivation). These are the better ROI targets if/when
they become `dgs/` modules + notebooks, per [[project_curriculum_timeline]]'s
depth-over-speed approach.

See [`jalali_lab_curriculum_map.md`](jalali_lab_curriculum_map.md) for how this
reader fits into the broader Jalali-lab prep plan.

## OCR'd chapter transcriptions

Raw tesseract output, lightly cleaned (dropped only near-empty/punctuation-only
noise lines — nothing rewritten), organized by chapter with a per-page
low-confidence flag. **This is a search/reference aid, not a clean copy** — the
scan quality means roughly a third of pages are flagged unreliable, almost
always because the page is dominated by a handwritten diagram or equation
rather than typewritten prose (tesseract handles the typewritten paragraphs
reasonably well; it does not handle handwriting).

| File | Physical pages | Pages | Low-confidence | Boundary confidence |
|---|---|---|---|---|
| [`front_matter.md`](front_matter.md) | 1–4 | 4 | 4 | confirmed |
| [`ch1_oscillations.md`](ch1_oscillations.md) | 5–50 | 46 | 18 | offset-inferred |
| [`ch2_fluids.md`](ch2_fluids.md) | 51–70 | 20 | 5 | offset-inferred |
| [`ch3_elasticity.md`](ch3_elasticity.md) | 71–79 | 9 | 4 | confirmed (heading found p.71) |
| [`ch4_waves.md`](ch4_waves.md) | 80–106 | 27 | 14 | offset-inferred, body cross-references confirm range |
| [`ch5_sound_waves.md`](ch5_sound_waves.md) | 107–140 | 34 | 7 | offset-inferred |
| [`ch6_thermodynamics.md`](ch6_thermodynamics.md) | 141–150 | 10 | 4 | confirmed (heading p.141, footer "146" p.150) |
| [`ch7_kinetic_theory.md`](ch7_kinetic_theory.md) | 151–175 | 25 | 2 | confirmed start (heading p.151); end inferred |
| [`appendix1_q.md`](appendix1_q.md) | 176–179 | 4 | 0 | confirmed start (heading p.176) |
| [`appendix2_sound_field.md`](appendix2_sound_field.md) | 180–181 | 2 | 1 | **scan incomplete** — ends mid-derivation |

**How the page mapping was determined:** the reader's own printed page numbers
(visible in page footers) run 4 behind the physical scan-page number
(`physical = reader_page + 4`) everywhere this was directly checked — confirmed
independently at three separate "CHAPTER N" heading pages (physical 71, 141,
151) spanning 80 pages of the document with no drift. Chapters 1/2/4/5 use that
offset without an independently-found heading page (tesseract's heading-line
detection missed them, likely due to scan glare), so their boundaries are
inferred, not confirmed — flagged in the table above.

**The scan is incomplete at the end.** Appendix II ("Energetics of a Sound
Field," reader pages 175–181, 7 pages) only has 2 physical pages in this scan
(180–181), and page 181 — the very last page provided — cuts off mid-derivation
(continuity equation, kinetic-energy-density definition). Whoever photographed
this reader in 1980-onward either stopped early or lost the last ~5 pages. If
you get the missing pages scanned, say so and I'll OCR and fold them in.
