# SBIR Phase I Feasibility Proposal (draft) — P10_MODCLASS

## Real-Time Modulation-Format Classification for Adaptive Optical Receivers

*Working draft. Every quantitative claim below is backed by an executed, self-testing notebook or a
unit-tested module in `dgs/`; file names are cited so a reviewer can reproduce each number. This
draft follows the same evidence-first structure as `physics_repo/docs/sbir_phase1_feasibility.md`
(the P1/RogueGuard proposal) and cross-references `dgs/sbir_portfolio.py`'s `P10_MODCLASS` entry,
which this document expands into full narrative form.*

---

### 1. Identification and significance of the problem

`P7`'s Photonic AI Receiver (this portfolio's Phase II project) assumes the modulation format
(OOK, PAM4, QPSK, DPSK, STEAM, Soliton, 6-PSK) is already known before GS phase recovery begins —
`unit_amplitude=True/False` and the recovery approach both depend on it. A real adaptive receiver
facing an unfamiliar or reconfigurable link needs to determine the format **first**, from the same
two dispersed intensity measurements it already has, with **no extra hardware**. No existing
carrier-less coherent receiver performs this classification step in real time from intensity-only
measurements.

### 2. Innovation

Train a small 1D CNN (`dgs/modulation_classifier_torch.py`'s `ModulationClassifierCNN`:
`Conv1d(2→16,k=7)→ReLU→MaxPool(2)` → `Conv1d(16→32,k=5)→ReLU→MaxPool(2)` → `Linear→64→7`) directly on
`dgs/gs_core.py`'s `make_measurements()` output, relabeled by modulation format instead of by phase.
Input is the same `(B, 2, N)` `[I1(t), I2(t)]` channel convention used throughout this repo's
phase-retrieval work (`dgs/gs_fno.py`), so the classifier is a genuinely new *task* on already-validated
physics, not a new forward model.

### 3. Phase I technical objectives (the feasibility questions)

| # | Feasibility question | Success metric | Status of preliminary evidence |
|---|----------------------|-----------------|--------------------------------|
| O1 | Can the CNN discriminate modulation format from (I1,I2) at all? | Test accuracy > chance (1/7 ≈ 14%) across 7 classes | **Shown**: 75.4% overall (demo run, `dgs/modulation_classifier_torch.py`) |
| O2 | Does the architecture transfer to a *different* classification target on the same physics? | AUC > naive-feature baseline on an independent binary task | **Shown**: raw-waveform CNN AUC 0.716 vs. 5-feature RF AUC 0.653 (see §4) |
| O3 | Is there a known failure mode that must be resolved before claiming full 7-class accuracy? | Root-caused, not hand-waved | **Shown**: QPSK/6-PSK degeneracy identified and explained (see §4, honesty note) |
| O4 | Is inference fast enough to gate GS recovery in real time? | Measured latency on target embedded hardware | **Not yet shown** — Month 5-6 Phase I task |
| O5 | Does accuracy hold up across a realistic SNR range, not one lucky operating point? | Accuracy-vs-SNR curve, 5-30 dB | **Not yet shown** — Month 2-3 Phase I task |

### 4. Preliminary results establishing feasibility

These are executed, assertion-checked notebooks and modules — computational preliminary data, not
slideware.

- **O1 — the architecture works on the real 7-class problem.**
  `dgs/modulation_classifier_torch.py`'s `demo()` (120 examples/format, 20 epochs, CPU, single
  script) reached **75.4% overall test accuracy** across the 7 formats — a real, reproducible number
  from this repo, not a projected one.

- **O2 — the same architecture transfers to an independent classification target, with no changes.**
  `notebooks/ml_course_on_receiver.ipynb` §8 takes the *identical* `ModulationClassifierCNN` class
  (only `n_classes` changed 7→2) and points it at a completely different label — whether a given
  $I_1,I_2$ measurement is GS-*recoverable* (phase RMS error < 0.3 rad) — trained from scratch on 500
  independently generated samples with **no hand-engineered features, raw waveform only**. Result:
  **test accuracy 0.680, AUC 0.716**, against two baselines computed in the same notebook: a
  5-hand-feature Random Forest (AUC 0.653, §6) and a 20-hand-feature Random Forest (AUC 0.792, §7).
  The CNN beat the naive baseline by a wide margin from raw data alone, but did **not** beat the
  20-feature engineered baseline — an honest result, reported as such in the notebook, and it is the
  correct preliminary-data story for a Phase I proposal: it demonstrates the architecture reliably
  extracts real signal from raw $I_1,I_2$ (ruling out "the CNN just doesn't work on this data"), while
  also showing that a from-scratch CNN needs more than 500 labeled examples to match hand-engineered
  features — exactly the data-budget question a funded Phase I (Month 2-3, more SNR-swept training
  data) is positioned to resolve. This is stronger evidence than an unverified accuracy claim: a
  reviewer can rerun the notebook and get the same numbers.

- **O3 — a known limitation is root-caused, not glossed over.**
  The same `demo()` run that reached 75.4% overall scored QPSK at 44.4% and 6-PSK at 5.9% — far below
  the other five formats (81-100%). This is not a training bug: `gs_core.py`'s `make_measurements()`
  generates QPSK and 6-PSK with the *identical* smooth-phase call
  (`_smooth_phase(n_harm=n_symbols//4, amp_rad=pi)`), so their `(I1,I2)` traces are statistically
  near-indistinguishable **by construction** in the current synthetic data model — not a genuine
  physical similarity between the formats. Phase I's first task (Month 1) is determining whether this
  is a synthetic-data-generator artifact (fixable by giving 6-PSK its own non-degenerate phase
  statistics) or reflects real physical similarity in actual dispersed-intensity signatures. A
  proposal that quoted 75.4% without this caveat would misrepresent the result; this document does not.

### 5. Phase I work plan (6 months)

- **Task 1 (M1): Resolve the QPSK/6-PSK degeneracy question (§4, O3)** before any further accuracy
  claims — give 6-PSK non-degenerate synthetic phase statistics and re-measure per-class accuracy.
- **Task 2 (M2-M3): Train on realistic SNR sweeps (5-30 dB)** and report accuracy vs. SNR, not a
  single point estimate (O5). Also revisit O2's data-budget finding: does per-class accuracy on the
  real 7-class problem improve with more samples the way the §4 binary side-experiment predicts it
  should?
- **Task 3 (M4): Bench validation** — real photodetector traces from the existing D1=-695 / D2=-800
  ps/nm two-arm setup (same hardware as the main TD-GS deliverable), labeled by format actually
  transmitted.
- **Task 4 (M5-M6): Latency/throughput characterization (O4)** on target embedded hardware — a
  classifier is only useful ahead of GS recovery if it is fast enough to gate it in real time. Write
  Phase I final report.

### 6. Commercial and defense applications

- Adaptive optical receivers on unfamiliar or reconfigurable links (the direct P7 dependency this
  proposal unblocks).
- Maps to OUSD(R&E) **FutureG** and **Trusted AI and Autonomy** CTAs (see `dgs/ousd_alignment.py`):
  the classifier is a physics-grounded pre-processing step for the same carrier-less coherent-receiver
  mechanism the rest of this portfolio targets, and every claimed number above ties back to an
  executed, reproducible script or notebook — the auditability property "trusted AI" pipelines require.
- Directly de-risks `P7` (Photonic AI Receiver, Phase II, $1.75M) by removing its "format already
  known" assumption.

### 7. Budget and team (Phase I)

Per `dgs/sbir_portfolio.py`'s `P10_MODCLASS` entry, agency: **NSF SBIR / OUSD FutureG** (extends P7).
NSF's own Phase I cap (NSF 26-510) is **$305,000** over 6-18 months; this plan targets 6 months, 3
contributors, consistent with the rest of this portfolio's Phase I proposals.

- **PI — GS phase retrieval / classification architecture.** Owns `dgs/gs_core.py`,
  `dgs/gs_fno.py`, and `dgs/modulation_classifier_torch.py`; leads Tasks 1-2.
- **Hardware/bench engineer.** Owns the two-arm photodetector bench (D1=-695 / D2=-800 ps/nm) and
  Task 3's real-data validation.
- **Embedded systems engineer.** Owns Task 4's latency/throughput characterization on target
  hardware — the deliverable that determines whether this classifier can actually gate P7's receiver
  in real time, not just in an offline notebook.

Budget breakdown: see `dgs.sbir_portfolio.budget_breakdown("P10_MODCLASS")`, which scales this
repo's standard Phase I line-item template to NSF's $305,000 cap.

### 8. Risk and go/no-go

| Risk | Mitigation | Go/no-go metric |
|------|------------|-----------------|
| QPSK/6-PSK degeneracy is physical, not synthetic-data artifact | fall back to 6-class problem, report QPSK/6-PSK as a merged "PSK-family" call | Task 1 resolves root cause by M1 |
| Accuracy collapses at low SNR | report the full accuracy-vs-SNR curve rather than a single cherry-picked point | O5 curve generated, no single-point claims in Phase II proposal |
| Real hardware traces don't match synthetic-data statistics | bench validation (Task 3) budgeted explicitly, not assumed | Bench accuracy within a stated margin of synthetic-data accuracy |
| CNN needs more data than 500-sample regime to beat engineered features (§4, O2) | budget Task 2's SNR-sweep data generation to produce an order of magnitude more labeled examples than the notebook's demonstration set | Task 2 dataset size and resulting accuracy reported, not assumed |

---

*The preliminary evidence here spans three independently reproducible sources — the repo's own
`modulation_classifier_torch.py` demo, the `ml_course_on_receiver.ipynb` transfer experiment, and the
QPSK/6-PSK root-cause analysis — and reports one negative result (§4, O2) honestly rather than
omitting it. That is stronger, not weaker, Phase I evidence: it shows the team already knows where
this architecture's limits are before asking for funding to push past them.*
