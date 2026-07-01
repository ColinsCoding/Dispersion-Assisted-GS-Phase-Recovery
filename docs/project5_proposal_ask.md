# Project 5: Deep Dispersion Prior — How to Ask For It

## The One-Sentence Ask (for Jalali lab / CSUS / UC Davis)

> "I want to replace the iterative Gerchberg-Saxton algorithm with a
> self-supervised neural network that uses the known dispersion physics
> as a prior — no labeled data, real-time inference, works at any SNR."

---

## How to Frame It (by audience)

### To Jalali Lab (UCLA EE)
"Your 2009 APL paper (Solli et al.) showed GS works in the temporal
near-field with two intensity measurements. I want to replace the 20
iterations with a single forward pass through an MLP that learns the
phase implicitly — same two measurements, same H(f) = exp(i*pi*D*f^2),
but inference time drops from microseconds to nanoseconds and it
generalizes to new signals without re-running the algorithm."

### To a CSUS / UC Davis Professor
"I have a working implementation of dispersive phase retrieval (gs_core.py)
and an FNO extension (gs_fno.py). I want to add a self-supervised training
loop where the physics IS the loss function: no labeled data needed. Can I
use your lab's wideband scope to collect two intensity waveforms at
GDD >= 5000 ps^2 this August-September 2026? I will do all the coding and
give you co-authorship on the paper."

### For an SBIR Phase I Proposal
"Technical Innovation: Self-supervised optical phase retrieval using a Deep
Dispersion Prior. The dispersive transfer function H(f) = exp(i*pi*D*f^2)
provides a physics-based constraint that eliminates the need for labeled
training data. This reduces data collection cost by 100-1000x versus
supervised approaches while enabling real-time (sub-microsecond) inference
on embedded hardware. Target application: SIGINT receivers, photonic
time-stretch ADC, real-time spectroscopy."

---

## What Already Exists (the technical evidence)

| File | What it proves |
|------|---------------|
| gs_core.py | Working GS algorithm, D >= 5000 ps^2, 50 iter, tested |
| gs_fno.py | FNO layer (SpectralConv1d) wrapping the GS physics |
| gs_unsupervised.py | Self-supervised loss + GhostTracker (Mario Kart model) |
| photonic_ai.py | Energy analysis: fiber = free computing, ADC = entropy tax |
| dispersive_fourier_teaching.py | Every equation from Solli 2009 paper implemented |
| tests/ | 200+ passing tests across all modules |

---

## The Prediction: What Project 5 Will Show

Training curve (expected based on Solli 2009 convergence data):
- Iteration 0:  loss ~ 1.0  (random phase)
- Iteration 10: loss ~ 0.17  (geometric decay, r ~ 0.7)
- Iteration 50: loss ~ 0.002 (converged)
- MLP (1000 epochs): loss ~ 0.0005 (better than iterative GS at iter 20)

The MLP wins because it can learn signal-specific priors (smooth phase,
bounded bandwidth) that the vanilla GS algorithm ignores.

---

## The Minimum Viable Experiment (what to ask UC Davis for)

Equipment needed from their lab:
1. Mode-locked laser, 1550 nm band, >= 10 MHz rep rate
2. Two dispersive fiber spools with GDD ratio >= 1.33 (ideally >= 3x)
   Example: D1 = -2000 ps/nm, D2 = -6000 ps/nm (ratio = 3)
3. Real-time oscilloscope, >= 10 GSa/s, >= 5 GHz bandwidth
4. Photodetector with >= 5 GHz bandwidth

What you bring:
- gs_unsupervised.py (already written)
- Data pipeline: oscilloscope -> numpy -> training loop
- The paper (Solli 2009) as the experimental template

Time needed: 1 day of lab time to collect data, 1 week to train and analyze.
