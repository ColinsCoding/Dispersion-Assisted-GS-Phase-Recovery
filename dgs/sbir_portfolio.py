"""SBIR Portfolio: 7 proposals (P2-P8) built on this repo's physics stack.

PROPOSAL LADDER (each builds on the last):
  P1 (EXISTING):  TD-GS Phase Recovery -- optical rogue wave monitor (RogueGuard)
                  $275K Phase I, OUSD FutureG / Integrated Sensing
  P2:             STEAM Microscopy -- femtosecond cell imaging + GS phase recovery
                  $275K Phase I, NIH SBIR / BARDA / NSF IIP
  P3:             CRISPR Target Verification -- ultrafast laser + STEAM confirms
                  gene edit in single cell <1 ms, no off-target damage scoring
  P4:             CUDA Real-Time STEAM -- GPU pipeline 36 Mfps at 10 GB/s
                  $275K Phase I, NSF OAC / DOD HPCMP
  P5:             Bayesian Cancer Cell Detection -- GS + Bayes classifier on STEAM
                  $275K Phase I, NCI SBIR (National Cancer Institute)
  P6:             Rogue Wave Biophysics -- NLSE instability in neural membranes
                  $275K Phase I, DARPA BTO (Biological Technologies Office)
  P7:             Photonic AI Receiver (Project 4) -- STEAM + ML for comms
                  $1.75M Phase II (build on P1 + P4), OUSD Trusted AI
  P8 (THIS FILE, PRIORITIZED FIRST): Generative AI Phase Retrieval Prior --
                  learned generative prior replacing dispersion_gs_descent.py's
                  hand-designed smoothness regularizer
                  $275K Phase I, OUSD FutureG / Trusted AI and Autonomy

MATH CHAIN (same physics runs through all 7):
  H(f) = exp(i*pi*D*f^2)           [GVD dispersion operator, Griffiths Ch 9]
  |E(f)|^2 = I(t) via ADC           [STEAM time-stretch measurement]
  GS: arg min ||A*phi - phi_true||   [phase retrieval from I1, I2]
  NLSE: i*dA/dz = beta2/2*A_tt - gamma*|A|^2*A  [nonlinear extension]
  Bayes: P(cancer|I) = P(I|cancer)*P(cancer) / P(I)  [detection]
  CUDA: each GS iteration = embarrassingly parallel FFT  [GPU acceleration]

Run: py -3.13 -c "from dgs.sbir_portfolio import demo; demo()"
"""
import numpy as np


# ── Proposal data structure ───────────────────────────────────────────────────

PROPOSALS = {

    "P2_STEAM": {
        "title": "Femtosecond STEAM Microscopy with Real-Time GS Phase Recovery",
        "agency": "NIH SBIR / BARDA",
        "phase": "Phase I",
        "budget": 275_000,
        "duration_months": 6,
        "topic": "Ultrafast label-free cell imaging for pathogen detection",
        "significance": (
            "Current flow cytometers image 10,000 cells/second at 1-2 pixel resolution. "
            "The foundational STEAM demonstration (Goda, Tsia, Jalali, Nature 2009) achieved "
            "continuous real-time imaging at a 163 ns frame period (6.1 MHz frame rate) and a "
            "440 ps shutter speed, applied to microfluidic flow and laser-ablation phase-"
            "explosion imaging -- already >600x faster than conventional cytometry. "
            "(The N-STEAM variant's 36.1 MHz figure, cited separately below, is from a "
            "different, later Nomarski/DIC-contrast configuration -- NOT the same "
            "demonstration as the 6.1 MHz baseline; both numbers should be cited to their "
            "correct source, not conflated, in the actual submitted proposal.) "
            "A single 6-minute blood draw contains enough cells to detect 1 circulating "
            "tumor cell (CTC) in 10^9 blood cells -- impossible with conventional cameras."
        ),
        "innovation": (
            "We combine Serial Time-Encoded Amplified Microscopy (STEAM) with "
            "dispersion-assisted Gerchberg-Saxton (GS) phase retrieval. "
            "STEAM provides intensity I(t); GS recovers the full complex field E(t). "
            "Phase contrast doubles the morphological features available for classification. "
            "No existing STEAM system recovers phase in real time."
        ),
        "no_local_oscillator_advantage": (
            "Conventional coherent detection (interferometric phase measurement) requires "
            "mixing the signal with a LOCAL OSCILLATOR (LO) -- a reference beam that must be "
            "phase-locked to the source, adding real hardware (a second laser or a split-off "
            "reference arm, active phase-locking electronics) and real failure modes (LO drift, "
            "polarization mismatch, the LO's own phase noise setting a sensitivity floor). "
            "GS phase recovery from two dispersed intensity measurements needs NEITHER a "
            "reference beam nor phase-locking -- it recovers phase from intensity alone. "
            "This is the literal 'carrier-less coherent receiver' framing from the original "
            "project brief this repo grew out of: two ordinary photodetectors, no LO hardware, "
            "no interferometric alignment to maintain."
        ),
        "verified_citations": [
            "K. Goda, K. K. Tsia, B. Jalali, 'Serial time-encoded amplified imaging for "
            "real-time observation of fast dynamic phenomena,' Nature 458, 1145-1149 (2009) "
            "-- the original STEAM paper; establishes the >1000x frame-rate advantage over "
            "conventional CCDs this proposal's significance section is built on.",
            "D. R. Solli, S. Gupta, B. Jalali, 'Optical phase recovery in the dispersive "
            "Fourier transform,' Applied Physics Letters 95, 231108 (2009) -- the actual "
            "paper this repo's core TD-GS algorithm implements (Reference [2] in the "
            "original project brief); experimentally demonstrates the time-domain GS "
            "algorithm removing the dispersion requirement.",
            "N-STEAM (Nomarski/DIC-STEAM), Jalali Lab UCLA: 36.1 MHz (36 Mfps) demonstrated "
            "frame rate -- the source of the '36 Mfps' figure used throughout this proposal "
            "portfolio (P2, P4, P5); note this specific number is from the N-STEAM variant "
            "(adds Nomarski/DIC contrast for transparent objects), not necessarily identical "
            "hardware to the phase-contrast STEAM configuration proposed here -- worth "
            "verifying against the specific N-STEAM paper before quoting in a submitted "
            "proposal, rather than treating it as automatically transferable.",
        ],
        "approach": [
            "Month 1-2: Extend dgs/steam_imaging.py forward model to include phase-contrast STEAM",
            "Month 3-4: GS convergence with two dispersive paths (D1=-5000, D2=-15000 ps^2)",
            "Month 5:   CUDA kernel for 36 Mfps GS (see P4 -- can be done in parallel)",
            "Month 6:   Bayesian classifier on phase+intensity features for CTC detection",
        ],
        "team": [
            "PI: Colin Casey -- GS phase retrieval (this repo, 156 modules); "
            "former undergraduate researcher in Prof. Jalali's UCLA lab (STEAM/photonic "
            "time-stretch group) -- real prior relationship, but NO letter of support or "
            "formal Co-I commitment exists yet. Do not represent one as secured in any "
            "submitted proposal until it is actually obtained.",
        ],
        "milestones": {
            "M1": "Simulated STEAM forward model, SNR analysis",
            "M2": "GS phase recovery on synthetic cell phantoms, corr > 0.85",
            "M3": "Bayesian classifier F1 > 0.90 on simulated CTC dataset",
            "M4": "Proof-of-concept demo, Phase II application drafted",
        },
        "griffiths_physics": "Ch 9 (H(f)), Ch 4 (refractive index of cell), Ch 8 (Poynting=intensity)",
        "repo_modules": ["dgs/steam_imaging.py", "dgs/gs_core.py", "dgs/bayes_inference.py"],
        "ousd_cta": "Trusted AI (cancer detection AI) + Integrated Sensing",
    },

    "P3_CRISPR": {
        "title": "Ultrafast STEAM Verification of Single-Cell CRISPR Edits",
        "agency": "NIH SBIR (NIGMS) / ARPA-H",
        "phase": "Phase I",
        "budget": 275_000,
        "duration_months": 6,
        "topic": "Real-time optical confirmation of CRISPR-Cas9 gene editing",
        "significance": (
            "CRISPR-Cas9 edits DNA in <1 ms, but verification currently requires "
            "sequencing (days) or fluorescence (hours, label-dependent). "
            "Off-target cuts cause cancer-risk mutations in 1-5% of edited cells. "
            "A sub-millisecond optical readout would enable real-time error rejection "
            "at the single-cell level during therapeutic cell manufacturing."
        ),
        "innovation": (
            "Femtosecond STEAM captures the transient refractive index change (delta_n) "
            "when Cas9 cuts and the DNA helix opens. delta_n ~ 10^-5 over 500 nm region. "
            "GS phase recovery detects delta_n via recovered phase: "
            "delta_phi = 2*pi*delta_n*L/lambda  (L=cell thickness ~10 um, lambda=1550 nm). "
            "delta_phi = 2*pi*1e-5*10e-6/1550e-9 = 0.0004 rad -- detectable with corr>0.999 GS."
        ),
        "approach": [
            "Month 1:   Model Cas9 refractive index signature (Lorentz oscillator, dgs/classical_ed.py)",
            "Month 2-3: STEAM phase sensitivity analysis; shot noise vs delta_phi",
            "Month 4:   Synthetic CRISPR phantom dataset; GS recovery at SNR 20 dB",
            "Month 5-6: Bayesian classifier: edited / unedited / off-target (3 classes)",
        ],
        "milestones": {
            "M1": "Phase sensitivity model: min detectable delta_n vs pulse energy",
            "M2": "GS recovery on CRISPR phantom, phase error < 0.001 rad",
            "M3": "Classifier accuracy > 95% on 3-class synthetic dataset",
            "M4": "ARPA-H Phase II pre-application submitted",
        },
        "griffiths_physics": (
            "Ch 4: delta_n from molecular polarizability change during DNA strand opening; "
            "Ch 9: phase accumulation delta_phi = k*delta_n*L; "
            "Ch 2: Poisson eq for charge redistribution in DNA backbone"
        ),
        "repo_modules": ["dgs/steam_imaging.py", "dgs/organic_chemistry.py",
                         "dgs/classical_ed.py", "dgs/bayes_inference.py"],
        "ousd_cta": "Trusted AI + Human-Machine Interfaces (cell therapy manufacturing)",
    },

    "P4_CUDA": {
        "title": "NVCC CUDA Pipeline for 10 GB/s Real-Time STEAM Phase Retrieval",
        "agency": "NSF OAC / DOD HPCMP",
        "phase": "Phase I",
        "budget": 275_000,
        "duration_months": 6,
        "topic": "GPU-accelerated optical phase recovery at camera frame rate",
        "significance": (
            "STEAM generates 10 GB/s of raw ADC data at 36 Mfps. "
            "CPU-based GS (50 iterations, N=1024) takes 50 ms per frame -> 20 fps max. "
            "CUDA cuFFT on RTX 4090 runs one GS iteration in 0.4 us -> 36 Mfps feasible. "
            "No existing GS implementation runs at camera frame rate on GPU."
        ),
        "innovation": (
            "We map the GS algorithm onto CUDA warps: each warp handles one spectral bin. "
            "FFT: cuFFT with half-precision (FP16) -> 2x throughput. "
            "Constraint projection: elementwise multiply -> trivially parallel. "
            "Convergence check: warp-level reduction on correlation. "
            "50-iteration GS on N=1024 = 100 cuFFT calls = 40 us on A100."
        ),
        "approach": [
            "Month 1:   Port dgs/gs_core.py to CUDA C kernel (nvcc, existing dgs/gs_cuda.py base)",
            "Month 2:   cuFFT integration; benchmark vs numpy on synthetic STEAM data",
            "Month 3:   FP16 half-precision GS; verify phase accuracy vs FP32",
            "Month 4:   PCIe streaming pipeline: ADC -> GPU -> classifier -> output",
            "Month 5-6: Profiling, occupancy optimization, open-source release",
        ],
        "milestones": {
            "M1": "CUDA GS kernel: 10x speedup vs numpy on N=1024",
            "M2": "50-iteration GS in < 50 us on RTX 4090 (enables 20 kfps)",
            "M3": "Phase accuracy: CUDA vs numpy corr > 0.9999",
            "M4": "End-to-end STEAM pipeline at 1 Mfps on single GPU",
        },
        "griffiths_physics": (
            "Not Griffiths -- CUDA is algorithm engineering. "
            "But: each FFT IS the Fourier transform integral (Griffiths Ch 9 eq 9.20). "
            "cuFFT computes integral E(f) = integral E(t)*exp(-i*2*pi*f*t) dt exactly."
        ),
        "repo_modules": ["dgs/gs_cuda.py", "dgs/gs_core.py", "dgs/steam_imaging.py"],
        "ousd_cta": "Advanced Computing + Integrated Sensing",
    },

    "P5_BAYES": {
        "title": "Bayesian Real-Time Cancer Cell Detection in STEAM Blood Flow",
        "agency": "NCI SBIR (National Cancer Institute)",
        "phase": "Phase I",
        "budget": 275_000,
        "duration_months": 6,
        "topic": "Probabilistic classification of circulating tumor cells",
        "significance": (
            "Liquid biopsy (CTC detection from blood) is less invasive than tissue biopsy "
            "but requires finding 1 cell in 10^9 -- sensitivity/specificity tradeoff is severe. "
            "Current gold standard (CellSearch) misses 30-40% of CTCs. "
            "STEAM + GS phase + Bayes classifier can achieve 99.9% sensitivity."
        ),
        "innovation": (
            "Bayes theorem: P(CTC | features) = P(features | CTC) * P(CTC) / P(features). "
            "Features = {I_max, phi_mean, phi_std, morphology_entropy} from GS recovery. "
            "Prior P(CTC) = 1/10^6 (prevalence) -- makes Bayes essential to avoid false positives. "
            "With STEAM at 36 Mfps: 10^9 cells screened in 28 seconds from 6 mL blood draw."
        ),
        "approach": [
            "Month 1-2: Generate synthetic CTC dataset (Mie scattering phase profile)",
            "Month 3:   Train Gaussian Naive Bayes + MLP on {I, phi} features",
            "Month 4:   ROC curve; optimize threshold for 99.9% sensitivity",
            "Month 5-6: Prior sensitivity analysis; clinical false-positive rate projection",
        ],
        "milestones": {
            "M1": "Synthetic dataset: 10^6 cells, 1 ppm CTC prevalence",
            "M2": "Bayes classifier AUC > 0.999",
            "M3": "False positive rate < 1 per 10^6 normal cells",
            "M4": "NCI Phase II LOI submitted",
        },
        "griffiths_physics": (
            "Mie scattering cross section (Ch 9 scattering): sigma = (2*pi/k)^2 * sum |a_n|^2. "
            "CTC has larger radius -> larger sigma -> distinct phase profile. "
            "Bayes likelihood P(I|CTC) modeled as Mie scattering distribution."
        ),
        "repo_modules": ["dgs/bayes_inference.py", "dgs/steam_imaging.py",
                         "dgs/statistics.py", "dgs/hypothesis.py"],
        "ousd_cta": "Trusted AI + Human-Machine Interfaces",
    },

    "P6_ROGUE_BIO": {
        "title": "NLSE Rogue Wave Analogs in Neural Membrane Biophysics",
        "agency": "DARPA BTO (Biological Technologies Office)",
        "phase": "Phase I",
        "budget": 275_000,
        "duration_months": 6,
        "topic": "Optical detection of rare catastrophic events in neural tissue",
        "significance": (
            "Epileptic seizures and cardiac fibrillation are rare, extreme, spatially coherent "
            "events -- the biological analog of optical rogue waves. "
            "The nonlinear Schrodinger equation (NLSE) describes both: fiber MI instability "
            "and neural membrane potential instability follow the same math. "
            "STEAM can image the 2D wavefront of a seizure-onset in real time."
        ),
        "innovation": (
            "Map neural membrane potential V(x,t) to NLSE amplitude A(x,t): "
            "C_m * dV/dt = -I_ion(V) + I_ext  (Hodgkin-Huxley, Ch 7 analog) "
            "At threshold: modulation instability -> exponential growth of perturbations "
            "= optical rogue wave mechanism (dgs/nlse.py). "
            "STEAM optically reads V(x,t) via voltage-sensitive dye delta_n ~ 0.001."
        ),
        "approach": [
            "Month 1-2: NLSE MI analysis for HH parameters (dgs/nlse.py + dgs/cellular_biophysics.py)",
            "Month 3:   STEAM phase sensitivity for delta_n=0.001 voltage-sensitive dye",
            "Month 4:   Rogue wave detection algorithm: extreme value statistics (GEV distribution)",
            "Month 5-6: 2D simulation of seizure wavefront + STEAM image reconstruction",
        ],
        "milestones": {
            "M1": "NLSE MI gain spectrum for HH parameters at resting potential",
            "M2": "STEAM detects delta_n=0.001 at SNR > 10 dB (simulation)",
            "M3": "Rogue wave event detector: false alarm rate < 1/hour",
            "M4": "DARPA BTO white paper submitted",
        },
        "griffiths_physics": (
            "Ch 7: membrane current = displacement current analog (dV/dt term). "
            "Ch 9: NLSE = Schrodinger analog with cubic nonlinearity (solitons). "
            "Ch 4: voltage-sensitive dye delta_n from Lorentz oscillator shift."
        ),
        "repo_modules": ["dgs/nlse.py", "dgs/cellular_biophysics.py",
                         "dgs/steam_imaging.py", "dgs/classical_ed.py"],
        "ousd_cta": "Human-Machine Interfaces + Directed Energy",
    },

    "P7_PHOTONIC_AI": {
        "title": "Photonic AI Receiver: STEAM + Neural Phase Retrieval for FutureG",
        "agency": "OUSD(R&E) SBIR -- FutureG + Trusted AI CTAs",
        "phase": "Phase II (builds on P1 + P4)",
        "budget": 1_750_000,
        "duration_months": 24,
        "topic": "Real-time ML phase recovery for FutureG optical communications",
        "significance": (
            "6G optical links require coherent detection at 100+ Gbaud. "
            "Traditional coherent receivers need a local oscillator laser (LO) -- expensive, "
            "alignment-sensitive, and power-hungry. "
            "STEAM + GS phase recovery eliminates the LO: dispersion encodes phase into time, "
            "neural net (Paper [3], dgs/nn_spectral_regression.py) breaks conjugate ambiguity."
        ),
        "innovation": (
            "Complete LO-free coherent receiver: "
            "1. STEAM time-stretches received signal (H(f)=exp(i*pi*D*f^2)) "
            "2. ADC at 36 GHz (P4 CUDA pipeline) "
            "3. GS phase retrieval (50 iterations, CUDA, P4) "
            "4. NN conjugate-ambiguity resolver (dgs/nn_spectral_regression.py) "
            "5. Bayesian symbol detector (P5 Bayes framework) "
            "Full stack: this repo covers all 5 layers."
        ),
        "approach": [
            "Month 1-6:   P4 CUDA pipeline integration + P1 RogueGuard hardware",
            "Month 7-12:  NN training on 6G modulation formats (6-PSK, 64-QAM)",
            "Month 13-18: Lab prototype: 10 Gbaud STEAM receiver demo",
            "Month 19-24: Field trial at OUSD partner site; Phase III CRADA",
        ],
        "milestones": {
            "M1": "CUDA pipeline at 1 Mfps (Month 6)",
            "M2": "NN BER < 10^-9 at SNR 15 dB for 6-PSK (Month 12)",
            "M3": "10 Gbaud hardware prototype (Month 18)",
            "M4": "Field demo at government partner site (Month 24)",
        },
        "griffiths_physics": (
            "All of Ch 9 (wave propagation, dispersion, group velocity). "
            "Jackson Ch 7 (full complex n derivation of H(f)). "
            "This is the graduate-level physics foundation of the entire repo."
        ),
        "repo_modules": ["dgs/steam_imaging.py", "dgs/gs_core.py",
                         "dgs/nn_spectral_regression.py", "dgs/gs_cuda.py",
                         "dgs/bayes_inference.py", "dgs/photonic_ai.py"],
        "ousd_cta": "FutureG + Trusted AI + Integrated Sensing",
    },

    "P8_GENERATIVE_PRIOR": {
        "title": "Generative AI Priors for Dispersion-Assisted Phase Retrieval",
        "agency": "OUSD FutureG / Trusted AI and Autonomy",
        "phase": "Phase I",
        "budget": 275_000,
        "duration_months": 6,
        "topic": "Learned generative priors replacing hand-designed regularization in GS phase recovery",
        "significance": (
            "Classical GS (P1, P2, P7) and even the existing differentiable gradient-descent "
            "variant (dgs/dispersion_gs_descent.py) regularize the ill-posed low-diversity case "
            "with a single HAND-DESIGNED prior (field smoothness). A LEARNED generative prior -- "
            "a small network trained on the actual distribution of plausible signals for a given "
            "modulation format -- can encode far richer structure than a smoothness penalty alone, "
            "the same idea behind compressed-sensing-with-generative-priors and 'Deep Image Prior' "
            "work in the broader inverse-problems literature, applied here to dispersion-assisted "
            "phase retrieval specifically."
        ),
        "innovation": (
            "Replace dispersion_gs_descent.py's fixed smoothness regularizer with a trained "
            "generative network G(z) whose output is constrained to lie on the learned manifold "
            "of realistic fields; optimize argmin_z ||disperse(G(z))|^2 - I2||^2 instead of "
            "argmin_phi directly. dgs/gs_diff.py's existing differentiable forward model and "
            "dgs/torch/gs_layer.py's GSLayer module are the exact building blocks this reuses -- "
            "the forward physics (disperse, |.|^2) is unchanged and already tested; only the "
            "PRIOR being optimized against changes, from a fixed penalty to a learned network."
        ),
        "approach": [
            "Month 1-2: Train a small generative network (VAE or simple GAN) on synthetic "
            "signal distributions per modulation format (reusing dgs/gs_core.py's make_measurements "
            "as the data generator)",
            "Month 3-4: Wire the trained generator into dispersion_gs_descent.py's optimization "
            "loop as a drop-in prior replacement; compare convergence rate and final error against "
            "the existing smoothness-only baseline",
            "Month 5: Stress-test on the low-diversity (|D1-D2| small) regime where classical GS "
            "is known to fail -- this is precisely where a richer learned prior should help most",
            "Month 6: Benchmark report; Phase II proposal drafted if the generative prior shows "
            "a measurable convergence/robustness improvement over the existing baseline",
        ],
        "milestones": {
            "M1": "Generative network trained, reconstructs held-out synthetic signals with "
                  "correlation > 0.9 to ground truth",
            "M2": "Generative-prior GS integrated into dispersion_gs_descent.py's existing "
                  "optimization loop, runs end-to-end without regressing the current baseline",
            "M3": "Head-to-head comparison vs. smoothness-only prior: convergence rate, final "
                  "phase error, and robustness at low D-diversity, all measured (not projected)",
            "M4": "Phase II go/no-go decision based on M3's actual measured improvement, not "
                  "assumed in advance",
        },
        "griffiths_physics": (
            "Same H(f)=exp(i*pi*D*f^2) dispersion operator as every other proposal in this "
            "portfolio (Ch 9 GVD) -- P8 changes only the INVERSE PROBLEM's regularization "
            "strategy, not the forward physics, which is unchanged and already validated "
            "elsewhere in this repo."
        ),
        "repo_modules": ["dgs/gs_diff.py", "dgs/dispersion_gs_descent.py",
                         "dgs/torch/gs_layer.py", "dgs/gs_core.py"],
        "ousd_cta": "Trusted AI and Autonomy + FutureG",
        "honesty_note": (
            "This extends EXISTING, already-working code (gs_diff.py, dispersion_gs_descent.py, "
            "gs_layer.py all currently in this repo) -- it is not a from-scratch concept. M3's "
            "comparison is explicitly framed as an open question (go/no-go), not a foregone "
            "conclusion -- the smoothness prior may turn out to be competitive in some regimes, "
            "and the proposal should say so rather than assume the generative approach always wins."
        ),
    },
}


# ── Budget breakdown ──────────────────────────────────────────────────────────

def budget_breakdown(proposal_key):
    """Standard SBIR Phase I budget ($275K / 6 months, 3 people)."""
    p = PROPOSALS[proposal_key]
    B = p["budget"]
    if B == 275_000:
        return {
            "PI_salary_50pct_FTE": 45_000,
            "co_I_salary_25pct_FTE": 22_500,
            "research_assistant": 23_500,
            "fringe_benefits_30pct": 27_300,
            "equipment_ADC_GPU": 40_000,
            "supplies_consumables": 15_000,
            "travel_conferences": 8_000,
            "indirect_costs_26pct": 52_000,
            "subcontracts_university": 41_700,
            "total": 275_000,
        }
    else:  # Phase II $1.75M
        return {
            "PI_salary_2yr": 180_000,
            "co_Is_x2_2yr": 180_000,
            "postdoc": 130_000,
            "grad_student_x2": 110_000,
            "fringe_benefits": 180_000,
            "equipment_prototype": 250_000,
            "fab_PCB_fiber": 100_000,
            "travel": 40_000,
            "indirect_costs": 330_000,
            "subcontracts": 250_000,
            "total": 1_750_000,
        }


# ── Timeline ─────────────────────────────────────────────────────────────────

def portfolio_timeline():
    """Gantt-style timeline for submitting all 7 new proposals (P8 prioritized first)."""
    return [
        {"proposal": "P8 Generative Prior", "submit_quarter": "Q3-2026", "status": "READY -- submit FIRST (prioritized)"},
        {"proposal": "P1 RogueGuard",    "submit_quarter": "Q1-2026", "status": "EXISTING"},
        {"proposal": "P2 STEAM",          "submit_quarter": "Q4-2026", "status": "READY -- submit after P8"},
        {"proposal": "P3 CRISPR",         "submit_quarter": "Q1-2027", "status": "6 months after P2"},
        {"proposal": "P4 CUDA",           "submit_quarter": "Q2-2027", "status": "parallel with P3"},
        {"proposal": "P5 Bayes CTC",      "submit_quarter": "Q3-2027", "status": "after P2 data"},
        {"proposal": "P6 Rogue Bio",      "submit_quarter": "Q4-2027", "status": "after P5"},
        {"proposal": "P7 Photonic AI P2", "submit_quarter": "Q2-2028", "status": "Phase II after P1+P4"},
    ]


# ── Startup phase: SBIR -> commercialization ──────────────────────────────────

def startup_phase_roadmap():
    """The real steps between 'SBIR proposals drafted' and 'an actual
    company' -- SBIR Phase I/II are federally-funded R&D; Phase III is
    EXPLICITLY not SBIR-funded (no federal money comes with it) -- it's
    where a spinoff has to survive on private capital, non-SBIR federal
    contracts, or real customers. This function returns the honest
    ordered checklist, not a hype timeline; step 0 is a blocking
    prerequisite, not optional groundwork to circle back to later."""
    return [
        {
            "step": 0,
            "name": "IP ownership resolution",
            "blocking": True,
            "detail": (
                "If any core algorithm work (GS phase retrieval, the dispersion "
                "operator implementation, etc.) was developed while affiliated with "
                "Jalali's UCLA lab -- using lab resources, lab data, or building "
                "directly on lab code -- UCLA's standard invention-assignment policy "
                "may give the university a real claim on that IP. This must be "
                "resolved (via UCLA's tech transfer office, or an actual IP attorney) "
                "BEFORE incorporating or raising money around it. Getting this "
                "backwards is expensive to unwind later, not just embarrassing."
            ),
        },
        {
            "step": 1,
            "name": "Pick ONE product to lead with",
            "blocking": False,
            "detail": (
                "Not all 8 proposals at once. P1 (RogueGuard) is the most 'real "
                "hardware' candidate (physical 1U monitoring unit, RPi CM4 + dual "
                "ADC -- something a customer can actually buy and install). "
                "P8 (generative AI phase retrieval prior) is the most novel software "
                "IP but has no physical product yet. A startup needs ONE clear "
                "product story for investors/customers, not a research portfolio."
            ),
        },
        {
            "step": 2,
            "name": "Patent timing, if pursuing IP protection",
            "blocking": False,
            "detail": (
                "US patent law is first-to-file, and public disclosure BEFORE filing "
                "(a published paper, a public GitHub repo describing the invention, "
                "even some conference talks) can start a 1-year clock (US) or bar "
                "patentability entirely (most other countries, no grace period). "
                "If patent protection matters to the business plan, a provisional "
                "patent application should be evaluated BEFORE any further public "
                "disclosure -- this is a real, time-sensitive legal question, not "
                "busywork."
            ),
        },
        {
            "step": 3,
            "name": "Entity formation",
            "blocking": False,
            "detail": (
                "Delaware C-corp is the standard structure if outside (VC/angel) "
                "investment is the plan -- it's what most institutional investors "
                "expect and know how to fund. An LLC is simpler/cheaper if the near-"
                "term plan is government contracts (SBIR Phase II, direct sales) "
                "without outside equity investors. This choice is hard to reverse "
                "cheaply later, so it should follow from steps 0-1, not precede them."
            ),
        },
        {
            "step": 4,
            "name": "Non-dilutive funding first (SBIR Phase II, if P1/P7 track holds)",
            "blocking": False,
            "detail": (
                "SBIR Phase II ($1.75M, no equity given up) is real money that "
                "doesn't require an investor pitch -- worth pursuing before or "
                "alongside any private fundraising, since it directly funds building "
                "the P1/P7 hardware+software without diluting ownership."
            ),
        },
        {
            "step": 5,
            "name": "SBIR Phase III: the actual 'startup phase'",
            "blocking": False,
            "detail": (
                "Phase III has NO federal R&D funding attached -- by SBIR program "
                "design, this is where the company must commercialize on private "
                "capital, follow-on non-SBIR federal contracts, or direct sales. "
                "This is the actual transition point 'startup phase' refers to: "
                "everything before this is still federally-funded research, not "
                "yet a company with revenue or investors."
            ),
        },
    ]


# ── Photonics HARDWARE manufacturing funding (distinct from software SBIR) ────

def photonics_manufacturing_funding_landscape():
    """Real funding mechanisms specific to scaling an integrated-photonics
    HARDWARE product (e.g. the fab process in
    dgs/silicon_photonics_manufacturing_chemistry.py) from a chip design to
    a manufactured device -- a genuinely different, more capital-intensive
    path than the pure-software/algorithm SBIR proposals above. Figures
    below are representative industry ballparks (hedged as such), not
    precise official numbers -- verify current amounts before citing them
    in an actual proposal."""
    return [
        {
            "mechanism": "AIM Photonics (Manufacturing USA institute)",
            "type": "DoD-backed manufacturing institute, not a grant per se",
            "detail": (
                "AIM Photonics (headquartered at SUNY Polytechnic Institute) is a "
                "real, existing DoD-backed Manufacturing USA institute created "
                "specifically to scale US integrated-photonics manufacturing. It "
                "runs shared fabrication access programs and workforce training -- "
                "worth investigating membership/access terms directly rather than "
                "assuming a specific cost here."
            ),
        },
        {
            "mechanism": "Multi-Project Wafer (MPW) shuttle runs",
            "type": "shared fabrication, NOT a full dedicated wafer lot",
            "detail": (
                "Several designs share one mask set/wafer run, splitting the "
                "otherwise-prohibitive NRE (non-recurring engineering) cost across "
                "customers. Real providers include AIM Photonics, imec, and "
                "commercial silicon photonics foundries. Representative industry "
                "range: roughly tens of thousands to a few hundred thousand dollars "
                "per shuttle slot, depending on process and foundry -- get a current "
                "quote before budgeting, this varies a lot."
            ),
        },
        {
            "mechanism": "CHIPS and Science Act (2022)",
            "type": "federal legislation, multiple sub-programs",
            "detail": (
                "Real 2022 US legislation. Most headline CHIPS Act money targets "
                "leading-edge logic fab construction, but CHIPS R&D and related NSF "
                "programs may have eligibility relevant to photonics-adjacent "
                "manufacturing research -- this needs direct verification against "
                "current program solicitations, not assumed from the Act's general "
                "reputation."
            ),
        },
        {
            "mechanism": "SBIR Phase I/II (this file's P2-P8)",
            "type": "non-dilutive, but NOT enough alone for a manufactured product",
            "detail": (
                "$275K Phase I / $1.75M Phase II typically covers algorithm/"
                "prototype development and MAYBE an MPW shuttle slot -- but not a "
                "dedicated wafer lot, packaging line, or test infrastructure. "
                "Realistic path: SBIR funds the design + shared-shuttle validation; "
                "scaling to a manufactured product needs the other mechanisms here "
                "or private capital."
            ),
        },
        {
            "mechanism": "Private capital (seed / Series A)",
            "type": "dilutive, but often necessary for hardware scale-up",
            "detail": (
                "Photonics hardware startups typically need MORE capital, EARLIER, "
                "than a comparable software startup -- NRE, packaging/testing "
                "equipment, and longer enterprise/defense sales cycles all cost "
                "real money before first revenue. Investors in this space usually "
                "expect working MPW-validated silicon (not just simulation results) "
                "before writing a check -- which is exactly why the non-dilutive "
                "mechanisms above matter as a bridge."
            ),
        },
    ]


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  SBIR PORTFOLIO  P2-P8  --  Dispersion-Assisted GS Platform")
    print("=" * 65)

    for key, p in PROPOSALS.items():
        print(f"\n{'='*65}")
        print(f"  {key}: {p['title']}")
        print(f"  Agency: {p['agency']}  |  {p['phase']}  |  ${p['budget']:,}")
        print(f"  OUSD CTA: {p['ousd_cta']}")
        print(f"\n  SIGNIFICANCE (first sentence):")
        print(f"  {p['significance'][:120]}...")
        print(f"\n  INNOVATION (core claim):")
        print(f"  {p['innovation'][:120]}...")
        if "verified_citations" in p:
            print(f"\n  VERIFIED CITATIONS ({len(p['verified_citations'])}):")
            for cite in p["verified_citations"]:
                print(f"  - {cite[:100]}...")
        print(f"\n  Griffiths: {p['griffiths_physics'][:80]}")
        print(f"  Modules:   {', '.join(p['repo_modules'][:3])}")

    print(f"\n{'='*65}")
    print("  PORTFOLIO TIMELINE")
    print(f"{'='*65}")
    print(f"  {'Proposal':22s} {'Submit':12s} {'Status'}")
    for row in portfolio_timeline():
        print(f"  {row['proposal']:22s} {row['submit_quarter']:12s} {row['status']}")

    print(f"\n{'='*65}")
    print("  PHYSICS CHAIN CONNECTING ALL 8 PROPOSALS")
    print(f"{'='*65}")
    chain = [
        ("H(f)=exp(i*pi*D*f^2)", "Griffiths Ch9 GVD", "ALL proposals"),
        ("GS phase retrieval",    "dgs/gs_core.py",    "P1,P2,P3,P4,P5,P7,P8"),
        ("NLSE soliton/rogue",    "dgs/nlse.py",       "P1,P6"),
        ("CUDA cuFFT",            "dgs/gs_cuda.py",    "P4,P7"),
        ("Bayes classifier",      "dgs/bayes_inference.py","P5,P3,P7"),
        ("STEAM forward model",   "dgs/steam_imaging.py",  "P2,P3,P4,P5,P6"),
        ("NN conjugate resolver", "dgs/nn_spectral_regression.py","P5,P7"),
        ("Differentiable GS + generative prior", "dgs/gs_diff.py, dgs/dispersion_gs_descent.py", "P8"),
    ]
    for physics, module, proposals in chain:
        print(f"  {physics:30s} {module:32s} {proposals}")

    print(f"\n  TOTAL PORTFOLIO VALUE: $275K x6 (P2-P6,P8) + $1.75M (P7) = $3.4M")
    print(f"  NEXT ACTION: P8 (Generative AI Phase Retrieval Prior) prioritized first --")
    print(f"  draft executive summary -> OUSD FutureG/Trusted AI submission portal")

    print(f"\n{'='*65}")
    print("  STARTUP PHASE ROADMAP (SBIR -> commercialization)")
    print(f"{'='*65}")
    for item in startup_phase_roadmap():
        blocking_tag = "  <-- BLOCKING, do this first" if item["blocking"] else ""
        print(f"\n  Step {item['step']}: {item['name']}{blocking_tag}")
        print(f"  {item['detail']}")

    print(f"\n{'='*65}")
    print("  PHOTONICS HARDWARE MANUFACTURING FUNDING (distinct from software SBIR)")
    print(f"{'='*65}")
    for item in photonics_manufacturing_funding_landscape():
        print(f"\n  {item['mechanism']}  [{item['type']}]")
        print(f"  {item['detail']}")


if __name__ == "__main__":
    demo()
