# ══════════════════════════════════════════════════════════════════════════════
# Makefile — Dispersion-Assisted GS Phase Recovery
# OUSD(R&E): FutureG · Integrated Sensing · Trusted AI · Directed Energy
# Branch: gs-torch-nd   |   SBIR Phase I $275K
# ══════════════════════════════════════════════════════════════════════════════

PYTHON   ?= py -3.13
PYTHON312 ?= py -3.12
JUPYTER  ?= py -3.13 -m jupyter
NB_DIR    = notebooks
OUT_DIR   = outputs

# ── core modules now live under dgs/ (see commit 5f67140) ──────────────────────
GS_CORE   = -m dgs.gs_core
GS_FNO    = -m dgs.gs_fno
GS_TORCH  = -m dgs.torch.gs_layer
OUSD      = -m dgs.ousd_alignment
GRASS     = -m dgs.grass
FDTD      = -m dgs.fdtd
GRATING   = -m dgs.diffraction_grating
GRATING_VIEWER = -m dgs.viewer_diffraction_pygame
BENCH_GS  = -m dgs.photonic_pipeline_benchmark
FOOD      = -m dgs.food_science_kinetics
SOLIDSTATE = -m dgs.solid_state_physics
MICHELSON = -m dgs.michelson_morley
BANDTHEORY = -m dgs.band_theory
PST       = -m dgs.phase_stretch_transform
CHARGE    = -m dgs.charge_fundamentals
CYLCAP    = -m dgs.cylindrical_capacitor_bessel
PARAXIAL  = -m dgs.paraxial_optics_abcd
QINTERNET = -m dgs.quantum_internet_link_budget
LTI       = -m dgs.lti_systems
HEATFT    = -m dgs.heat_equation_fourier
DOPPLERNUM = -m dgs.doppler_numerical_derivation
DUALAD    = -m dgs.dual_autodiff
PIERCE    = -m dgs.pierce_oscillator
CTYPES    = -m dgs.c_type_precision
DIMANALYSIS = -m dgs.dimensional_analysis
VECGEOM   = -m dgs.vector_calculus_geometric
CHIRPLOG  = -m dgs.chirp_log_diff
ANALOGCOMP = -m dgs.analog_computing_universality
TDR       = -m dgs.transmission_line_tdr
CURLDIVQM = -m dgs.curl_div_modern_physics
POLYGLOT  = -m dgs.circuits_polyglot
TRUSS_FEM = -m dgs.torch.truss_fem
GRADXFORM = -m dgs.torch.gradient_transform_verify
HARMONICGRAD = -m dgs.torch.harmonic_gradient_fields

.DEFAULT_GOAL := help

# ══════════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: help
help:
	@echo ""
	@echo "  Dispersion-Assisted GS Phase Recovery"
	@echo "  ──────────────────────────────────────"
	@echo "  make gs          — run GS phase retrieval demo"
	@echo "  make fno         — train FNO on synthetic TS-DFT data"
	@echo "  make test        — run full pytest suite (tests/)"
	@echo "  make ousd        — print OUSD CTA alignment table"
	@echo "  make grass       — render DoD grass field (Markov + EM)"
	@echo "  make fdtd        — 1D Yee-grid FDTD vs analytic Fabry-Perot slab spectrum"
	@echo "  make grating     — N-slit diffraction grating physics demo"
	@echo "  make grating-viewer — interactive Pygame N-slit diffraction grating viewer"
	@echo "  make bench-gs    — measured wall-clock timing of the cuda_photonic_ai GS pipeline"
	@echo "  make food        — food science kinetics demo (D/z/F-value, Q10, shelf life)"
	@echo "  make solidstate  — solid state physics (crystal packing, Fermi energy, semiconductors)"
	@echo "  make michelson   — Michelson-Morley interferometer geometry + historical null result"
	@echo "  make bandtheory  — Kronig-Penney band theory: SymPy-derived, NumPy/Torch swept"
	@echo "  make pst         — Phase-Stretch Transform (PhyCV-style physics edge detection)"
	@echo "  make charge      — Griffiths' 3 charge facts: quantization, conservation, XNOR sign rule"
	@echo "  make cylcap      — Bessel-function BVP: potential inside a grounded cylindrical can"
	@echo "  make paraxial    — ABCD ray matrices + Gaussian beam q-parameter (binomial-derived)"
	@echo "  make qinternet   — entangled-photon link budget: UC Merced<->Riverside, fiber vs satellite"
	@echo "  make lti         — impulse/step/frequency response of RC filter, convolution theorem verified"
	@echo "  make heatft      — heat/diffusion PDE solved as Fourier-space per-mode decay, 3 methods cross-checked"
	@echo "  make dopplernum  — relativistic Doppler shift derived numerically from a simulated emitter/receiver couple"
	@echo "  make dualad      — the derivative, 3 ways: math (limit), physics (rate of change), computer (autodiff/finite-diff)"
	@echo "  make pierce      — Pierce crystal oscillator: series/parallel resonance, inductive region, load-cap frequency pulling"
	@echo "  make ctypes      — char/int/float/double compiled+run in real gcc, IEEE 754 vs NumPy, ties to precision bugs found this session"
	@echo "  make dimanalysis — dimensional analysis of Maxwell's-equations-adjacent formulas + Mars Climate Orbiter case study"
	@echo "  make gradxform   — Griffiths 1.14 verified 3 ways: SymPy symbolic, torch autograd (original + direct rotated frame)"
	@echo "  make vecgeom     — geometric div/curl/Laplacian via flux/circulation/neighbor-avg + Faraday's law curl check"
	@echo "  make chirplog    — chirp instantaneous frequency via log-differentiation of analytic signal + matrix exp/log"
	@echo "  make analogcomp  — molecular/mechanical/RLC/op-amp analog computer all solve the SAME ODE, verified"
	@echo "  make tdr         — skin effect (log-differentiation) + transmission lines + a simulated TDR measurement"
	@echo "  make curldivqm   — curl/div in modern physics: QM probability continuity + Aharonov-Bohm effect"
	@echo "  make harmonicgrad — Problem 1.20 as linear algebra (Jacobian trace/antisymmetric part) via torch, + generator theorem"
	@echo "  make polyglot    — series RLC RK4 run for real in C, MATLAB, and Python, cross-checked"
	@echo "  make bridge      — torch differentiable truss FEM demo (py-3.12) + Unreal trajectory export"
	@echo "  make poker       — holographic poker demo"
	@echo "  make uncertainty — HUP, TBP, vector spaces, Wigner, Bayes, stats"
	@echo "  make cuda-ai     — GPU GS + attention + Bayesian + publishable pipeline"
	@echo "  make jalali      — run Jalali modern physics demo (DFT/STEAM/rogue)"
	@echo "  make coppinger   — run Coppinger/Jalali 1999 TS-ADC paper demo"
	@echo "  make grammar     — run PDL grammar parser demo (parse photonic system)"
	@echo "  make pts-grammar — evaluate Coppinger 1999 system via PDL grammar"
	@echo "  make notebook-jalali  — build + execute rogue_wave_ai_detection.ipynb"
	@echo "  make notebook-rogue   — alias for notebook-jalali"
	@echo "  make smoke-jalali     — quick smoke tests for jalali + grammar"
	@echo "  make smoke-coppinger  — quick smoke tests for coppinger_jalali_1999"
	@echo "  make notebooks   — execute all notebooks in $(NB_DIR)/"
	@echo "  make lab         — launch JupyterLab"
	@echo "  make nb          — launch classic Jupyter Notebook"
	@echo "  make lint        — flake8 + ruff on dgs/"
	@echo "  make fmt         — black autoformat dgs/"
	@echo "  make profile     — cProfile dgs.gs_core"
	@echo "  make clean       — remove pyc, __pycache__, .ipynb_checkpoints"
	@echo "  make clean-all   — clean + remove outputs/"
	@echo "  make status      — git status + last 5 commits"
	@echo "  make push        — git push origin main"
	@echo ""

# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNS
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: gs
gs:
	$(PYTHON) $(GS_CORE)

.PHONY: fno
fno:
	$(PYTHON) $(GS_FNO)

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -q

.PHONY: ousd
ousd:
	$(PYTHON) $(OUSD)

.PHONY: grass
grass:
	$(PYTHON) $(GRASS)

.PHONY: fdtd
fdtd:
	$(PYTHON) $(FDTD)

.PHONY: grating
grating:
	$(PYTHON) $(GRATING)

.PHONY: grating-viewer
grating-viewer:
	$(PYTHON) $(GRATING_VIEWER)

.PHONY: bench-gs
bench-gs:
	$(PYTHON) $(BENCH_GS)

.PHONY: food
food:
	$(PYTHON) $(FOOD)

.PHONY: solidstate
solidstate:
	$(PYTHON) $(SOLIDSTATE)

.PHONY: michelson
michelson:
	$(PYTHON) $(MICHELSON)

.PHONY: bandtheory
bandtheory:
	$(PYTHON) $(BANDTHEORY)

.PHONY: pst
pst:
	$(PYTHON) $(PST)

.PHONY: charge
charge:
	$(PYTHON) $(CHARGE)

.PHONY: cylcap
cylcap:
	$(PYTHON) $(CYLCAP)

.PHONY: paraxial
paraxial:
	$(PYTHON) $(PARAXIAL)

.PHONY: qinternet
qinternet:
	$(PYTHON) $(QINTERNET)

.PHONY: lti
lti:
	$(PYTHON) $(LTI)

.PHONY: heatft
heatft:
	$(PYTHON) $(HEATFT)

.PHONY: dopplernum
dopplernum:
	$(PYTHON) $(DOPPLERNUM)

.PHONY: dualad
dualad:
	$(PYTHON) $(DUALAD)

.PHONY: pierce
pierce:
	$(PYTHON) $(PIERCE)

.PHONY: ctypes
ctypes:
	$(PYTHON) $(CTYPES)

.PHONY: dimanalysis
dimanalysis:
	$(PYTHON) $(DIMANALYSIS)

.PHONY: polyglot
polyglot:
	$(PYTHON) $(POLYGLOT)

.PHONY: bridge
bridge:
	$(PYTHON312) $(TRUSS_FEM)

.PHONY: gradxform
gradxform:
	$(PYTHON312) $(GRADXFORM)

.PHONY: harmonicgrad
harmonicgrad:
	$(PYTHON312) $(HARMONICGRAD)

.PHONY: vecgeom
vecgeom:
	$(PYTHON) $(VECGEOM)

.PHONY: chirplog
chirplog:
	$(PYTHON) $(CHIRPLOG)

.PHONY: analogcomp
analogcomp:
	$(PYTHON) $(ANALOGCOMP)

.PHONY: tdr
tdr:
	$(PYTHON) $(TDR)

.PHONY: curldivqm
curldivqm:
	$(PYTHON) $(CURLDIVQM)

.PHONY: poker
poker:
	$(PYTHON) holographic_poker.py

# ══════════════════════════════════════════════════════════════════════════════
# JUPYTER
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: lab
lab:
	$(JUPYTER) lab

.PHONY: nb
nb:
	$(JUPYTER) notebook

.PHONY: notebooks
notebooks:
	@echo "Executing all notebooks in $(NB_DIR)/ ..."
	@for nb in $(NB_DIR)/*.ipynb; do \
		echo "  ▸ $$nb"; \
		$(JUPYTER) nbconvert --to notebook --execute --inplace "$$nb" \
		    --ExecutePreprocessor.timeout=300 2>&1 | tail -1; \
	done

# ══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: lint
lint:
	$(PYTHON) -m flake8 --max-line-length=100 --ignore=E203,W503 dgs/ || true
	$(PYTHON) -m ruff check dgs/ || true

.PHONY: fmt
fmt:
	$(PYTHON) -m black --line-length 100 dgs/

.PHONY: profile
profile:
	$(PYTHON) -m cProfile -o gs_core.prof $(GS_CORE)
	$(PYTHON) -m pstats gs_core.prof

# ══════════════════════════════════════════════════════════════════════════════
# GIT
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: status
status:
	@git status --short
	@echo ""
	@git log --oneline -5

.PHONY: push
push:
	git push origin main

# ══════════════════════════════════════════════════════════════════════════════
# CLEAN
# ══════════════════════════════════════════════════════════════════════════════
.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.prof" -delete
	@echo "  clean ✓"

.PHONY: clean-all
clean-all: clean
	rm -rf $(OUT_DIR)/
	@echo "  clean-all ✓"

# ══════════════════════════════════════════════════════════════════════════════
# JALALI / COPPINGER / PDL GRAMMAR  (added 2026-07-02)
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: uncertainty
uncertainty:
	$(PYTHON) -m dgs.uncertainty_qm

.PHONY: cuda-ai
cuda-ai:
	$(PYTHON) -m dgs.cuda_photonic_ai

.PHONY: jalali
jalali:
	$(PYTHON) -m dgs.jalali_modern_physics

.PHONY: coppinger
coppinger:
	$(PYTHON) -m dgs.coppinger_jalali_1999

.PHONY: grammar
grammar:
	$(PYTHON) -m dgs.grammar_pts

.PHONY: pts-grammar
pts-grammar:
	$(PYTHON) -c "from dgs.grammar_pts import evaluate; import json; r=evaluate('LASER(P=0.001) -> EDFA(G=30) -> FIBER(D=17,L=5) -> EOM(Vpi=3.5,IL=3) -> FIBER(D=17,L=45) -> PD(R=0.8) -> ADC(fs=2,ENOB=8) -> GS(n=50,D=5000)'); print('M =', r['stretch']['M']); print('B_RF =', r['stretch']['B_RF_GHz'], 'GHz'); print('SNR =', round(r['power']['SNR_dB'],1), 'dB'); print('warnings:', r['warnings'])"

.PHONY: smoke-jalali
smoke-jalali:
	$(PYTHON) -m pytest tests/test_jalali_modern_physics.py tests/test_grammar_pts.py -q

.PHONY: smoke-coppinger
smoke-coppinger:
	$(PYTHON) -m pytest tests/test_coppinger_jalali_1999.py -q 2>/dev/null || \
		$(PYTHON) -c "from dgs.coppinger_jalali_1999 import coppinger_1999_stretch_factor; r=coppinger_1999_stretch_factor(); print('M =', r['M'], '(paper: 10)'); print('T_w =', r['T_window_ps'], 'ps (paper: 850 ps)')"

.PHONY: notebook-jalali
notebook-jalali:
	$(PYTHON) scripts/build_rogue_wave_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/rogue_wave_ai_detection.ipynb --output $(NB_DIR)/rogue_wave_ai_detection.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-jalali done: $(NB_DIR)/rogue_wave_ai_detection.ipynb"

.PHONY: notebook-rogue
notebook-rogue: notebook-jalali

# ══════════════════════════════════════════════════════════════════════════════
# INCOMPLETE — TODO for Phase I delivery
# ══════════════════════════════════════════════════════════════════════════════
# TODO: make adc        — real ADC capture pipeline (RPi CM4 driver)
# TODO: make tsdft-live — live TS-DFT from scope via VISA/pyvisa
# TODO: make train-fno  — full FNO training on experimental data
# TODO: make report     — compile LaTeX Phase I technical report
# TODO: make docker     — build Docker image for deployment
# TODO: make ci         — run full CI suite (lint + verify + notebooks)
