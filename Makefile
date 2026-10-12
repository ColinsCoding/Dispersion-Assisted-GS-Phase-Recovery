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
LAPLACE   = -m dgs.laplace_transform
LASERSAFETY = -m dgs.laser_safety
ENGECON   = -m dgs.engineering_economics
GNSSBATCHED = -m dgs.torch.gnss_batched
CO2FOCAL  = -m dgs.co2_laser_focal_picker
BUSARB    = -m dgs.master_slave_bus_arbitration
CRITPTS   = -m dgs.critical_points
SHORDLOG  = -m dgs.shor_discrete_log

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
	@echo "  make notebook-kk-certificate — KK-consistency certificate for GS phase retrieval"
	@echo "  make notebook-kk-attempts    — 3 attempts to fix (not just detect) the KK-ambiguity + symmetry search"
	@echo "  make notebook-kk      — both KK notebooks"
	@echo "  make laplace          — Laplace transform pairs (SymPy-verified) + Levy rational-fit regression demo"
	@echo "  make lasersafety      — NOHD / eyewear optical density for a real bench laser"
	@echo "  make engecon          — interest factors (F/P..A/G) + exact-bisection IRR + profitability index"
	@echo "  make notebook-latent-time-stretch    — FNO-style spectral layer learns exp(j*pi*D*f^2) exactly"
	@echo "  make notebook-laplace-maxwell        — Maxwell -> Laplace -> regression: why dispersion resists rational fits"
	@echo "  make notebook-laser-safety           — NOHD/OD notebook for a real HeNe bench setup"
	@echo "  make notebook-engineering-economics  — interest-factor toolkit + fixes ml_econ_physics.ipynb's crude IRR"
	@echo "  make notebook-chirp-log-diff         — chain rule + log-differentiation + matrix exp/log, one idea three ways"
	@echo "  make gnssbatched       — batched torch GNSS solve vs. the existing NumPy solver + Monte Carlo DOP check"
	@echo "  make bench-gnss        — mean+-std scaling-sweep benchmark, batched vs. looped GNSS solve"
	@echo "  make notebook-gnss-batched            — CUDA-shaped GNSS positioning notebook (RTX 4060 measured 20.7x)"
	@echo "  make notebook-session-0906    — all 6 notebooks added 2026-09-06"
	@echo "  make co2focal          — CO2 laser focal-length picker: recommended lens for cut/engrave jobs"
	@echo "  make busarb            — I2C wired-AND arbitration + SPI daisy-chain shift simulation"
	@echo "  make critpts           — relative max/min/saddle classification + a real neural-net saddle point"
	@echo "  make shordlog          — Shor's discrete-log algorithm (the half that threatens Ethereum, not RSA)"
	@echo "  make notebook-truss-symbolic    — SymPy parametric truss solver (fixed a broken dgs.statics import too)"
	@echo "  make notebook-shor-discrete-log — Shor discrete log + cited secp256k1/Ethereum threat estimates"
	@echo "  make notebook-kk-monte-carlo    — 200-trial statistical answer to the GS/KK ambiguity question"
	@echo "  make notebook-quadrupole-wavefunction — l=2 hydrogen states as surfaces + quadrupole term of the multipole expansion"
	@echo "  make notebook-fire-imaging      — RF-tagged fluorescence imaging: FDM, lock-in, lifetime roll-off, multiplex shot noise"
	@echo "  make notebook-units-dimensions  — Buckingham Pi, scaling laws, natural scales (dimensions as linear algebra)"
	@echo "  make notebook-qpe-certificate   — Phase II Thrust A prototype: two-scale QPE phase unwrapping (98.5% vs 17.5% naive)"
	@echo "  make notebook-c-pointers-memory — pointers/arrays/structs + stack/heap, incl. a real compiled C binary"
	@echo "  make notebook-circuit-laplace   — VCVS dependent sources + Laplace-domain circuit analysis"
	@echo "  make notebook-critical-points   — 2-var/eigenvalue extrema tests + a verified neural-net saddle"
	@echo "  make notebook-bus-arbitration   — I2C/SPI master-slave protocol notebook"
	@echo "  make notebook-photonics-shop    — real Shopify hobbyist shop analyzed via beam optics/safety/cost"
	@echo "  make notebook-shopify-revenue   — 12-month NPV launch model for a Shopify companion product"
	@echo "  make notebook-session-0906-part2 — the 7 notebooks above, as a batch"
	@echo "  make notebook-statics-hw2       — Hibbeler statics HW2 in SymPy + E&M, recursion, photonics, logic, Ethereum model"
	@echo "  make notebook-ch36-relativity   — special relativity by learning objective (36.1-36.8), all SymPy-checked"
	@echo "  make notebook-ch36-quiz-log     — Chapter 36 reading-quiz study log: attempts, status, a computation per question"
	@echo "  make notebook-ch37-modern-physics — spectra, Thomson e/m, simulated Millikan + Rutherford, classical atom collapse"
	@echo "  make notebook-photoelectric-qkd — photoelectric effect as y = mx + b (closed form, CUDA gradient descent, NN contrast) + BB84 QKD"
	@echo "  make notebook-sympy-drone       — quadcopter: hover power, Lagrange EOM in SymPy, LQR flight, cost per flight"
	@echo "  make notebook-drone-build       — pick drone parts: loops/conditions, Boolean polynomial, arm materials, laser/photomask, radio, POSIX sh"
	@echo "  make notebook-optical-science   — 7-stage optical science automation project (FFT, ADC, torch fit, classifier)"
	@echo "  make notebook-circuit-compiler  — graphs -> KCL/KVL -> netlist compiler -> Laplace -> logic probability algebra"
	@echo "  make notebook-boolean-braket    — Boolean -> polynomial -> probability vectors -> bras/kets; 100k-sample torch Monte Carlo (CUDA if present)"
	@echo "  make notebook-session-1002      — the 10 notebooks above, as a batch"
	@echo "  make kernels                    — list running python processes / Jupyter kernels with memory use (Windows)"
	@echo "  make kill-pid PID=12345         — stop one python process by PID (Windows; refuses non-python PIDs)"
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

.PHONY: laplace
laplace:
	$(PYTHON) $(LAPLACE)

.PHONY: lasersafety
lasersafety:
	$(PYTHON) $(LASERSAFETY)

.PHONY: engecon
engecon:
	$(PYTHON) $(ENGECON)

.PHONY: gnssbatched
gnssbatched:
	$(PYTHON) $(GNSSBATCHED)

.PHONY: bench-gnss
bench-gnss:
	$(PYTHON) scripts/bench_gnss_batched.py

.PHONY: co2focal
co2focal:
	$(PYTHON) $(CO2FOCAL)

.PHONY: busarb
busarb:
	$(PYTHON) $(BUSARB)

.PHONY: critpts
critpts:
	$(PYTHON) $(CRITPTS)

.PHONY: shordlog
shordlog:
	$(PYTHON) $(SHORDLOG)

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
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/rogue_wave_ai_detection.ipynb --output rogue_wave_ai_detection.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-jalali done: $(NB_DIR)/rogue_wave_ai_detection.ipynb"

.PHONY: notebook-rogue
notebook-rogue: notebook-jalali

# ══════════════════════════════════════════════════════════════════════════════
# KRAMERS-KRONIG / PHASE-RETRIEVAL CERTIFICATE  (added 2026-09-05)
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: notebook-kk-certificate
notebook-kk-certificate:
	$(PYTHON) scripts/build_kk_phase_retrieval_certificate_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/kk_phase_retrieval_certificate.ipynb --output kk_phase_retrieval_certificate.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-kk-certificate done: $(NB_DIR)/kk_phase_retrieval_certificate.ipynb"

.PHONY: notebook-kk-attempts
notebook-kk-attempts:
	$(PYTHON) scripts/build_kk_constrained_gs_attempts_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/kk_constrained_gs_attempts.ipynb --output kk_constrained_gs_attempts.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-kk-attempts done: $(NB_DIR)/kk_constrained_gs_attempts.ipynb"

.PHONY: notebook-kk
notebook-kk: notebook-kk-certificate notebook-kk-attempts

.PHONY: notebook-qpe-certificate
notebook-qpe-certificate:
	$(PYTHON) scripts/build_qpe_causality_certificate_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/qpe_causality_certificate.ipynb --output qpe_causality_certificate.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-qpe-certificate done: $(NB_DIR)/qpe_causality_certificate.ipynb"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION NOTEBOOKS  (added 2026-09-06: representation-vs-latent on the real
# dispersion operator, Laplace/Maxwell/regression, laser safety, engineering
# economics, and the chirp/log-diff/matrix-exp companion notebook)
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: notebook-latent-time-stretch
notebook-latent-time-stretch:
	$(PYTHON) scripts/build_latent_time_stretch_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/latent_time_stretch.ipynb --output latent_time_stretch.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-latent-time-stretch done: $(NB_DIR)/latent_time_stretch.ipynb"

.PHONY: notebook-laplace-maxwell
notebook-laplace-maxwell:
	$(PYTHON) scripts/build_laplace_maxwell_dispersion_regression_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/laplace_maxwell_dispersion_regression.ipynb --output laplace_maxwell_dispersion_regression.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-laplace-maxwell done: $(NB_DIR)/laplace_maxwell_dispersion_regression.ipynb"

.PHONY: notebook-laser-safety
notebook-laser-safety:
	$(PYTHON) scripts/build_laser_safety_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/laser_safety.ipynb --output laser_safety.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-laser-safety done: $(NB_DIR)/laser_safety.ipynb"

.PHONY: notebook-engineering-economics
notebook-engineering-economics:
	$(PYTHON) scripts/build_engineering_economics_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/engineering_economics_interest_factors.ipynb --output engineering_economics_interest_factors.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-engineering-economics done: $(NB_DIR)/engineering_economics_interest_factors.ipynb"

.PHONY: notebook-chirp-log-diff
notebook-chirp-log-diff:
	$(PYTHON) scripts/build_chirp_log_diff_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/chirp_log_diff.ipynb --output chirp_log_diff.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-chirp-log-diff done: $(NB_DIR)/chirp_log_diff.ipynb"

.PHONY: notebook-gnss-batched
notebook-gnss-batched:
	$(PYTHON) scripts/build_gnss_batched_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/gnss_batched.ipynb --output gnss_batched.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-gnss-batched done: $(NB_DIR)/gnss_batched.ipynb"

.PHONY: notebook-session-0906
notebook-session-0906: notebook-latent-time-stretch notebook-laplace-maxwell notebook-laser-safety notebook-engineering-economics notebook-chirp-log-diff notebook-gnss-batched

.PHONY: notebook-kk-monte-carlo
notebook-kk-monte-carlo:
	$(PYTHON) scripts/build_kk_certificate_monte_carlo_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/kk_certificate_monte_carlo.ipynb --output kk_certificate_monte_carlo.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-kk-monte-carlo done: $(NB_DIR)/kk_certificate_monte_carlo.ipynb"

.PHONY: notebook-c-pointers-memory
notebook-c-pointers-memory:
	$(PYTHON) scripts/build_c_pointers_memory_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/c_pointers_memory.ipynb --output c_pointers_memory.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-c-pointers-memory done: $(NB_DIR)/c_pointers_memory.ipynb"

.PHONY: notebook-circuit-laplace
notebook-circuit-laplace:
	$(PYTHON) scripts/build_circuit_laplace_dependent_sources_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/circuit_laplace_dependent_sources.ipynb --output circuit_laplace_dependent_sources.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-circuit-laplace done: $(NB_DIR)/circuit_laplace_dependent_sources.ipynb"

.PHONY: notebook-critical-points
notebook-critical-points:
	$(PYTHON) scripts/build_critical_points_neurons_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/critical_points_neurons.ipynb --output critical_points_neurons.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-critical-points done: $(NB_DIR)/critical_points_neurons.ipynb"

.PHONY: notebook-bus-arbitration
notebook-bus-arbitration:
	$(PYTHON) scripts/build_master_slave_bus_arbitration_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/master_slave_bus_arbitration.ipynb --output master_slave_bus_arbitration.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-bus-arbitration done: $(NB_DIR)/master_slave_bus_arbitration.ipynb"

.PHONY: notebook-photonics-shop
notebook-photonics-shop:
	$(PYTHON) scripts/build_amateur_photonics_shop_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/amateur_photonics_shop_optics.ipynb --output amateur_photonics_shop_optics.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-photonics-shop done: $(NB_DIR)/amateur_photonics_shop_optics.ipynb"

.PHONY: notebook-shopify-revenue
notebook-shopify-revenue:
	$(PYTHON) scripts/build_focal_picker_shopify_revenue_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/focal_picker_shopify_revenue.ipynb --output focal_picker_shopify_revenue.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-shopify-revenue done: $(NB_DIR)/focal_picker_shopify_revenue.ipynb"

.PHONY: notebook-truss-symbolic
notebook-truss-symbolic:
	$(PYTHON) scripts/build_truss_symbolic_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/truss_symbolic.ipynb --output truss_symbolic.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-truss-symbolic done: $(NB_DIR)/truss_symbolic.ipynb"

.PHONY: notebook-shor-discrete-log
notebook-shor-discrete-log:
	$(PYTHON) scripts/build_shor_discrete_log_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/shor_discrete_log_ethereum.ipynb --output shor_discrete_log_ethereum.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-shor-discrete-log done: $(NB_DIR)/shor_discrete_log_ethereum.ipynb"

.PHONY: notebook-quadrupole-wavefunction
notebook-quadrupole-wavefunction:
	$(PYTHON) scripts/build_quadrupole_wavefunction_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/quadrupole_wavefunction.ipynb --output quadrupole_wavefunction.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-quadrupole-wavefunction done: $(NB_DIR)/quadrupole_wavefunction.ipynb"

.PHONY: notebook-fire-imaging
notebook-fire-imaging:
	$(PYTHON) scripts/build_fire_imaging_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/fire_imaging.ipynb --output fire_imaging.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-fire-imaging done: $(NB_DIR)/fire_imaging.ipynb"

.PHONY: notebook-units-dimensions
notebook-units-dimensions:
	$(PYTHON) scripts/build_units_dimensions_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/units_dimensions.ipynb --output units_dimensions.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-units-dimensions done: $(NB_DIR)/units_dimensions.ipynb"

.PHONY: notebook-session-0906-part2
notebook-session-0906-part2: notebook-kk-monte-carlo notebook-c-pointers-memory notebook-circuit-laplace notebook-critical-points notebook-bus-arbitration notebook-photonics-shop notebook-shopify-revenue notebook-truss-symbolic notebook-shor-discrete-log

.PHONY: notebook-statics-hw2
notebook-statics-hw2:
	$(PYTHON) scripts/build_statics_hw2_sympy_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/statics_hw2_sympy.ipynb --output statics_hw2_sympy.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-statics-hw2 done: $(NB_DIR)/statics_hw2_sympy.ipynb"

.PHONY: notebook-ch36-relativity
notebook-ch36-relativity:
	$(PYTHON) scripts/build_ch36_relativity_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/ch36_relativity.ipynb --output ch36_relativity.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-ch36-relativity done: $(NB_DIR)/ch36_relativity.ipynb"

.PHONY: notebook-optical-science
notebook-optical-science:
	$(PYTHON) scripts/build_optical_science_automation_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/optical_science_automation.ipynb --output optical_science_automation.ipynb --ExecutePreprocessor.timeout=600
	@echo "  notebook-optical-science done: $(NB_DIR)/optical_science_automation.ipynb"

.PHONY: notebook-circuit-compiler
notebook-circuit-compiler:
	$(PYTHON) scripts/build_circuit_graph_compiler_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/circuit_graph_compiler.ipynb --output circuit_graph_compiler.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-circuit-compiler done: $(NB_DIR)/circuit_graph_compiler.ipynb"

.PHONY: notebook-boolean-braket
notebook-boolean-braket:
	$(PYTHON) scripts/build_boolean_to_braket_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/boolean_to_braket.ipynb --output boolean_to_braket.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-boolean-braket done: $(NB_DIR)/boolean_to_braket.ipynb"

.PHONY: notebook-ch37-modern-physics
notebook-ch37-modern-physics:
	$(PYTHON) scripts/build_ch37_foundations_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/ch37_foundations_modern_physics.ipynb --output ch37_foundations_modern_physics.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-ch37-modern-physics done: $(NB_DIR)/ch37_foundations_modern_physics.ipynb"

.PHONY: notebook-photoelectric-qkd
notebook-photoelectric-qkd:
	$(PYTHON) scripts/build_photoelectric_regression_qkd_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/photoelectric_regression_qkd.ipynb --output photoelectric_regression_qkd.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-photoelectric-qkd done: $(NB_DIR)/photoelectric_regression_qkd.ipynb"

.PHONY: notebook-sympy-drone
notebook-sympy-drone:
	$(PYTHON) scripts/build_sympy_drone_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/sympy_drone.ipynb --output sympy_drone.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-sympy-drone done: $(NB_DIR)/sympy_drone.ipynb"

.PHONY: notebook-drone-build
notebook-drone-build:
	$(PYTHON) scripts/build_drone_build_design_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/drone_build_design.ipynb --output drone_build_design.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-drone-build done: $(NB_DIR)/drone_build_design.ipynb"

.PHONY: notebook-ch36-quiz-log
notebook-ch36-quiz-log:
	$(PYTHON) scripts/build_ch36_reading_quiz_log_nb.py
	$(JUPYTER) nbconvert --to notebook --execute $(NB_DIR)/ch36_reading_quiz_log.ipynb --output ch36_reading_quiz_log.ipynb --ExecutePreprocessor.timeout=300
	@echo "  notebook-ch36-quiz-log done: $(NB_DIR)/ch36_reading_quiz_log.ipynb"

.PHONY: notebook-session-1002
notebook-session-1002: notebook-statics-hw2 notebook-ch36-relativity notebook-ch36-quiz-log notebook-ch37-modern-physics notebook-optical-science notebook-circuit-compiler notebook-boolean-braket notebook-photoelectric-qkd notebook-sympy-drone notebook-drone-build

# ── Windows process helpers (leaked Jupyter kernels exhaust the commit limit) ──
.PHONY: kernels
kernels:
	$(PYTHON) scripts/kernels.py list

.PHONY: kill-pid
kill-pid:
ifndef PID
	$(error usage: make kill-pid PID=12345 -- run 'make kernels' to find the PID)
endif
	$(PYTHON) scripts/kernels.py kill $(PID)

# ══════════════════════════════════════════════════════════════════════════════
# INCOMPLETE — TODO for Phase I delivery
# ══════════════════════════════════════════════════════════════════════════════
# TODO: make adc        — real ADC capture pipeline (RPi CM4 driver)
# TODO: make tsdft-live — live TS-DFT from scope via VISA/pyvisa
# TODO: make train-fno  — full FNO training on experimental data
# TODO: make report     — compile LaTeX Phase I technical report
# TODO: make docker     — build Docker image for deployment
# TODO: make ci         — run full CI suite (lint + verify + notebooks)
