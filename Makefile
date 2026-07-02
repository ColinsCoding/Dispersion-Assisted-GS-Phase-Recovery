# ══════════════════════════════════════════════════════════════════════════════
# Makefile — Dispersion-Assisted GS Phase Recovery
# OUSD(R&E): FutureG · Integrated Sensing · Trusted AI · Directed Energy
# Branch: gs-torch-nd   |   SBIR Phase I $275K
# ══════════════════════════════════════════════════════════════════════════════

PYTHON   ?= py -3.13
JUPYTER  ?= py -3.13 -m jupyter
NB_DIR    = notebooks
OUT_DIR   = outputs

# ── core modules now live under dgs/ (see commit 5f67140) ──────────────────────
GS_CORE   = -m dgs.gs_core
GS_FNO    = -m dgs.gs_fno
GS_TORCH  = -m dgs.torch.gs_layer
OUSD      = -m dgs.ousd_alignment
GRASS     = -m dgs.grass

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
