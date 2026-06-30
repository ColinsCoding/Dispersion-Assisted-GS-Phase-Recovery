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
# INCOMPLETE — TODO for Phase I delivery
# ══════════════════════════════════════════════════════════════════════════════
# TODO: make adc        — real ADC capture pipeline (RPi CM4 driver)
# TODO: make tsdft-live — live TS-DFT from scope via VISA/pyvisa
# TODO: make train-fno  — full FNO training on experimental data
# TODO: make report     — compile LaTeX Phase I technical report
# TODO: make docker     — build Docker image for deployment
# TODO: make ci         — run full CI suite (lint + verify + notebooks)
