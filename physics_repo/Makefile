# physics_repo build targets.
# Notebooks are generated from tools/build_NN_*.py, then executed in place.

PYTHON      ?= python
KERNEL      ?= python3
NBDIR        = notebooks
TOOLS        = tools
BUILDERS    := $(sort $(wildcard $(TOOLS)/build_*.py))
NOTEBOOKS   := $(patsubst $(TOOLS)/build_%.py,$(NBDIR)/%.ipynb,$(BUILDERS))
TIMEOUT     ?= 600

.PHONY: all help install test notebooks execute clean figures

help:
	@echo "make install    - install physkit (editable) and requirements"
	@echo "make test       - run unit tests in tests/"
	@echo "make notebooks  - regenerate and execute every notebook"
	@echo "make clean      - remove executed outputs and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e src

test:
	$(PYTHON) -m pytest -q tests

# Generate one notebook from its builder, then execute it in place.
$(NBDIR)/%.ipynb: $(TOOLS)/build_%.py
	$(PYTHON) $<
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=$(KERNEL) \
		--ExecutePreprocessor.timeout=$(TIMEOUT) $@

notebooks: $(NOTEBOOKS)

# Build without executing (fast syntax/generation check).
generate:
	@for b in $(BUILDERS); do $(PYTHON) $$b; done

clean:
	rm -rf $(NBDIR)/.ipynb_checkpoints tests/__pycache__ src/physkit/__pycache__ \
		figures/*.png .pytest_cache
