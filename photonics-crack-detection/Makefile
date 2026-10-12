# photonics-crack-detection -- common tasks.
# Usage:  make help
#
# Activate a virtual environment first (see README), or override the
# interpreter:  make test PYTHON=.venv/bin/python

PYTHON ?= python3
INPUT  ?= data/example.png
OUTPUT ?= results/example

# Makes the package importable without `pip install -e .`
export PYTHONPATH := src

.PHONY: help install test demo notebook clean

help:
	@echo "make install   install dependencies and this package (editable)"
	@echo "make test      run the unit tests"
	@echo "make demo      headless demo on the synthetic example image"
	@echo "make notebook  execute the notebook in place (no browser needed)"
	@echo "make clean     delete generated results and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -v

# --make-example creates data/example.png (seed 0) only if it is missing.
demo:
	$(PYTHON) -m crack_photonics.cli --input $(INPUT) --output $(OUTPUT) --headless --make-example

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		notebooks/symmetry_crack_detection.ipynb

clean:
	find results -mindepth 1 ! -name .gitkeep -delete
	rm -f data/example.png
	rm -rf .pytest_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
