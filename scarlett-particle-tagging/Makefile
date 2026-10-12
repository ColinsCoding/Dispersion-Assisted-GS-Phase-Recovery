# Scarlett Particle Tagging -- common tasks.   make help
#
# Activate a virtual environment first, or override: make test PYTHON=.venv/bin/python
# On Windows without `make`, run the commands shown in the README instead.

PYTHON ?= python3
MATLAB ?= matlab
CC     ?= gcc

.PHONY: help install test data plots calibrate c c-test notebook matlab-test matlab-plots all clean

help:
	@echo "make install       install Python dependencies and this package (editable)"
	@echo "make test          run the Python unit tests (includes C parity if a compiler exists)"
	@echo "make data          generate the synthetic dataset           -> results/"
	@echo "make plots         reproduce the main plots                  -> results/"
	@echo "make calibrate     calibration + uncertainty analysis        -> results/"
	@echo "make c             build the C estimator                     -> c/scarlett_estimator"
	@echo "make c-test        run the C estimator on exported vectors"
	@echo "make notebook      rebuild and execute the notebook"
	@echo "make matlab-test   run the MATLAB unit tests (needs MATLAB)"
	@echo "make matlab-plots  MATLAB plots and parity CSV files         -> results/matlab/"
	@echo "make all           data, plots, calibrate, notebook, test"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -v

data:
	$(PYTHON) scripts/generate_synthetic_data.py

plots:
	$(PYTHON) scripts/make_plots.py

calibrate:
	$(PYTHON) scripts/calibrate_and_uncertainty.py

c: c/scarlett_estimator

c/scarlett_estimator: c/main.c c/scarlett_estimator.c c/scarlett_estimator.h
	$(CC) -std=c99 -O2 -Wall -Wextra -o $@ c/main.c c/scarlett_estimator.c -lm

c-test: c
	$(PYTHON) scripts/export_c_test_vectors.py
	./c/scarlett_estimator estimate results/c_vectors/estimate.txt
	./c/scarlett_estimator acf results/c_vectors/acf.txt | head -3

notebook:
	$(PYTHON) scripts/build_notebook.py
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=600 notebooks/scarlett_main_plots.ipynb

matlab-test:
	$(MATLAB) -batch "addpath('matlab'); scarlett_setup; test_rayleigh_mie_limits"

matlab-plots:
	$(MATLAB) -batch "addpath('matlab'); scarlett_setup; run_scarlett_plots"

all: data plots calibrate notebook test

clean:
	find results -mindepth 1 ! -name .gitkeep -delete
	rm -f c/scarlett_estimator c/scarlett_estimator.exe
	rm -rf .pytest_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
