# photonics-ml-pipeline

Research-grade project bridging **computational physics, optics/photonics, feature
extraction, machine learning, and C code generation** into one reproducible pipeline:

```
Physics -> SymPy -> Numerical verification -> Feature extraction
        -> Machine learning -> Generated C -> Benchmarking -> Documentation
```

Everything is CPU-first and portable (Windows + Linux); no GPU is required.

## Directory structure

```
photonics_ml_pipeline/
  config.py               typed dataclass configuration + logging
  configs/                YAML configs
  physics/                SymPy symbolic engine + numeric Gaussian beam
  optics/                 ABCD ray/beam matrix optics
  photonics/              dispersion operator H_D(f) = exp(i pi D f^2)
  feature_extraction/     field/image -> interpretable feature vectors
  ml/                     PyTorch dataset, MLP, training, inference
  c_codegen/              SymPy -> C/Fortran/JS (with CSE), compile & run
  visualization/          Matplotlib figure builders
  datasets/               data location (synthetic samples generated on demand)
  include/  src/          generated C headers / sources (kept separate)
  tests/                  pytest: symbolic, numerical, features, codegen, ML
  examples/               run_pipeline.py end-to-end
  scripts/                benchmark.py, gen_docs.py, build_notebook.py
  notebooks/              research notebooks (theory -> ... -> generated C)
  docs/                   index.md + auto-generated api.md
```

## Quick start

```bash
pip install -e .                       # or: pip install -e .[dev,vision]
pytest -q                              # run the test suite
python examples/run_pipeline.py        # end-to-end; writes figures + generated C
python scripts/benchmark.py            # sympy vs numpy vs torch vs generated C
python scripts/gen_docs.py             # regenerate docs/api.md
python scripts/build_notebook.py       # (re)generate the research notebook
```

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| Physics / SymPy | `physics.symbolic` | derive `w(z)`, gradients, Hessian, CSE, lambdify |
| Numerics | `physics.gaussian_beam`, `optics.abcd`, `photonics.dispersion` | verify against independent formulations |
| Features | `feature_extraction.features` | 8 interpretable features from a field/image |
| ML | `ml.*` | PyTorch dataset/MLP classifying beam-divergence class |
| C codegen | `c_codegen.generator` | emit optimized C/Fortran/JS from finalized symbols |
| Benchmark | `scripts/benchmark.py` | time each backend |
| Docs | `scripts/gen_docs.py` | Markdown from docstrings |

## Design rules

- Python 3.12+, type hints, `dataclass`es, `pathlib`, `logging`, docstrings.
- Single-responsibility functions; no global mutable state (explicit `Config`).
- Generated code lives in `src/`/`include/`, never mixed with handwritten code.
- Every module documents purpose, equations, references, assumptions, limitations.
