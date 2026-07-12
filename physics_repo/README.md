# physics_repo

A reproducible educational repository bridging mathematics, physics, and scientific software for a
computer engineering student preparing for graduate-level photonics, spectroscopy, computational
imaging, and scientific software.

The repository develops one continuous line of reasoning: mathematics builds physics, physics builds
instruments, instruments build data, and software turns data into scientific interpretation. Each
notebook reuses code from earlier notebooks through the installable `physkit` package in `src/`.

## Layout

```
physics_repo/
    README.md              this file
    Makefile               build, test, and execute targets
    requirements.txt       pinned scientific-Python stack (PyTorch optional)
    src/physkit/           reusable library: constants, units, linear algebra, plotting
    tests/                 unit tests for physkit (pytest)
    tools/                 notebook generators (build_NN_*.py) and shared nbkit helper
    notebooks/             executed .ipynb chapters 00..28
    docs/                  template and writing conventions
    figures/              generated figures (gitignored)
    data/                 sample and generated datasets
```

## Notebook order

| # | chapter | physics | mathematics | software |
|---|---------|---------|-------------|----------|
| 00 | scientific_python | reproducibility | numerics | Python/NumPy/SymPy |
| 01 | units_dimensions | SI, dimensional analysis | groups of exponents | `physkit.units` |
| 02 | vectors | fields, forces | vector spaces | NumPy arrays |
| 03 | complex_numbers | phasors, waves | complex plane | NumPy complex |
| 04 | precalculus_review | oscillation | functions, trig | SymPy |
| 05 | calculus_review | motion, flux | derivative, integral | SymPy/SciPy |
| 06 | linear_algebra | superposition | matrices, bases | NumPy linalg |
| 07 | eigenvalues | normal modes, Hamiltonians | spectra | `physkit.linalg` |
| 08 | differential_equations | dynamics | ODEs | SciPy integrate |
| 09 | fourier_series | periodic signals | orthogonal bases | NumPy |
| 10 | fourier_transform | spectra | FT pairs | NumPy FFT |
| 11 | electromagnetism | Maxwell, waves | vector calculus | NumPy |
| 12 | modern_physics | quanta | operators | SymPy/NumPy |
| 13 | hydrogen_atom | central force | radial equation | NumPy eigensolve |
| 14 | atomic_structure | quantum numbers | angular momentum | NumPy |
| 15 | helium | two electrons | perturbation | NumPy |
| 16 | screening | effective charge | variational | SciPy optimize |
| 17 | hartree | self-consistency | fixed point | NumPy |
| 18 | hartree_fock | exchange | eigenproblem | NumPy |
| 19 | periodic_table | shells | aufbau | Pandas |
| 20 | xray_spectra | inner shells | transitions | Pandas |
| 21 | moseley | Z scaling | regression | NumPy/Pandas |
| 22 | spectroscopy | selection rules | matrix elements | NumPy |
| 23 | phase_retrieval | diffraction | Gerchberg-Saxton | NumPy/SciPy |
| 24 | quadrature | integration | numerical analysis | SciPy |
| 25 | photonics | dispersion, guiding | wave equation | NumPy |
| 26 | signal_processing | sampling, noise | DFT, filters | SciPy |
| 27 | cuda | acceleration | roofline | PyTorch (optional) |
| 28 | capstone | complete instrument | full pipeline | all of the above |

## Every notebook answers

- What physics problem?
- What mathematics?
- What algorithm?
- What software?
- What experiment?
- What engineering application?

and follows subject-verb-object scientific writing (electron occupies orbital; detector measures
intensity; GPU computes FFT).

## Reproduce

```
pip install -r requirements.txt
pip install -e src            # installs physkit
make test                     # run unit tests
make notebooks                # regenerate and execute all notebooks
```

PyTorch is optional. Notebooks import it through `physkit.optional_torch()` and degrade to NumPy when
it is absent, so the repository runs on a CPU-only scientific-Python install.
