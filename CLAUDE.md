# CLAUDE.md

Dispersion-assisted Gerchberg-Saxton phase recovery, grown into a teaching/research repo across
photonics, quantum information, EE and physics. Owner: Colin Casey (GitHub ColinsCoding). Origin:
ECE 279AS project with Gabriel Morozowsky, Prof. Bahram Jalali, UCLA, 2024.

Latest session context: `docs/handoff_2026-10-10_tdgsa_qis_qec.md` (read it first when continuing
TDGSA, qubit GS, or QEC work).

## Layout and conventions

* `dgs/<topic>.py`: library modules with a long plain-English docstring at the top; `tests/test_<topic>.py` alongside (pytest).
* Notebooks are generated: `scripts/build_<name>_nb.py` (nbformat) writes `notebooks/<name>.ipynb`. Edit the builder, not the notebook, then rebuild and execute.
* Figures: Okabe-Ito palette, curves distinguished by line style as well as color, and a printed text description of every figure.
* Each notebook starts with an "In plain words" section, then the math (SymPy where possible), then numbers checked against closed forms.
* `projects/<name>/`: self-contained sub-projects (e.g. `projects/tdgsa_qis/`: Python, ANSI C, C++/POSIX, CUDA).
* Citations in recent work are IEEE style.

## Running things

* Tests: `PYTHONPATH=. python -m pytest tests/test_<topic>.py -q`.
* Jupyter: use the **py312** kernel (Python 3.12, Qiskit installed). The `circuits32` conda env's matplotlib crashes the kernel on the first plot.
* CUDA builds need MSVC loaded first (`vcvars64.bat` from VS 2022 Community); RTX 4060 → `-arch=sm_89`; CUDA 13.0.

## Working with Colin

* Prompts are often short keyword lists; infer the deliverable, build it, verify it runs, and open it.
* Explain results in plain English as well as math; keep setup steps short and numbered.
* The working tree usually has many uncommitted changes: never mass-commit or clean; commit only when asked, and only the relevant files.
* State limitations and anything unverified (e.g. references written from memory) plainly.
