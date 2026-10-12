# statics_truss -- a real Unix C program for 2D truss statics (method of
# joints), cross-validated against dgs.truss (NumPy) and dgs.truss_symbolic
# (SymPy). See statics_truss.c's own header comment for the input format
# and Unix-philosophy design (stdin/file in, stdout out, exit codes).
#
#   make build    compile statics_truss.c with gcc
#   make run      build (if needed) + run on the reference 3-bar truss
#   make verify   build + run + cross-check against the two Python solvers
#   make mechanism  demonstrate the exit-code-2 path (a non-determinate truss)
#   make clean    remove build artifacts
#
# NOTE (Windows): use PowerShell, not the Bash tool, for `make build`/`run`/
# `verify` -- gcc's PATH doesn't resolve the same way through the Bash tool
# on this machine (a documented quirk elsewhere in this repo, unrelated to
# this program itself).
#
# NOTE (Windows): `make clean` uses `rm`, which isn't on PATH under a plain
# PowerShell/cmd `make` -- same known limitation as hardware/mie_scattering/
# Makefile. Run `make clean` from Git Bash (or WSL) instead, or delete the
# built executable by hand.

CC     = gcc
CFLAGS = -O2 -Wall -Wextra
PYTHON = py -3.13
EXE    = statics_truss$(if $(filter Windows_NT,$(OS)),.exe,)

build: $(EXE)

$(EXE): statics_truss.c
	$(CC) $(CFLAGS) -o $(EXE) statics_truss.c -lm

run: build
	./$(EXE) three_bar_truss.txt

verify: build
	$(PYTHON) verify_cross_language.py

mechanism: build
	-./$(EXE) mechanism.txt
	@echo "(exit code 2 above is expected -- this truss is a mechanism, not an error in the program)"

clean:
	rm -f $(EXE) statics_truss.exe statics_truss

.PHONY: build run verify mechanism clean
