# Mie scattering -- single-sphere BHMIE algorithm in CUDA, cross-checked
# against an independent Python reimplementation of the same algorithm.
#   make build    compile mie_kernel.cu + mie_main.cu with nvcc
#   make run      build (if needed) + run, writes mie_output.csv
#   make verify   run + cross-check against generate_mie_reference.py
#   make all      verify (the full pipeline)
#   make clean    remove build artifacts and generated output
#
# NOTE (Windows): nvcc needs cl.exe on PATH. If `make build` fails with
# "Cannot find compiler 'cl.exe'", add the MSVC bin directory first, e.g.:
#   set PATH=%PATH%;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64
# then re-run make from the same shell.
#
# NOTE (Windows): `make clean` uses `rm`, which isn't on PATH under a
# plain PowerShell/cmd `make` -- run `make clean` from Git Bash (or
# WSL) instead, or delete mie.exe/mie_output.csv by hand.

NVCC   = nvcc
PYTHON = py -3.13
EXE    = mie.exe

build: $(EXE)

$(EXE): mie_kernel.cu mie_main.cu mie.h
	$(NVCC) mie_kernel.cu mie_main.cu -o $(EXE)

run: build
	./$(EXE)

verify: run
	$(PYTHON) generate_mie_reference.py

all: verify

clean:
	rm -f $(EXE) mie_output.csv *.exp *.lib *.obj

.PHONY: build run verify all clean
