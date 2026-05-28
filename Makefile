NOTEBOOK     := phase_retrieval.ipynb
SEALS_NB     := notebooks/seals_simulation.ipynb
PYTHON       := python
PIP          := pip

# Firmware cross-compilation (aarch64 Linux, requires aarch64-linux-gnu-gcc)
FW_DIR    := firmware
FW_SRC    := $(FW_DIR)/rogueguard_firmware.c
FW_BIN    := $(FW_DIR)/rogueguard_firmware.elf
FW_CC     := aarch64-linux-gnu-gcc
FW_CFLAGS := -O2 -std=c11 -Wall -Wextra
FW_LIBS   := -lm -lfftw3f -lpthread

.PHONY: install run seals execute clean push help firmware

help:
	@echo "Dispersion-Assisted Optical Phase Recovery"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install Python dependencies"
	@echo "  run          Open phase_retrieval.ipynb in Jupyter"
	@echo "  seals        Open SEALS simulation notebook"
	@echo "  execute      Run notebook top-to-bottom and save outputs"
	@echo "  firmware     Cross-compile rogueguard firmware -> aarch64 ELF"
	@echo "  clean        Remove cache files and generated outputs"
	@echo "  push         Push main branch to GitHub"

install:
	$(PIP) install -r requirements.txt

run:
	jupyter notebook $(NOTEBOOK)

seals:
	jupyter notebook $(SEALS_NB)

execute:
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)

firmware: $(FW_SRC) $(FW_DIR)/rogueguard_firmware.h
	$(FW_CC) $(FW_CFLAGS) -o $(FW_BIN) $(FW_SRC) $(FW_LIBS)
	@echo "Built: $(FW_BIN)"

clean:
	find . -type d -name "__pycache__"        -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	-rm -f $(FW_BIN)

push:
	git push origin main
