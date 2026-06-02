PY      := D:/WinterInterSession2025to2026/jupyter/phase-retrieval-gs/.venv/Scripts/python.exe
NB      := phase_retrieval.ipynb
NB_OUT  := phase_retrieval_executed.ipynb

.PHONY: help install test torch notebook patch clean distclean

help:
	@echo "Targets:"
	@echo "  install    pip install requirements into venv"
	@echo "  test       gs_core self-test  (numpy, ~1 s)"
	@echo "  torch      gs_torch self-test (PyTorch, single + batched)"
	@echo "  notebook   execute $(NB) -> $(NB_OUT)"
	@echo "  patch      re-apply gs_core cells to notebook"
	@echo "  clean      remove executed notebook + temp scripts + *.pyc"
	@echo "  distclean  clean + figures/*.png"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) gs_core.py

torch:
	$(PY) gs_torch.py

notebook:
	jupyter nbconvert --to notebook --execute --allow-errors \
		--ExecutePreprocessor.timeout=300 \
		--output $(NB_OUT) $(NB)

patch:
	$(PY) _patch_cells.py

data-small:
	$(PY) gen_training_data.py --n 1000 --preview

data-full:
	$(PY) gen_training_data.py --n 50000

overnight:
	$(PY) overnight_train.py --hours 7

clean:
	-del /f /q $(NB_OUT) gs_core_test.png 2>nul
	-del /f /q _patch_cells.py _add_dimanalysis.py _cleanup_and_insert.py _check_cells.py 2>nul
	-for /d /r . %%d in (__pycache__) do @rd /s /q "%%d" 2>nul
	-del /s /q *.pyc 2>nul

distclean: clean
	-del /f /q figures\*.png 2>nul
