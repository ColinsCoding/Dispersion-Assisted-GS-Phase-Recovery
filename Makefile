PY      := D:/WinterInterSession2025to2026/jupyter/phase-retrieval-gs/.venv/Scripts/python.exe
NB      := phase_retrieval.ipynb
NB_OUT  := phase_retrieval_executed.ipynb
NB_DEMO     := notebooks/dispersion_gs_demo.ipynb
NB_DEMO_OUT := notebooks/dispersion_gs_demo_executed.ipynb
NB_MODPHY   := notebooks/modern_physics_ee.ipynb
NB_MODPHY_OUT := notebooks/modern_physics_ee_executed.ipynb
NB_PHOTON   := notebooks/integrated_photonics_intro.ipynb
NB_PHOTON_OUT := notebooks/integrated_photonics_intro_executed.ipynb
NB_EMPIPE   := notebooks/em_pipe_formalization.ipynb
NB_EMPIPE_OUT := notebooks/em_pipe_formalization_executed.ipynb

.PHONY: help install test torch fno notebook notebook-demo notebook-modphy notebook-photon notebook-empipe notebooks-all patch clean distclean

help:
	@echo "Targets:"
	@echo "  install       pip install requirements into venv"
	@echo "  test          gs_core self-test  (numpy, ~1 s)"
	@echo "  torch         gs_torch self-test (PyTorch, single + batched)"
	@echo "  notebook       execute $(NB) -> $(NB_OUT)"
	@echo "  notebook-demo  execute $(NB_DEMO) -> $(NB_DEMO_OUT)"
	@echo "  notebook-modphy execute $(NB_MODPHY) -> $(NB_MODPHY_OUT)"
	@echo "  notebook-photon execute $(NB_PHOTON) -> $(NB_PHOTON_OUT)"
	@echo "  notebook-empipe execute $(NB_EMPIPE) -> $(NB_EMPIPE_OUT)"
	@echo "  notebooks-all  execute all notebooks sequentially"
	@echo "  fno            gs_fno self-test (PyTorch, ~2 min)"
	@echo "  patch         re-apply gs_core cells to notebook"
	@echo "  clean         remove executed notebooks + temp scripts + *.pyc"
	@echo "  distclean     clean + figures/*.png + demo PNGs"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) gs_core.py

verify:
	$(PY) gs_verify.py

backtest:
	$(PY) gs_backtest.py

torch:
	$(PY) gs_torch.py

fno:
	$(PY) gs_fno.py

notebook:
	jupyter nbconvert --to notebook --execute --allow-errors \
		--ExecutePreprocessor.timeout=300 \
		--output $(NB_OUT) $(NB)

notebook-demo:
	jupyter nbconvert --to notebook --execute --allow-errors \
		--ExecutePreprocessor.timeout=600 \
		--output-dir notebooks \
		--output dispersion_gs_demo_executed.ipynb \
		$(NB_DEMO)

notebook-modphy:
	jupyter nbconvert --to notebook --execute --allow-errors \
		--ExecutePreprocessor.timeout=600 \
		--output-dir notebooks \
		--output modern_physics_ee_executed.ipynb \
		$(NB_MODPHY)

notebook-photon:
	jupyter nbconvert --to notebook --execute --allow-errors \
		--ExecutePreprocessor.timeout=600 \
		--output-dir notebooks \
		--output integrated_photonics_intro_executed.ipynb \
		$(NB_PHOTON)

notebook-empipe:
	jupyter nbconvert --to notebook --execute --allow-errors \
		--ExecutePreprocessor.timeout=600 \
		--output-dir notebooks \
		--output em_pipe_formalization_executed.ipynb \
		$(NB_EMPIPE)

notebooks-all: notebook notebook-demo notebook-modphy notebook-photon notebook-empipe

patch:
	$(PY) _patch_cells.py

data-small:
	$(PY) gen_training_data.py --n 1000 --preview

data-full:
	$(PY) gen_training_data.py --n 50000

overnight:
	$(PY) overnight_train.py --hours 7

clean:
	-del /f /q $(NB_OUT) $(NB_DEMO_OUT) $(NB_MODPHY_OUT) $(NB_PHOTON_OUT) gs_core_test.png 2>nul
	-del /f /q _patch_cells.py _add_dimanalysis.py _cleanup_and_insert.py _check_cells.py 2>nul
	-for /d /r . %%d in (__pycache__) do @rd /s /q "%%d" 2>nul
	-del /s /q *.pyc 2>nul

distclean: clean
	-del /f /q figures\*.png 2>nul
	-del /f /q notebooks\demo_*.png 2>nul
