PYTHON := C:/Users/mrjel/AppData/Local/Programs/Python/Python312/python.exe
NOTEBOOK := phase_retrieval.ipynb

.PHONY: install test notebook clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) gs_core.py

notebook:
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --output $(NOTEBOOK)

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
