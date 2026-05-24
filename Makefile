.PHONY: setup download ml-pipeline rag-pipeline monitoring test serve docker-run clean

VENV = .venv
PYTHON = $(VENV)\Scripts\python.exe
PIP = $(VENV)\Scripts\pip.exe
UVicorn = $(VENV)\Scripts\uvicorn.exe
PYTEST = $(VENV)\Scripts\pytest.exe

setup:
	python -m venv $(VENV)
	$(PIP) install -e .

download:
	$(PYTHON) src\data\download.py

ml-pipeline:
	$(PYTHON) pipelines\run_ml_pipeline.py

rag-pipeline:
	$(PYTHON) pipelines\run_rag_pipeline.py

monitoring:
	$(PYTHON) pipelines\run_monitoring.py

test:
	$(PYTEST) tests\ -v

serve:
	$(UVicorn) app.main:app --host 0.0.0.0 --port 8000 --reload

docker-run:
	docker compose up --build

clean:
	if exist mlruns rmdir /s /q mlruns
	if exist data\raw rmdir /s /q data\raw
	if exist data\processed rmdir /s /q data\processed
	if exist monitoring\reports rmdir /s /q monitoring\reports
