.PHONY: all install run analyze validate benchmark audit audit-prob drift orchestrate dvc-repro test coverage lint format typecheck typecheck-strict ui docker-build docker-up clean help

help:
	@echo "Russian IT Community Data Platform — Command Shortcuts:"
	@echo "  make install    - Install package in editable mode with dev tools"
	@echo "  make run        - Execute full data curation pipeline"
	@echo "  make analyze    - Run Statistical & Semantic Analytics Engine"
	@echo "  make validate   - Validate dataset schema & zero-PII leak"
	@echo "  make benchmark  - Export and run 100-question domain benchmark"
	@echo "  make audit      - Run Red-Team adversarial PII penetration audit"
	@echo "  make audit-prob - Run stratified probabilistic PII audit (Wilson CI, 99% bounds)"
	@echo "  make drift      - Run dataset drift monitoring (PSI / JS / vocabulary)"
	@echo "  make orchestrate- Run Prefect orchestration flow (graceful fallback)"
	@echo "  make dvc-repro  - Reproduce the DVC pipeline (dvc repro)"
	@echo "  make test       - Run unit test suite (pytest + coverage gate)"
	@echo "  make coverage   - Run tests with detailed coverage report"
	@echo "  make lint       - Run ruff linter (same pinned version as CI)"
	@echo "  make format     - Auto-format code with ruff format"
	@echo "  make typecheck  - Run mypy over src/ (slow-mode rollout)"
	@echo "  make typecheck-strict - Run mypy --strict on fully-typed modules"
	@echo "  make reports    - Regenerate markdown reports from JSON sources"
	@echo "  make ui         - Launch Streamlit Web Data Studio"
	@echo "  make docker-up  - Launch via Docker Compose"

install:
	python -m pip install -e ".[dev]"

run:
	python main.py

analyze:
	python cli.py analyze

validate:
	python cli.py validate

benchmark:
	python cli.py benchmark

audit:
	python -c "from pathlib import Path; from src.validation.pii_redteam import RedTeamPIIAuditor; auditor = RedTeamPIIAuditor(Path('dataset_output/parquet/full_clean_messages.parquet')); auditor.generate_audit_certificate(Path('reports/pii_validation_report.json'))"

audit-prob:
	python scripts/run_probabilistic_pii_audit.py

drift:
	python scripts/run_drift_monitoring.py

orchestrate:
	python -c "from src.orchestration.prefect_flow import run_flow; import json; print(json.dumps(run_flow(run_pipeline=False), indent=2, ensure_ascii=False, default=str))"

dvc-repro:
	dvc repro

test:
	python -m pytest -q

coverage:
	python -m pytest -q --cov=src --cov-report=term-missing

# Ruff version is pinned in pyproject.toml dev extras == the pre-commit rev,
# so `make lint` reproduces CI exactly.
lint:
	ruff check .

format:
	ruff format .

typecheck:
	python -m mypy src/ --ignore-missing-imports

# Strict rollout: modules are added here as they pass `mypy --strict`.
# Mirrors the [[tool.mypy.overrides]] ignore_errors list in pyproject.toml:
# heavy GPU/NLP modules (engine, pipeline, inference, lora, evaluation,
# graph, finalize_sync_all) stay ignored there and are excluded from this path.
typecheck-strict:
	python -m mypy src/config.py src/bootstrap.py src/ingestion/schema.py src/ingestion/loader.py src/analytics/metrics.py src/analytics/network.py src/analytics/report_generator.py src/taxonomy/classifier.py src/taxonomy/tagger.py src/deduplication/exact_dedup.py src/deduplication/minhash_lsh.py src/monitoring/drift.py --strict

# Markdown model zoo catalog is generated from local & hub models.
reports:
	python scripts/regenerate_model_catalog.py

ui:
	streamlit run app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up data-studio

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
