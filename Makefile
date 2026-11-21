.PHONY: all install run analyze validate benchmark audit audit-prob drift slo orchestrate dvc-repro test coverage lint format typecheck typecheck-strict security ui docker-build docker-up k8s-dry-run clean help reports-real help

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
	@echo "  make slo        - Run SLO release gate (SHIP/HOLD from reports/)"
	@echo "  make orchestrate- Run Prefect orchestration flow (graceful fallback)"
	@echo "  make dvc-repro  - Reproduce the DVC pipeline (dvc repro)"
	@echo "  make test       - Run unit test suite (pytest + coverage gate)"
	@echo "  make coverage   - Run tests with detailed coverage report"
	@echo "  make lint       - Run ruff linter (same pinned version as CI)"
	@echo "  make format     - Auto-format code with ruff format"
	@echo "  make typecheck  - Run mypy over ricc/ (slow-mode rollout)"
	@echo "  make typecheck-strict - Run mypy --strict on fully-typed modules"
	@echo "  make security   - Bandit HIGH gate + secret scan (same as CI security job)"
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
	python -c "from pathlib import Path; from ricc.validation.pii_redteam import RedTeamPIIAuditor; auditor = RedTeamPIIAuditor(Path('dataset_output/parquet/full_clean_messages.parquet')); auditor.generate_audit_certificate(Path('reports/pii_validation_report.json'))"

audit-prob:
	python scripts/run_probabilistic_pii_audit.py

drift:
	python scripts/run_drift_monitoring.py

slo:
	python -m ricc.monitoring.slo_gate --json-out reports/slo_verdict.json

orchestrate:
	python -c "from ricc.orchestration.prefect_flow import run_flow; import json; print(json.dumps(run_flow(run_pipeline=False), indent=2, ensure_ascii=False, default=str))"

dvc-repro:
	dvc repro

test:
	python -m pytest -q

coverage:
	python -m pytest -q --cov=ricc --cov-report=term-missing --cov-fail-under=85

# Ruff version is pinned in pyproject.toml dev extras == the pre-commit rev,
# so `make lint` reproduces CI exactly.
lint:
	ruff check .

format:
	ruff format .

typecheck:
	python -m mypy ricc/ --ignore-missing-imports

# Strict rollout COMPLETE: every ricc/ module, root entrypoint, and script passes `mypy --strict`.
typecheck-strict:
	python -m mypy ricc cli.py main.py app_helpers.py chat_ui.py scripts --strict

# Supply-chain security: bandit HIGH gate + secret scan + pip-audit (same as CI `security` job).
# pip-audit is a HARD gate: any known CVE in the locked dependency tree fails the build.
security:
	python -m bandit -r ricc/ -q --severity-level high
	python -m detect_secrets.pre_commit_hook --baseline .secrets.baseline
	python -m pip_audit -r requirements.lock.txt --desc off --progress-spinner off

# Markdown model zoo catalog is generated from local & hub models.
reports:
	python scripts/regenerate_model_catalog.py

# JSON reports regenerated from REAL local parquet (no synthetic markers).
reports-real:
	python scripts/regenerate_real_reports.py --sample 10000 --drift-sample 50000

ui:
	streamlit run app.py

docker-build:
	docker compose build

docker-up:
	docker compose up data-studio

k8s-dry-run:
	kubectl apply --dry-run=client -f deploy/k8s/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
