.PHONY: all run analyze validate benchmark audit test coverage typecheck typecheck-strict ui docker-build docker-up clean help

help:
	@echo "Russian IT Community Data Platform — Command Shortcuts:"
	@echo "  make run        - Execute full data curation pipeline"
	@echo "  make analyze    - Run Statistical & Semantic Analytics Engine"
	@echo "  make validate   - Validate dataset schema & zero-PII leak"
	@echo "  make benchmark  - Export and run 100-question domain benchmark"
	@echo "  make audit      - Run Red-Team adversarial PII penetration audit"
	@echo "  make test       - Run unit test suite (pytest + coverage gate)"
	@echo "  make coverage   - Run tests with detailed coverage report"
	@echo "  make typecheck  - Run mypy over src/ (slow-mode rollout)"
	@echo "  make typecheck-strict - Run mypy --strict on fully-typed modules"
	@echo "  make reports    - Regenerate markdown reports from JSON sources"
	@echo "  make ui         - Launch Streamlit Web Data Studio"
	@echo "  make docker-up  - Launch via Docker Compose"

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

test:
	python -m pytest -q

coverage:
	python -m pytest -q --cov=src --cov-report=term-missing

typecheck:
	python -m mypy src/ --ignore-missing-imports

# Strict rollout: add modules here as they get fully typed.
typecheck-strict:
	python -m mypy src/ingestion/schema.py src/bootstrap.py --strict

# Markdown reports are derived artifacts: regenerate from JSON, never edit by hand.
reports:
	python scripts/regenerate_analytics_report.py

ui:
	streamlit run app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up data-studio

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
