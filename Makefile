.PHONY: all run analyze validate benchmark audit test ui docker-build docker-up clean help

help:
	@echo "Russian IT Community Data Platform — Command Shortcuts:"
	@echo "  make run        - Execute full data curation pipeline"
	@echo "  make analyze    - Run Deep Analytics Engine (800+ lines)"
	@echo "  make validate   - Validate dataset schema & zero-PII leak"
	@echo "  make benchmark  - Export and run 100-question domain benchmark"
	@echo "  make audit      - Run Red-Team adversarial PII penetration audit"
	@echo "  make test       - Run unit test suite (unittest)"
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
	python -m unittest discover -s tests

ui:
	streamlit run app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up data-studio

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
