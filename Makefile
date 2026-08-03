.PHONY: install format lint typecheck test validate quality validate-sources acquire-dry-run acquisition-report

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

format:
	ruff format .

lint:
	ruff check .

typecheck:
	mypy

test:
	pytest

validate:
	python scripts/validate_environment.py

quality: format lint typecheck test

validate-sources:
	python scripts/validate_sources.py --all

acquire-dry-run:
	python scripts/acquire_sources.py --all-approved --dry-run

acquisition-report:
	python scripts/generate_acquisition_report.py \
		--manifest-dir data/manifests \
		--output-json data/reports/acquisition-report.json \
		--output-markdown data/reports/acquisition-report.md
