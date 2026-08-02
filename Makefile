.PHONY: install format lint typecheck test validate quality

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
