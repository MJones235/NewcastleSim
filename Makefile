.PHONY: help install format lint type-check test check all clean

help:
	@echo "Available targets:"
	@echo "  install       Install development dependencies"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run ruff linter"
	@echo "  type-check    Run mypy type checker"
	@echo "  test          Run pytest test suite"
	@echo "  check         Run all checks (format, lint, type-check, test)"
	@echo "  all           Install deps and run all checks"
	@echo "  clean         Remove cache and temporary files"

install:
	@echo "Installing development dependencies..."
	.venv/bin/pip install black isort ruff mypy pytest pre-commit
	.venv/bin/pre-commit install

format:
	@echo "Formatting code with black..."
	.venv/bin/black scenarios/ tests/ run_jupedsim_station.py
	@echo "Sorting imports with isort..."
	.venv/bin/isort scenarios/ tests/ run_jupedsim_station.py

lint:
	@echo "Running ruff linter..."
	.venv/bin/ruff check scenarios/ tests/ run_jupedsim_station.py

type-check:
	@echo "Running mypy type checker..."
	.venv/bin/mypy scenarios/ run_jupedsim_station.py

test:
	@echo "Running pytest test suite..."
	.venv/bin/pytest tests/

check: format lint type-check test
	@echo "✓ All checks passed!"

all: install check

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"
