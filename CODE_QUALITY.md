# Code Quality Tools

This document describes the code quality tooling setup for NewcastleSim.

## Tools Configured

### 1. **Black** - Code Formatter
Automatically formats Python code to a consistent style.
- Line length: 100 characters
- Target: Python 3.10+

```bash
make format
# or
.venv/bin/black scenarios/ tests/ run_jupedsim_station.py
```

### 2. **Ruff** - Fast Linter
Modern, fast Python linter that replaces flake8, pylint, and more.
- Checks for code smells, bugs, and style issues
- Auto-fixes many issues automatically

```bash
make lint
# or
.venv/bin/ruff check scenarios/ tests/ run_jupedsim_station.py
.venv/bin/ruff check --fix scenarios/ tests/ run_jupedsim_station.py  # auto-fix
```

### 3. **Mypy** - Type Checker
Static type checker for Python type hints.
- Catches type errors before runtime
- Configured with relaxed settings initially

```bash
make type-check
# or
.venv/bin/mypy scenarios/ run_jupedsim_station.py
```

### 4. **Pre-commit** - Git Hooks
Automatically runs checks before each commit.
- Enforces code formatting
- Runs linters
- Checks for common issues

```bash
# Install hooks (one-time)
.venv/bin/pre-commit install

# Run manually
.venv/bin/pre-commit run --all-files
```

## Quick Start

### Initial Setup
```bash
make install
```

This installs all development dependencies and configures pre-commit hooks.

### Run All Checks
```bash
make check
```

Runs formatting, linting, type checking, and tests in sequence.

### Common Workflows

**Before committing:**
```bash
make format lint test
```

**Check specific file:**
```bash
.venv/bin/ruff check scenarios/station_jupedsim/core/simulation.py
.venv/bin/mypy scenarios/station_jupedsim/core/simulation.py
```

**Auto-fix all issues:**
```bash
.venv/bin/ruff check --fix scenarios/
.venv/bin/black scenarios/
```

## Configuration Files

- **`pyproject.toml`** - Main configuration for all tools (black, ruff, mypy, pytest)
- **`.pre-commit-config.yaml`** - Pre-commit hook configuration
- **`Makefile`** - Convenient shortcuts for common commands
- **`dev-requirements.txt`** - Development dependencies

## Current Status

✅ **Black**: All code formatted (26 files reformatted)
✅ **Ruff**: All checks passing (140 issues auto-fixed)
✅ **Tests**: All 48 tests passing
✅ **Python**: Using 3.12.3 (targeting 3.10+)
⚠️ **Mypy**: 26 errors remaining (mostly external library stubs and try/except imports)

## Ignoring Specific Issues

If you need to ignore a specific linter warning:

```python
# Ruff
# ruff: noqa: E501  (line too long)

# Mypy
# type: ignore[error-code]
```

## Tool Documentation

- Black: https://black.readthedocs.io/
- Ruff: https://docs.astral.sh/ruff/
- Mypy: https://mypy.readthedocs.io/
- Pre-commit: https://pre-commit.com/
