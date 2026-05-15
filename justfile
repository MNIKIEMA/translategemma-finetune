# List available recipes
help:
    @just --list

# Format code [ruff]"
format:
    uvx ruff format src
# Run linters [ruff]"
lint:
    uvx ruff check src --fixable all

# Run pre-commit [lint, format]"
pre-commit: lint format
    uvx prek run

# Install dependencies"
install:
    uv sync --all-groups

# Install deps on runpod
runpod-install:
	uv pip install -e . --python /usr/bin/python3 --break-system-packages

# Clean cache files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    rm -rf .pytest_cache .coverage htmlcov .ruff_cache

# Typecheck using ty
typecheck:
    uvx ty check

# Run using Pytest
test:
	uv run pytest
