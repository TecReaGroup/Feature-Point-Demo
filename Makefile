.PHONY: install run

install:
	uv sync --managed-python --python 3.12 --no-dev

run:
	uv run --managed-python --no-sync python -m feature_point
