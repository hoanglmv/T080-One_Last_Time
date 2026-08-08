.PHONY: run test lint format typecheck check clean docker-run docker-stop

run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

docker-run:
	docker compose up --build

docker-stop:
	docker compose down

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
