install:
	uv sync

install-test:
	uv sync --group dev

fmt:
	uv run ruff format src tests
	uv run ruff check src tests --select I --fix 

extract-openai:
	uv run arag-extract-openapi  $(DOCS_FILE) $(API_VERSION) $(HASH)

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src tests

start_local_db:
	brew services start postgresql

stop_local_db:
	brew services stop postgresql

create_db:
	POSTGRESQL_DSN=postgresql:///postgres alembic upgrade head

reset_db:
	psql -d postgres -c "DELETE FROM alembic_version;" || true
	POSTGRESQL_DSN=postgresql:///postgres alembic stamp head

reset_db_hard:
	psql -d postgres -c "DROP TABLE IF EXISTS alembic_version CASCADE;" || true
	psql -d postgres -c "DROP TABLE IF EXISTS download_requests CASCADE;" || true
	POSTGRESQL_DSN=postgresql:///postgres alembic upgrade head

check_db_version:
	psql -d postgres -c "SELECT * FROM alembic_version;" || echo "No alembic_version table found"

alembic_history:
	POSTGRESQL_DSN=postgresql:///postgres alembic history

generate_alembic_version:
	POSTGRESQL_DSN=postgresql:///postgres alembic revision --autogenerate


dockers:
	docker build -t nucliadb_agentic . -f NUCLIADB_AGENTIC.Dockerfile

pytest_flags := -s -rfE -v --tb=native 
pytest_extra_flags :=
pytest_record_flags := --record-mode=rewrite
pytest_play_record_flags := --record-mode=none
pytest_cov_report_flags := --cov-report xml --cov-report term-missing:skip-covered

PYTEST := pytest $(pytest_flags) $(pytest_extra_flags)


.PHONY: test
test:
	uv run $(PYTEST) $(pytest_play_record_flags) tests/ $(ARGS)

record:
	uv run $(PYTEST) $(pytest_record_flags) tests/ $(ARGS)