include nucliadb.mk

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
