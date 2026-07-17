import pathlib

import alembic.command
import alembic.config
import pytest
from sqlalchemy_utils import (
    create_database,
    database_exists,
    drop_database,
)

from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.db.settings import DataManagerSettings

_package_path = pathlib.Path(__file__).parent.parent.parent.absolute()


@pytest.fixture(scope="session")
def agentic_pg_dsn(pg):
    host, port = pg
    dsn = f"postgresql://postgres:postgres@{host}:{port}/agentic_test_db"
    if database_exists(dsn):
        drop_database(dsn)
    create_database(dsn)
    config = alembic.config.Config(str(_package_path) + "/alembic.ini")
    config.set_main_option("sqlalchemy.url", dsn)
    alembic.command.upgrade(config, "head")
    yield dsn


@pytest.fixture
async def nucliadb_agentic_data_manager_settings(agentic_pg_dsn):
    yield DataManagerSettings(postgresql_dsn=agentic_pg_dsn)


@pytest.fixture
async def agentic_configs_db_server(
    nucliadb_agentic_data_manager_settings: DataManagerSettings,
):
    agent_manager = await AgenticConfigs.from_settings(
        settings=nucliadb_agentic_data_manager_settings
    )
    await agent_manager.initialize()
    yield agent_manager
    await agent_manager.finalize()
