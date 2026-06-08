import pathlib

import pytest

from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.db.settings import DataManagerSettings
import alembic.command
import alembic.config
from sqlalchemy_utils import (  # type: ignore
    create_database,
    database_exists,
    drop_database,
)

_package_path = pathlib.Path(__file__).parent.parent.parent.absolute()


@pytest.fixture(scope="session")
def agentic_pg_dsn(pg_dsn):
    new_pg_dsn = pg_dsn.replace("/test_db", "/agentic_test_db")

    if database_exists(new_pg_dsn):
        drop_database(new_pg_dsn)
    create_database(new_pg_dsn)
    config = alembic.config.Config(str(_package_path) + "/alembic.ini")
    config.set_main_option("sqlalchemy.url", new_pg_dsn)
    alembic.command.upgrade(config, "head")
    yield new_pg_dsn


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
