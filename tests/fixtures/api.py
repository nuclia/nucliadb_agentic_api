import asyncio
import socket

import pytest
import uvicorn
from httpx import ASGITransport, AsyncClient
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogFormatType, LogLevel, LogSettings
from nucliadb_utils.settings import AuditSettings

from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.settings import Settings
from nucliadb_sdk.tests.fixtures import NucliaFixture
from nucliadb_agentic_api.ask.settings import settings as ask_settings


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
async def nucliadb_agentic_settings(valkey_url: str):
    yield Settings(
        running_environment="test",
        valkey_url=valkey_url,
        valkey_cluster_mode=False,
    )


@pytest.fixture
async def nucliadb_agentic_audit_settings(nats_server: str):
    yield AuditSettings(
        audit_jetstream_auth=None,
        audit_jetstream_servers=[nats_server],
        audit_hash_seed=1234,
    )


@pytest.fixture
async def nucliadb_agentic_api_app(
    nucliadb_agentic_settings: Settings,
    nucliadb_agentic_data_manager_settings: DataManagerSettings,
    nucliadb_agentic_audit_settings: AuditSettings,
    nucliadb: NucliaFixture,
):

    # Configure ask to connect to a real NucliaDB
    ask_settings.nucliadb_reader_address = nucliadb.url
    ask_settings.nucliadb_search_address = nucliadb.url

    setup_logging(
        settings=LogSettings(
            log_format_type=LogFormatType.PLAIN,
            debug=True,
            log_level=LogLevel(LogLevel.DEBUG),
            logger_levels={
                "uvicorn.error": LogLevel.ERROR,
                "nucliadb_telemetry": LogLevel.ERROR,
                "mcp.client.streamable_http": LogLevel.WARNING,
                "mcp.server.lowlevel.server": LogLevel.WARNING,
                "hyperforge.configure": LogLevel.WARNING,
            },
        )
    )

    application = HTTPApplication(
        settings=nucliadb_agentic_settings,
        data_manager_settings=nucliadb_agentic_data_manager_settings,
        audit_settings=nucliadb_agentic_audit_settings,
    )

    await application.startup()

    yield application

    await application.shutdown()


@pytest.fixture
async def nucliadb_agentic_api(nucliadb_agentic_api_app: HTTPApplication, load_agents):
    yield AsyncClient(
        transport=ASGITransport(app=nucliadb_agentic_api_app), base_url="http://test"
    )


@pytest.fixture
async def nucliadb_agentic_api_http(
    nucliadb_agentic_api_app: HTTPApplication,
):
    """Serve the already-started arag_api_app over real HTTP/WebSocket.

    Reuses the same HTTPApplication instance as arag_api_app so that only one
    PredictEngine (and one aiohttp session) is created per test. Uvicorn is
    started with lifespan="off" to avoid calling startup/shutdown a second time.
    """
    http_port = free_port()
    config = uvicorn.Config(
        nucliadb_agentic_api_app,
        host="127.0.0.1",
        port=http_port,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    while not server.started and not server.should_exit:
        await asyncio.sleep(0.01)
    if not server.started:
        await server_task
        raise RuntimeError("arag_api_http failed to start")

    yield f"127.0.0.1:{http_port}"

    server.should_exit = True
    await server_task


@pytest.fixture
async def nucliadb_agentic_api_http_client(
    nucliadb_agentic_api_http: str,
):
    """
    Fixture to provide an HTTP client for the NucliaDB Agentic API.
    """
    async with AsyncClient(
        base_url=f"http://{nucliadb_agentic_api_http}",
        headers={"X-NUCLIADB-ROLES": "MANAGER;READER;WRITER"},
    ) as client:
        yield client
