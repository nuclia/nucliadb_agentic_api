from collections.abc import AsyncIterator, Callable
from enum import Enum
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from hyperforge_nucliadb_agentic.ask.audit import StreamAuditStorage
from hyperforge_nucliadb_agentic.ask.search.rpc import __SDK
from hyperforge_nucliadb_agentic.ask.settings import (
    settings as ask_settings,
)
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_sdk.tests.fixtures import NucliaFixture
from nucliadb_utils.settings import AuditSettings, nuclia_settings

from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.settings import Settings

from .predict import DummyPredictEngine

# Nuclia ARAG Ask


@pytest.fixture(scope="function")
async def nucliadb_agentic_ask_api_server(
    standalone_nucliadb: NucliaFixture,
    audit: StreamAuditStorage,
    dummy_predict: DummyPredictEngine,
    nucliadb_agentic_audit_settings: AuditSettings,
    nucliadb_agentic_data_manager_settings: DataManagerSettings,
    valkey_url: str,
) -> AsyncIterator[FastAPI]:
    nucliadb_address = (
        f"http://{standalone_nucliadb.host}:{standalone_nucliadb.port}/api"
    )
    with (
        patch.object(ask_settings, "nucliadb_reader_address", nucliadb_address),
        patch.object(ask_settings, "nucliadb_search_address", nucliadb_address),
        patch.object(nuclia_settings, "nuclia_zone", "test"),
        # We must clear the global SDK instances each time we create the
        # application. Otherwise the client is not attached to the same loop we
        # run the next test and everything fails
        patch.dict(__SDK, {}, clear=True),
    ):
        app = HTTPApplication(
            settings=Settings(
                running_environment="test",
                valkey_url=valkey_url,
                valkey_cluster_mode=False,
            ),
            data_manager_settings=nucliadb_agentic_data_manager_settings,
            audit_settings=nucliadb_agentic_audit_settings,
        )
        await app.startup()
        yield app
        await app.shutdown()


@pytest.fixture(scope="function")
async def nucliadb_agentic_ask_api(
    storage_settings,
    nucliadb_agentic_ask_api_server: FastAPI,
) -> AsyncIterator[AsyncClient]:
    client_factory = create_api_client_factory(nucliadb_agentic_ask_api_server)
    async with client_factory(roles=[NucliaDBRoles.READER]) as client:
        yield client


# Utils


def create_api_client_factory(application: FastAPI) -> Callable[..., AsyncClient]:
    def _make_client_fixture(
        roles: list[Enum] | None = None,
        user: str = "",
        version: str = "1",
        root: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncClient:
        roles = roles or []
        client_base_url = "http://test"

        if root is False:
            client_base_url = f"{client_base_url}/api/v{version}"

        transport = ASGITransport(app=application)  # type: ignore
        client = AsyncClient(transport=transport, base_url=client_base_url)
        client.headers["X-NUCLIADB-ROLES"] = ";".join([role.value for role in roles])
        client.headers["X-NUCLIADB-USER"] = user

        extra_headers = extra_headers or {}
        if len(extra_headers) == 0:
            return client

        for header, value in extra_headers.items():
            client.headers[f"{header}"] = value

        return client

    return _make_client_fixture
