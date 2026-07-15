"""
Internal NucliaDB driver for nucliadb_agentic_api.

Registered as provider 'nucliadb_internal'. Routes requests to the correct
NucliaDB component (reader or search) via InternalNucliaDBTransport.
"""

import os
from typing import Literal

import httpx
from httpx import AsyncHTTPTransport
from hyperforge.configure import driver
from hyperforge.driver import DriverConfig
from hyperforge_nucliadb.driver import NucliaDBDriver, manager_connect
from hyperforge_nucliadb.driver_config import NucliaDBConnection
from nucliadb_sdk.v2 import NucliaDBAsync
from pydantic.config import ConfigDict

# Paths served by the search component; everything else goes to reader.
_SEARCH_PATH_SEGMENTS = (
    "/search",
    "/find",
    "/catalog",
    "/graph",
    "/summarize",
    "/retrieve",
    "/augment",
)


class InternalNucliaDBTransport(AsyncHTTPTransport):
    """Routes requests to the NucliaDB reader or search component based on path."""

    def __init__(self, reader_url: str, search_url: str):
        super().__init__()
        self._reader_url = reader_url.rstrip("/")
        self._search_url = search_url.rstrip("/")

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if any(seg in path for seg in _SEARCH_PATH_SEGMENTS):
            base = self._search_url
        else:
            base = self._reader_url

        target = httpx.URL(base)
        new_url = request.url.copy_with(
            scheme=target.scheme,
            host=target.host,
            port=target.port,
        )
        request = httpx.Request(
            method=request.method,
            url=new_url,
            headers=request.headers,
            content=request.content,
            extensions=request.extensions,
        )
        return await super().handle_async_request(request)


class InternalNucliaDBConnection(NucliaDBConnection):
    """Connection config for internal cluster access. Provider is nucliadb_internal."""
    pass


class InternalNucliaDBConfig(DriverConfig[InternalNucliaDBConnection]):
    model_config = ConfigDict(title="Internal Knowledge Box")
    provider: Literal["nucliadb_internal"]
    config: InternalNucliaDBConnection


@driver(
    id="nucliadb_internal",
    title="Internal KnowledgeBox Source",
    description="Internal cluster source for KnowledgeBox. Routes to reader/search components.",
    config_schema=InternalNucliaDBConfig,
)
class InternalNucliaDBDriver(NucliaDBDriver):
    @classmethod
    async def init(cls, driver: InternalNucliaDBConfig):
        reader_url = os.environ.get("NUCLIADB_READER_INTERNAL_URL")
        search_url = os.environ.get("NUCLIADB_SEARCH_INTERNAL_URL")
        if not reader_url or not search_url:
            raise RuntimeError(
                "NUCLIADB_READER_INTERNAL_URL and NUCLIADB_SEARCH_INTERNAL_URL must be set "
                "to use the nucliadb_internal driver"
            )
        headers = {"X-NUCLIADB-ROLES": "READER"}
        ndb = NucliaDBAsync(
            url=reader_url,
            headers=headers,
            _httpx_transport=InternalNucliaDBTransport(reader_url, search_url),
        )
        return cls(
            provider=driver.provider,
            name=driver.name,
            config=driver.config,
            driver=ndb,
            manager=await manager_connect(driver.config),
            _synonyms=None,
        )
