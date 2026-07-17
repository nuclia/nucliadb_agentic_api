"""
Internal NucliaDB driver for nucliadb_agentic_api.

Registered as provider 'nucliadb_internal'. Creates two NucliaDBAsync clients
(reader + search) and exposes a proxy that delegates to the correct one.
"""

import os
from typing import Literal

from hyperforge.configure import driver
from hyperforge.driver import DriverConfig
from hyperforge_nucliadb.driver import NucliaDBDriver, manager_connect
from hyperforge_nucliadb.driver_config import NucliaDBConnection
from nucliadb_sdk.v2 import NucliaDBAsync
from pydantic.config import ConfigDict

# Methods served by the search component; everything else goes to reader.
_SEARCH_METHODS = {
    "ask",
    "find",
    "search",
    "catalog",
    "catalog_facets",
    "graph_nodes",
    "graph_relations",
    "graph_search",
    "summarize",
    "retrieve",
    "augment",
}


class NucliaDBProxy:
    """Holds two NucliaDBAsync clients and dispatches method calls to the correct one."""

    def __init__(self, reader: NucliaDBAsync, search: NucliaDBAsync):
        self._reader = reader
        self._search = search

    def __getattr__(self, name: str):
        if name in _SEARCH_METHODS:
            return getattr(self._search, name)
        return getattr(self._reader, name)


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
    driver: NucliaDBProxy

    @classmethod
    async def init(cls, driver: InternalNucliaDBConfig):
        base_url = os.environ.get("INTERNAL_NUCLIADB_URL")
        if not base_url:
            raise RuntimeError(
                "INTERNAL_NUCLIADB_URL must be set to use the nucliadb_internal driver"
            )
        reader_url = base_url.format(component="reader")
        search_url = base_url.format(component="search")
        headers = {"X-NUCLIADB-ROLES": "READER"}
        reader_ndb = NucliaDBAsync(url=reader_url, headers=headers)
        search_ndb = NucliaDBAsync(url=search_url, headers=headers)
        return cls(
            provider=driver.provider,
            name=driver.name,
            config=driver.config,
            driver=NucliaDBProxy(reader_ndb, search_ndb),
            manager=await manager_connect(driver.config),
            _synonyms=None,
        )
