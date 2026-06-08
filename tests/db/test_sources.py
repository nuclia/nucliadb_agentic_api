"""Unit tests for the sources API endpoints.

Uses an in-memory FakeSources store and monkeypatching so no real database or
network connection is required.
"""

from httpx import ASGITransport, AsyncClient
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_utils.settings import AuditSettings

from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.db.sources import Sources
from nucliadb_agentic_api.settings import Settings

# ---------------------------------------------------------------------------
# In-memory fake
# ---------------------------------------------------------------------------


class FakeSources:
    def __init__(self):
        self.sources = {}

    async def initialize(self):
        pass

    async def finalize(self):
        pass

    async def create_source(self, account, kbid, source_id, source):
        key = (account, kbid, source_id)
        if key in self.sources:
            from nucliadb_agentic_api import exceptions

            raise exceptions.Conflict("Source already exists")
        self.sources[key] = source

    async def get_source(self, account, kbid, source_id):
        key = (account, kbid, source_id)
        if key not in self.sources:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Source not found")
        return self.sources[key]

    async def patch_source(self, account, kbid, source_id, source):
        key = (account, kbid, source_id)
        if key not in self.sources:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Source not found")
        self.sources[key] = source

    async def delete_source(self, account, kbid, source_id):
        key = (account, kbid, source_id)
        if key not in self.sources:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Source not found")
        del self.sources[key]

    async def list_sources(self, account, kbid):
        return {
            source_id: source
            for (
                stored_account,
                stored_kbid,
                source_id,
            ), source in self.sources.items()
            if stored_account == account and stored_kbid == kbid
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_api_client(application, roles: list[NucliaDBRoles]) -> AsyncClient:
    client = AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test/api/v1"
    )
    client.headers["X-NUCLIADB-ROLES"] = ";".join([role.value for role in roles])
    client.headers["X-NUCLIADB-ACCOUNT"] = "account"
    return client


async def create_app(monkeypatch) -> HTTPApplication:
    fake_sources = FakeSources()

    async def from_settings(cls, settings):
        return fake_sources

    class _FakeAgentManager:
        """No-op agent manager satisfying the cascade-delete contract."""

        async def delete_configs_referencing_source(
            self, account, kbid, source_id
        ) -> int:
            return 0

    async def startup(self):
        self.source_manager = await Sources.from_settings(
            settings=self.data_manager_settings
        )
        await self.source_manager.initialize()
        self.agent_manager = _FakeAgentManager()

    async def shutdown(self):
        await self.source_manager.finalize()

    monkeypatch.setattr(Sources, "from_settings", classmethod(from_settings))
    monkeypatch.setattr(HTTPApplication, "startup", startup)
    monkeypatch.setattr(HTTPApplication, "shutdown", shutdown)

    app = HTTPApplication(
        settings=Settings(running_environment="test", valkey_cluster_mode=False),
        data_manager_settings=DataManagerSettings(postgresql_dsn="postgresql://test"),
        audit_settings=AuditSettings(),
    )
    await app.startup()
    return app


# ---------------------------------------------------------------------------
# Test payloads (no None fields so round-trip assertions are straightforward)
# ---------------------------------------------------------------------------

NUCLIADB_PAYLOAD = {
    "type": "nucliadb",
    "title": "My KB source",
    "config": {
        "filter_expression": "label=important",
        "labels": ["important", "verified"],
    },
}

PERPLEXITY_PAYLOAD = {
    "type": "perplexity",
    "title": "Perplexity web",
    "config": {"enabled_domains": ["wikipedia.org", "github.com"]},
}

MCP_PAYLOAD = {
    "type": "mcp",
    "title": "Internal tools",
    "config": {"uri": "http://mcp.internal:8080"},
}

GOOGLE_PAYLOAD = {
    "type": "google",
    "title": "Google news",
    "config": {"time_range": "past_week", "exclude_domains": ["example.com"]},
}


# ---------------------------------------------------------------------------
# Full CRUD for the NucliaDB source type
# ---------------------------------------------------------------------------


async def test_sources_crud_nucliadb(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        # Create
        resp = await client.post("/kb/kb1/sources/src-nucliadb", json=NUCLIADB_PAYLOAD)
        assert resp.status_code == 201, resp.text

        # Get
        resp = await client.get("/kb/kb1/sources/src-nucliadb")
        assert resp.status_code == 200, resp.text
        assert resp.json() == NUCLIADB_PAYLOAD

        # List
        resp = await client.get("/kb/kb1/sources")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"src-nucliadb": NUCLIADB_PAYLOAD}

        # Patch
        updated = {
            "type": "nucliadb",
            "title": "Updated KB source",
            "config": {"labels": ["updated"]},
        }
        resp = await client.patch("/kb/kb1/sources/src-nucliadb", json=updated)
        assert resp.status_code == 204, resp.text

        # Get after patch
        resp = await client.get("/kb/kb1/sources/src-nucliadb")
        assert resp.status_code == 200, resp.text
        assert resp.json() == updated

        # Delete
        resp = await client.delete("/kb/kb1/sources/src-nucliadb")
        assert resp.status_code == 204, resp.text

        # Get after delete → 404
        resp = await client.get("/kb/kb1/sources/src-nucliadb")
        assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# Source type smoke tests
# ---------------------------------------------------------------------------


async def test_sources_perplexity(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        resp = await client.post(
            "/kb/kb1/sources/src-perplexity", json=PERPLEXITY_PAYLOAD
        )
        assert resp.status_code == 201, resp.text

        resp = await client.get("/kb/kb1/sources/src-perplexity")
        assert resp.status_code == 200, resp.text
        assert resp.json() == PERPLEXITY_PAYLOAD


async def test_sources_mcp(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        resp = await client.post("/kb/kb1/sources/src-mcp", json=MCP_PAYLOAD)
        assert resp.status_code == 201, resp.text

        resp = await client.get("/kb/kb1/sources/src-mcp")
        assert resp.status_code == 200, resp.text
        assert resp.json() == MCP_PAYLOAD


async def test_sources_google(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        resp = await client.post("/kb/kb1/sources/src-google", json=GOOGLE_PAYLOAD)
        assert resp.status_code == 201, resp.text

        resp = await client.get("/kb/kb1/sources/src-google")
        assert resp.status_code == 200, resp.text
        assert resp.json() == GOOGLE_PAYLOAD


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_source_not_found(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.get("/kb/kb1/sources/missing")
        assert resp.status_code == 404


async def test_source_conflict(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.MANAGER]) as client:
        resp = await client.post("/kb/kb1/sources/src-dupe", json=NUCLIADB_PAYLOAD)
        assert resp.status_code == 201, resp.text

        resp = await client.post("/kb/kb1/sources/src-dupe", json=NUCLIADB_PAYLOAD)
        assert resp.status_code == 409


async def test_source_patch_not_found(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.MANAGER]) as client:
        resp = await client.patch("/kb/kb1/sources/missing", json=NUCLIADB_PAYLOAD)
        assert resp.status_code == 404


async def test_source_delete_not_found(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.MANAGER]) as client:
        resp = await client.delete("/kb/kb1/sources/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


async def test_sources_require_manager_for_writes(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.post("/kb/kb1/sources/src", json=NUCLIADB_PAYLOAD)
        assert resp.status_code == 403

        resp = await client.patch("/kb/kb1/sources/src", json=NUCLIADB_PAYLOAD)
        assert resp.status_code == 403

        resp = await client.delete("/kb/kb1/sources/src")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


async def test_sources_list_empty(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.get("/kb/empty-kb/sources")
        assert resp.status_code == 200
        assert resp.json() == {}


async def test_sources_list_multiple(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        await client.post("/kb/kb1/sources/src-a", json=NUCLIADB_PAYLOAD)
        await client.post("/kb/kb1/sources/src-b", json=PERPLEXITY_PAYLOAD)

        resp = await client.get("/kb/kb1/sources")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert set(data.keys()) == {"src-a", "src-b"}
        assert data["src-a"] == NUCLIADB_PAYLOAD
        assert data["src-b"] == PERPLEXITY_PAYLOAD


async def test_sources_list_isolated_by_kb(monkeypatch):
    """Sources in different knowledge boxes must not bleed into each other."""
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        await client.post("/kb/kb-a/sources/src-1", json=NUCLIADB_PAYLOAD)
        await client.post("/kb/kb-b/sources/src-2", json=PERPLEXITY_PAYLOAD)

        resp = await client.get("/kb/kb-a/sources")
        assert resp.json() == {"src-1": NUCLIADB_PAYLOAD}

        resp = await client.get("/kb/kb-b/sources")
        assert resp.json() == {"src-2": PERPLEXITY_PAYLOAD}
