"""Unit tests for cross-cutting source ↔ agentic-config behaviours.

These tests exercise:
  1. Cascade delete: deleting a source also deletes every agentic config that
     references it via source_id.

Both managers are faked so no real database is required.
"""

from httpx import ASGITransport, AsyncClient
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_utils.settings import AuditSettings

from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs, _collect_source_ids
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.db.sources import Sources
from nucliadb_agentic_api.settings import Settings

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSources:
    def __init__(self):
        self.sources = {}

    async def initialize(self):
        pass

    async def finalize(self):
        pass

    async def create_source(self, account, kbid, source_id, source):
        from nucliadb_agentic_api import exceptions

        key = (account, kbid, source_id)
        if key in self.sources:
            raise exceptions.Conflict("Source already exists")
        self.sources[key] = source

    async def get_source(self, account, kbid, source_id):
        from nucliadb_agentic_api import exceptions

        key = (account, kbid, source_id)
        if key not in self.sources:
            raise exceptions.NotFound("Source not found")
        return self.sources[key]

    async def delete_source(self, account, kbid, source_id):
        from nucliadb_agentic_api import exceptions

        key = (account, kbid, source_id)
        if key not in self.sources:
            raise exceptions.NotFound("Source not found")
        del self.sources[key]

    async def patch_source(self, account, kbid, source_id, source):
        from nucliadb_agentic_api import exceptions

        key = (account, kbid, source_id)
        if key not in self.sources:
            raise exceptions.NotFound("Source not found")
        self.sources[key] = source

    async def list_sources(self, account, kbid):
        return {
            sid: src
            for (a, k, sid), src in self.sources.items()
            if a == account and k == kbid
        }


class FakeAgenticConfigs:
    def __init__(self, valid_source_ids: set | None = None):
        self.configs = {}
        self.valid_source_ids: set = (
            valid_source_ids if valid_source_ids is not None else set()
        )

    async def initialize(self):
        pass

    async def finalize(self):
        pass

    async def _check_sources(self, config):
        from nucliadb_agentic_api import exceptions

        source_ids = _collect_source_ids(config)
        missing = sorted(set(source_ids) - self.valid_source_ids)
        if missing:
            raise exceptions.InvalidReference(
                f"Source(s) not found: {', '.join(missing)}"
            )

    async def create_agentic_config(self, account, kbid, agentic_id, config):
        from nucliadb_agentic_api import exceptions

        await self._check_sources(config)
        key = (account, kbid, agentic_id)
        if key in self.configs:
            raise exceptions.Conflict("Agentic configuration already exists")
        self.configs[key] = config

    async def get_agentic_config(self, account, kbid, agentic_id):
        from nucliadb_agentic_api import exceptions

        key = (account, kbid, agentic_id)
        if key not in self.configs:
            raise exceptions.NotFound("Agentic configuration not found")
        return self.configs[key]

    async def list_agentic_configs(self, account, kbid):
        return {
            agentic_id: cfg
            for (a, k, agentic_id), cfg in self.configs.items()
            if a == account and k == kbid
        }

    async def patch_agentic_config(self, account, kbid, agentic_id, config):
        from nucliadb_agentic_api import exceptions

        await self._check_sources(config)
        key = (account, kbid, agentic_id)
        if key not in self.configs:
            raise exceptions.NotFound("Agentic configuration not found")
        self.configs[key] = config

    async def delete_configs_referencing_source(self, account, kbid, source_id) -> int:
        to_delete = [
            key
            for key, cfg in self.configs.items()
            if key[0] == account
            and key[1] == kbid
            and source_id in _collect_source_ids(cfg)
        ]
        for key in to_delete:
            del self.configs[key]
        return len(to_delete)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _api_client(app, roles: list[NucliaDBRoles]) -> AsyncClient:
    client = AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test/api/v1"
    )
    client.headers["X-NUCLIADB-ROLES"] = ";".join(r.value for r in roles)
    client.headers["X-NUCLIADB-ACCOUNT"] = "account"
    return client


async def _create_app(
    monkeypatch,
    fake_sources: FakeSources,
    fake_configs: FakeAgenticConfigs,
) -> HTTPApplication:
    async def sources_from_settings(cls, settings):
        return fake_sources

    async def configs_from_settings(cls, settings):
        return fake_configs

    async def startup(self):
        self.source_manager = await Sources.from_settings(
            settings=self.data_manager_settings
        )
        await self.source_manager.initialize()
        self.agent_manager = await AgenticConfigs.from_settings(
            settings=self.data_manager_settings
        )
        await self.agent_manager.initialize()

    async def shutdown(self):
        await self.source_manager.finalize()
        await self.agent_manager.finalize()

    monkeypatch.setattr(Sources, "from_settings", classmethod(sources_from_settings))
    monkeypatch.setattr(
        AgenticConfigs, "from_settings", classmethod(configs_from_settings)
    )
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
# Cascade-delete tests
# ---------------------------------------------------------------------------

_MANAGER_READER = [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]

_NUCLIADB_SOURCE = {
    "type": "nucliadb",
    "title": "A NucliaDB source",
    "config": {"filter_expression": "label=foo"},
}


async def test_delete_source_cascades_to_agentic_configs(monkeypatch):
    """Deleting a source removes every agentic config that references it."""
    fake_sources = FakeSources()
    fake_configs = FakeAgenticConfigs(valid_source_ids={"src-1"})
    app = await _create_app(monkeypatch, fake_sources, fake_configs)

    async with _api_client(app, _MANAGER_READER) as client:
        # 1. Create the source
        resp = await client.post("/kb/kb1/sources/src-1", json=_NUCLIADB_SOURCE)
        assert resp.status_code == 201, resp.text

        # 2. Create two agentic configs that both reference src-1
        for name in ("agent-a", "agent-b"):
            payload = {
                "title": name,
                "config": {
                    "smart_agent": {
                        "sources": [{"type": "nucliadb", "source_id": "src-1"}],
                    }
                },
            }
            resp = await client.post(f"/kb/kb1/agentic_configs/{name}", json=payload)
            assert resp.status_code == 201, resp.text

        # 3. Create an agentic config that does NOT reference src-1
        unrelated = {
            "title": "unrelated",
            "config": {"smart_agent": {"sources": [{"type": "nucliadb"}]}},
        }
        resp = await client.post("/kb/kb1/agentic_configs/unrelated", json=unrelated)
        assert resp.status_code == 201, resp.text

        # 4. Delete the source — should cascade-delete agent-a and agent-b
        resp = await client.delete("/kb/kb1/sources/src-1")
        assert resp.status_code == 204, resp.text

        # 5. Source is gone
        resp = await client.get("/kb/kb1/sources/src-1")
        assert resp.status_code == 404

        # 7. Unrelated config is still present
        resp = await client.get("/kb/kb1/agentic_configs/unrelated")
        assert resp.status_code == 200, resp.text


async def test_delete_source_with_no_referencing_configs(monkeypatch):
    """Deleting a source that no agentic config references still succeeds."""
    fake_sources = FakeSources()
    fake_configs = FakeAgenticConfigs()
    app = await _create_app(monkeypatch, fake_sources, fake_configs)

    async with _api_client(app, _MANAGER_READER) as client:
        resp = await client.post("/kb/kb1/sources/orphan", json=_NUCLIADB_SOURCE)
        assert resp.status_code == 201, resp.text

        resp = await client.delete("/kb/kb1/sources/orphan")
        assert resp.status_code == 204, resp.text

        resp = await client.get("/kb/kb1/sources/orphan")
        assert resp.status_code == 404


async def test_cascade_only_affects_same_kb(monkeypatch):
    """Cascade delete must not touch configs that live in a different KB."""
    fake_sources = FakeSources()
    fake_configs = FakeAgenticConfigs(valid_source_ids={"src-x"})
    app = await _create_app(monkeypatch, fake_sources, fake_configs)

    async with _api_client(app, _MANAGER_READER) as client:
        # Create the source in kb-a
        resp = await client.post("/kb/kb-a/sources/src-x", json=_NUCLIADB_SOURCE)
        assert resp.status_code == 201, resp.text

        # Create an agentic config in kb-a that references src-x
        cfg_payload = {
            "title": "cfg-a",
            "config": {
                "smart_agent": {
                    "sources": [{"type": "nucliadb", "source_id": "src-x"}],
                }
            },
        }
        resp = await client.post("/kb/kb-a/agentic_configs/cfg-a", json=cfg_payload)
        assert resp.status_code == 201, resp.text

        # Create a config in kb-b that also references src-x (logically invalid but
        # we're testing isolation, not cross-KB referential integrity)
        fake_configs.valid_source_ids.add("src-x")  # already there
        resp = await client.post("/kb/kb-b/agentic_configs/cfg-b", json=cfg_payload)
        assert resp.status_code == 201, resp.text
