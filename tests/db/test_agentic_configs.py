from httpx import ASGITransport, AsyncClient
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_utils.settings import AuditSettings

from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.settings import Settings


class FakeAgenticConfigs:
    def __init__(self, valid_source_ids: set | None = None):
        self.configs = {}
        # Set of source_ids that are considered to exist for validation purposes.
        self.valid_source_ids: set = (
            valid_source_ids if valid_source_ids is not None else set()
        )

    async def initialize(self):
        pass

    async def finalize(self):
        pass

    async def _check_sources(self, config):
        from nucliadb_agentic_api import exceptions
        from nucliadb_agentic_api.db.agentic_configs import _collect_source_ids

        source_ids = _collect_source_ids(config)
        missing = sorted(set(source_ids) - self.valid_source_ids)
        if missing:
            raise exceptions.InvalidReference(
                f"Source(s) not found: {', '.join(missing)}"
            )

    async def create_agentic_config(self, account, kbid, agentic_id, config):
        await self._check_sources(config)
        key = (account, kbid, agentic_id)
        if key in self.configs:
            from nucliadb_agentic_api import exceptions

            raise exceptions.Conflict("Agentic configuration already exists")
        self.configs[key] = config

    async def get_agentic_config(self, account, kbid, agentic_id):
        key = (account, kbid, agentic_id)
        if key not in self.configs:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Agentic configuration not found")
        return self.configs[key]

    async def list_agentic_configs(self, account, kbid):
        return {
            agentic_id: config
            for (
                stored_account,
                stored_kbid,
                agentic_id,
            ), config in self.configs.items()
            if stored_account == account and stored_kbid == kbid
        }

    async def patch_agentic_config(self, account, kbid, agentic_id, config):
        await self._check_sources(config)
        key = (account, kbid, agentic_id)
        if key not in self.configs:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Agentic configuration not found")
        self.configs[key] = config

    async def delete_configs_referencing_source(self, account, kbid, source_id) -> int:
        from nucliadb_agentic_api.db.agentic_configs import _collect_source_ids

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


def create_api_client(application, roles: list[NucliaDBRoles]) -> AsyncClient:
    client = AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test/api/v1"
    )
    client.headers["X-NUCLIADB-ROLES"] = ";".join([role.value for role in roles])
    client.headers["X-NUCLIADB-ACCOUNT"] = "account"
    return client


async def create_app(
    monkeypatch, valid_source_ids: set | None = None
) -> HTTPApplication:
    fake_configs = FakeAgenticConfigs(valid_source_ids=valid_source_ids)

    async def from_settings(cls, settings):
        return fake_configs

    async def startup(self):
        self.agent_manager = await AgenticConfigs.from_settings(
            settings=self.data_manager_settings
        )
        await self.agent_manager.initialize()

    async def shutdown(self):
        await self.agent_manager.finalize()

    monkeypatch.setattr(AgenticConfigs, "from_settings", classmethod(from_settings))
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
# Existing CRUD tests (unchanged)
# ---------------------------------------------------------------------------


async def test_agentic_config_crud(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        payload = {
            "title": "Support agent",
            "config": {
                "smart_agent": {
                    "mode": "reactive",
                    "sources": [{"type": "nucliadb", "description": "Current KB"}],
                },
                "summarize": {"conversational": True},
            },
        }

        resp = await client.post("/kb/kb/agentic_configs/support", json=payload)
        assert resp.status_code == 201, resp.text

        resp = await client.get("/kb/kb/agentic_configs/support")
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        resp = await client.get("/kb/kb/agentic_configs")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"support": payload}

        updated_payload = {
            "title": "Updated support agent",
            "config": {"summarize": {"conversational": False}},
        }
        resp = await client.patch(
            "/kb/kb/agentic_configs/support", json=updated_payload
        )
        assert resp.status_code == 204, resp.text

        resp = await client.get("/kb/kb/agentic_configs/support")
        assert resp.status_code == 200, resp.text
        assert resp.json() == updated_payload


async def test_agentic_config_not_found(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.get("/kb/kb/agentic_configs/missing")
        assert resp.status_code == 404


async def test_agentic_config_requires_manager(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.post(
            "/kb/kb/agentic_configs/support",
            json={"title": "Support agent", "config": {"summarize": {}}},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Source-reference validation tests
# ---------------------------------------------------------------------------


async def test_agentic_config_create_rejects_unknown_source_id(monkeypatch):
    """Creating a config that references a non-existent source_id returns 422."""
    app = await create_app(monkeypatch, valid_source_ids=set())
    async with create_api_client(app, [NucliaDBRoles.MANAGER]) as client:
        payload = {
            "title": "Agent",
            "config": {
                "smart_agent": {
                    "sources": [
                        {"type": "nucliadb", "source_id": "nonexistent-source"}
                    ],
                }
            },
        }
        resp = await client.post("/kb/kb/agentic_configs/agent1", json=payload)
        assert resp.status_code == 422, resp.text
        assert "nonexistent-source" in resp.json()["detail"]


async def test_agentic_config_create_accepts_known_source_id(monkeypatch):
    """Creating a config with a valid source_id succeeds."""
    app = await create_app(monkeypatch, valid_source_ids={"my-source"})
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        payload = {
            "title": "Agent",
            "config": {
                "smart_agent": {
                    "mode": "reactive",
                    "sources": [{"type": "nucliadb", "source_id": "my-source"}],
                }
            },
        }
        resp = await client.post("/kb/kb/agentic_configs/agent1", json=payload)
        assert resp.status_code == 201, resp.text

        resp = await client.get("/kb/kb/agentic_configs/agent1")
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload


async def test_agentic_config_patch_rejects_unknown_source_id(monkeypatch):
    """Patching a config to reference a non-existent source_id returns 422."""
    app = await create_app(monkeypatch, valid_source_ids={"good-source"})
    async with create_api_client(app, [NucliaDBRoles.MANAGER]) as client:
        # Create with a valid source_id
        payload = {
            "title": "Agent",
            "config": {
                "smart_agent": {
                    "mode": "reactive",
                    "sources": [{"type": "nucliadb", "source_id": "good-source"}],
                }
            },
        }
        await client.post("/kb/kb/agentic_configs/agent1", json=payload)

        # Patch to an invalid source_id
        bad_patch = {
            "title": "Agent",
            "config": {
                "smart_agent": {
                    "mode": "reactive",
                    "sources": [{"type": "nucliadb", "source_id": "bad-source"}],
                }
            },
        }
        resp = await client.patch("/kb/kb/agentic_configs/agent1", json=bad_patch)
        assert resp.status_code == 422, resp.text


async def test_agentic_config_no_source_id_skips_validation(monkeypatch):
    """Configs without source_id fields bypass validation entirely."""
    app = await create_app(monkeypatch, valid_source_ids=set())
    async with create_api_client(app, [NucliaDBRoles.MANAGER]) as client:
        payload = {
            "title": "Agent",
            "config": {
                "smart_agent": {
                    "sources": [
                        # Inline nucliadb source, no source_id
                        {"type": "nucliadb", "filter_expression": "label=foo"}
                    ],
                }
            },
        }
        resp = await client.post("/kb/kb/agentic_configs/agent1", json=payload)
        assert resp.status_code == 201, resp.text
