import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from hyperforge.db.encryption import decrypt_data
from nucliadb_utils.settings import AuditSettings

from nucliadb_agentic_api import exceptions
from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.db.sources import (
    Sources,
    _serialize_source,
    _source_from_row,
)
from nucliadb_agentic_api.models import MCPSourceSchema, NucliaDBSourceSchema
from nucliadb_agentic_api.settings import Settings


@pytest.fixture
def source_manager(monkeypatch, endecryptor_settings):
    monkeypatch.setattr("nucliadb_agentic_api.db.sources.CACHE", {})
    return Sources(
        database=AsyncMock(),
        settings=DataManagerSettings(postgresql_dsn="postgresql://test"),
    )


async def test_mcp_secret_is_encrypted_on_create_and_decrypted_on_load(source_manager):
    source = MCPSourceSchema(
        uri="https://example.com/mcp", client_secret="test-client-secret"
    )
    source_manager.database.fetch_one.return_value = None

    await source_manager.create_source("account", "kb", "mcp", source)

    statement = source_manager.database.execute.call_args.args[0]
    stored = statement.compile().params["config"]
    assert stored["client_secret"] != source.client_secret
    assert decrypt_data(stored["client_secret"]) == source.client_secret
    assert source.client_secret == "test-client-secret"
    assert "test-client-secret" not in repr(source)

    from nucliadb_agentic_api.db.sources import CACHE

    CACHE.clear()
    source_manager.database.fetch_one.return_value = {"config": stored}
    loaded = await source_manager.get_source("account", "kb", "mcp")
    assert loaded.client_secret == source.client_secret

    source_manager.database.fetch_all.return_value = [
        {"source_id": "mcp", "config": stored}
    ]
    listed = await source_manager.list_sources("account", "kb")
    assert listed["mcp"].client_secret == source.client_secret


async def test_mcp_http_edit_preserves_hidden_secret(source_manager):
    app = HTTPApplication(
        settings=Settings(running_environment="test"),
        data_manager_settings=source_manager.settings,
        audit_settings=AuditSettings(),
    )
    app.source_manager = source_manager
    source_manager.database.fetch_one.return_value = None
    payload = {
        "type": "mcp",
        "uri": "https://example.com/mcp",
        "client_secret": "original-secret",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={
            "X-NUCLIADB-ROLES": "OWNER;READER",
            "X-NUCLIADB-ACCOUNT": "account",
        },
    ) as client:
        response = await client.post("/api/v1/kb/kb/sources/mcp", json=payload)
        assert response.status_code == 201, response.text
        statement = source_manager.database.execute.call_args.args[0]
        stored = statement.compile().params["config"]

        from nucliadb_agentic_api.db.sources import CACHE

        CACHE.clear()
        source_manager.database.fetch_one.return_value = {"config": stored}
        response = await client.get("/api/v1/kb/kb/sources/mcp")
        assert response.status_code == 200, response.text
        editable = response.json()
        assert "client_secret" not in editable

        editable["scope"] = "updated-scope"
        source_manager.database.fetch_one.return_value = {
            "config": {**stored, "scope": "updated-scope"}
        }
        response = await client.patch("/api/v1/kb/kb/sources/mcp", json=editable)
        assert response.status_code == 204, response.text

    statement = source_manager.database.fetch_one.call_args.args[0]
    assert "jsonb_build_object" in str(statement)
    updated = await source_manager.get_source("account", "kb", "mcp")
    assert updated.scope == "updated-scope"
    assert updated.client_secret == payload["client_secret"]


@pytest.mark.parametrize(
    "update,expected_secret",
    [
        ({}, "original-secret"),
        ({"client_secret": "replacement-secret"}, "replacement-secret"),
        ({"client_secret": None}, None),
        ({"client_secret": ""}, ""),
    ],
)
async def test_mcp_patch_preserves_replaces_or_removes_secret(
    source_manager, update, expected_secret
):
    expected = MCPSourceSchema(
        uri="https://updated.example.com/mcp", client_secret=expected_secret
    )
    stored = _serialize_source(expected)
    source_manager.database.fetch_one.return_value = {"config": stored}

    await source_manager.patch_source(
        "account",
        "kb",
        "mcp",
        MCPSourceSchema.model_validate(
            {"uri": "https://updated.example.com/mcp", **update}
        ),
    )

    statement = source_manager.database.fetch_one.call_args.args[0]
    source_manager.database.fetch_one.assert_awaited_once()
    if "client_secret" not in update:
        assert "jsonb_build_object" in str(statement)
    else:
        written = statement.compile().params["config"]
        if expected_secret is None:
            assert written["client_secret"] is None
        else:
            assert written["client_secret"] != expected_secret
            assert decrypt_data(written["client_secret"]) == expected_secret
    assert _source_from_row({"config": stored}).client_secret == expected_secret
    cached = await source_manager.get_source("account", "kb", "mcp")
    assert cached.client_secret == expected_secret


@pytest.mark.parametrize("current_secret", ["rotated-secret", None])
async def test_mcp_patch_does_not_restore_stale_cached_secret(
    source_manager, current_secret
):
    source_manager.database.fetch_one.return_value = {
        "config": _serialize_source(
            MCPSourceSchema(uri="https://example.com/mcp", client_secret="old-secret")
        )
    }
    cached = await source_manager.get_source("account", "kb", "mcp")
    assert cached.client_secret == "old-secret"
    source_manager.database.fetch_one.reset_mock()
    source_manager.database.fetch_one.return_value = {
        "config": _serialize_source(
            MCPSourceSchema(
                uri="https://example.com/mcp",
                scope="updated-scope",
                client_secret=current_secret,
            )
        )
    }

    await source_manager.patch_source(
        "account",
        "kb",
        "mcp",
        MCPSourceSchema(uri="https://example.com/mcp", scope="updated-scope"),
    )

    source_manager.database.fetch_one.assert_awaited_once()
    statement = source_manager.database.fetch_one.call_args.args[0]
    assert "jsonb_build_object" in str(statement)
    assert "RETURNING sources_table.config" in str(statement)
    updated = await source_manager.get_source("account", "kb", "mcp")
    assert updated.client_secret == current_secret
    assert updated.scope == "updated-scope"


@pytest.mark.parametrize("with_secret", [True, False])
async def test_mcp_patch_missing_source_is_not_found(source_manager, with_secret):
    source_manager.database.fetch_one.return_value = None
    payload = {"uri": "https://example.com/mcp"}
    if with_secret:
        payload["client_secret"] = "test-client-secret"

    with pytest.raises(exceptions.NotFound, match="Source not found"):
        await source_manager.patch_source(
            "account", "kb", "missing", MCPSourceSchema.model_validate(payload)
        )


def test_mcp_encryption_roundtrip_is_json_serializable(endecryptor_settings):
    source = MCPSourceSchema(
        uri="https://example.com/mcp",
        client_secret="test-client-secret",
        tool_choice_model={"model_id": "tool-model", "reasoning": "enabled"},
    )
    serialized = json.loads(json.dumps(_serialize_source(source)))
    assert serialized["client_secret"] != source.client_secret
    assert _source_from_row({"config": serialized}) == source


def test_mcp_without_secret_does_not_need_encryption(monkeypatch):
    def unexpected_encryption(value):
        pytest.fail("A source without a secret must not require encryption")

    monkeypatch.setattr("hyperforge.db.encryption.encrypt_data", unexpected_encryption)
    source = MCPSourceSchema(uri="https://example.com/mcp")
    assert _source_from_row({"config": _serialize_source(source)}) == source


def test_plaintext_mcp_secret_is_readable_for_hyperforge_compatibility(
    endecryptor_settings,
):
    source = _source_from_row(
        {
            "config": {
                "type": "mcp",
                "uri": "https://example.com/mcp",
                "client_secret": "legacy-secret",
            }
        }
    )
    assert source.client_secret == "legacy-secret"
    encrypted = _serialize_source(source)
    assert encrypted["client_secret"] != "legacy-secret"
    assert decrypt_data(encrypted["client_secret"]) == "legacy-secret"


def test_other_sources_keep_existing_serialization():
    source = NucliaDBSourceSchema(description="KB source")
    assert _serialize_source(source) == source.model_dump(mode="json")
    assert _source_from_row({"config": _serialize_source(source)}) == source
