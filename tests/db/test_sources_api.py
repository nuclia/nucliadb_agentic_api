"""Integration tests for source persistence and the sources API.

Persistence tests require PostgreSQL (managed by agentic_pg_dsn). HTTP API
tests additionally require NucliaDB and use uvicorn to exercise the full
middleware stack.
"""

from uuid import uuid4

import pytest
import sqlalchemy as sa
from httpx import AsyncClient

from nucliadb_agentic_api.db.sources import (
    Sources,
    _serialize_source,
    _source_from_row,
    sources_table,
)
from nucliadb_agentic_api.models import MCPSourceSchema


@pytest.mark.parametrize(
    "current_secret,include_secret",
    [("rotated-secret", True), (None, True), (None, False)],
)
async def test_mcp_patch_preserves_database_secret_with_stale_cache(
    nucliadb_agentic_data_manager_settings,
    endecryptor_settings,
    current_secret,
    include_secret,
):
    manager = await Sources.from_settings(nucliadb_agentic_data_manager_settings)
    await manager.initialize()
    source_id = uuid4().hex
    condition = (
        (sources_table.c.account == "test-account")
        & (sources_table.c.kbid == "test-kb")
        & (sources_table.c.source_id == source_id)
    )
    try:
        await manager.create_source(
            "test-account",
            "test-kb",
            source_id,
            MCPSourceSchema(uri="https://example.com/mcp", client_secret="old-secret"),
        )
        current = _serialize_source(
            MCPSourceSchema(uri="https://example.com/mcp", client_secret=current_secret)
        )
        if not include_secret:
            del current["client_secret"]
        await manager.database.execute(
            sa.update(sources_table).where(condition).values(config=current)
        )

        await manager.patch_source(
            "test-account",
            "test-kb",
            source_id,
            MCPSourceSchema(uri="https://example.com/mcp", scope="updated-scope"),
        )

        row = await manager.database.fetch_one(
            sa.select(sources_table.c.config).where(condition)
        )
        assert row is not None
        assert row["config"]["client_secret"] == current.get("client_secret")
        assert _source_from_row(row).client_secret == current_secret
        assert row["config"]["scope"] == "updated-scope"
        cached = await manager.get_source("test-account", "test-kb", source_id)
        assert cached.client_secret == current_secret
        assert cached.scope == "updated-scope"
    finally:
        await manager.database.execute(sa.delete(sources_table).where(condition))
        await manager.finalize()


async def test_sources_api(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    kbid = knowledgebox
    payload = {
        "type": "nucliadb",
        "description": "My KB source",
        "search_config": "default-rag",
        "labels": ["important", "verified"],
    }

    # Create
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{kbid}/sources/src-nucliadb", json=payload
    )
    assert resp.status_code == 201, resp.text

    # Get
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{kbid}/sources/src-nucliadb"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == payload

    # List
    resp = await nucliadb_agentic_api_http_client.get(f"/api/v1/kb/{kbid}/sources")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"src-nucliadb": payload}

    # Patch
    updated = {
        "type": "nucliadb",
        "description": "Updated KB source",
        "search_config": "focused-rag",
        "labels": ["updated"],
    }
    resp = await nucliadb_agentic_api_http_client.patch(
        f"/api/v1/kb/{kbid}/sources/src-nucliadb", json=updated
    )
    assert resp.status_code == 204, resp.text

    # Get after patch
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{kbid}/sources/src-nucliadb"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == updated

    # Delete
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{kbid}/sources/src-nucliadb"
    )
    assert resp.status_code == 204, resp.text

    # Get after delete → 404
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{kbid}/sources/src-nucliadb"
    )
    assert resp.status_code == 404, resp.text


async def test_sources_api_all_types(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    """One source of each type can be created and retrieved correctly."""
    kbid = knowledgebox
    sources = {
        "src-nucliadb": {
            "type": "nucliadb",
            "description": "NucliaDB source",
            "labels": ["foo"],
        },
        "src-perplexity": {
            "type": "perplexity",
            "description": "Perplexity source",
            "enabled_domains": ["wikipedia.org"],
        },
        "src-mcp": {
            "type": "mcp",
            "description": "MCP source",
            "uri": "http://mcp.internal:8080",
            "tool_choice_model": {
                "_type": "llm_config",
                "model_id": "chatgpt-4.1",
            },
        },
        "src-google": {
            "type": "google",
            "description": "Google source",
            "time_range": "past_month",
            "exclude_domains": ["spam.com"],
        },
    }

    for source_id, payload in sources.items():
        resp = await nucliadb_agentic_api_http_client.post(
            f"/api/v1/kb/{kbid}/sources/{source_id}", json=payload
        )
        assert resp.status_code == 201, f"{source_id}: {resp.text}"

        resp = await nucliadb_agentic_api_http_client.get(
            f"/api/v1/kb/{kbid}/sources/{source_id}"
        )
        assert resp.status_code == 200, f"{source_id}: {resp.text}"
        assert resp.json() == payload, source_id

    # List returns all four
    resp = await nucliadb_agentic_api_http_client.get(f"/api/v1/kb/{kbid}/sources")
    assert resp.status_code == 200, resp.text
    listed = resp.json()
    assert set(listed.keys()) == set(sources.keys())
    for source_id, payload in sources.items():
        assert listed[source_id] == payload, source_id


async def test_sources_api_conflict(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    kbid = knowledgebox
    payload = {
        "type": "perplexity",
        "description": "Dupe",
        "enabled_domains": ["example.com"],
    }

    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{kbid}/sources/dupe-src", json=payload
    )
    assert resp.status_code == 201, resp.text

    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{kbid}/sources/dupe-src", json=payload
    )
    assert resp.status_code == 409


async def test_sources_api_not_found(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    kbid = knowledgebox

    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{kbid}/sources/does-not-exist"
    )
    assert resp.status_code == 404

    resp = await nucliadb_agentic_api_http_client.patch(
        f"/api/v1/kb/{kbid}/sources/does-not-exist",
        json={"type": "nucliadb", "description": "x"},
    )
    assert resp.status_code == 404

    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{kbid}/sources/does-not-exist"
    )
    assert resp.status_code == 404


async def test_delete_source_rejects_referenced_agentic_configs(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    """Deleting a referenced source reports every blocking agentic config."""
    kbid = knowledgebox

    # Create a source
    source_payload = {
        "type": "nucliadb",
        "description": "Cascade source",
        "labels": ["cascade"],
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{kbid}/sources/cascade-src", json=source_payload
    )
    assert resp.status_code == 201, resp.text

    # Create two agentic configs that reference the source
    for name in ("agent-cascade-a", "agent-cascade-b"):
        cfg_payload = {
            "title": name,
            "smart_agent": {
                "sources": ["cascade-src"],
            },
        }
        resp = await nucliadb_agentic_api_http_client.post(
            f"/api/v1/kb/{kbid}/agentic_configs/{name}", json=cfg_payload
        )
        assert resp.status_code == 201, f"{name}: {resp.text}"

    # Create an unrelated agentic config.
    unrelated_payload = {
        "title": "unrelated",
        "summarize": {"conversational": True},
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{kbid}/agentic_configs/unrelated-cfg", json=unrelated_payload
    )
    assert resp.status_code == 201, resp.text

    # Deleting the source is rejected and lists the blocking configurations.
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{kbid}/sources/cascade-src"
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == (
        "Source is in use by agentic configuration(s): agent-cascade-a, agent-cascade-b"
    )

    # The source remains available after the rejected deletion.
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{kbid}/sources/cascade-src"
    )
    assert resp.status_code == 200, resp.text
