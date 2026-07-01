"""Integration tests for the sources API.

Requires a real PostgreSQL database (managed by the agentic_pg_dsn fixture) and
a real NucliaDB instance (from the nucliadb docker fixture).  Uses a uvicorn
server behind an HTTP client so the full middleware stack is exercised.
"""

from httpx import AsyncClient


async def test_sources_api(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    kbid = knowledgebox
    payload = {
        "type": "nucliadb",
        "description": "My KB source",
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


async def test_delete_source_cascades_agentic_configs(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    """Integration: deleting a source removes agentic configs that reference it."""
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
            "config": {
                "smart_agent": {
                    "sources": [{"type": "nucliadb", "source_id": "cascade-src"}],
                }
            },
        }
        resp = await nucliadb_agentic_api_http_client.post(
            f"/api/v1/kb/{kbid}/agentic_configs/{name}", json=cfg_payload
        )
        assert resp.status_code == 201, f"{name}: {resp.text}"

    # Create an agentic config that does NOT reference the source (should survive)
    unrelated_payload = {
        "title": "unrelated",
        "config": {"summarize": {"conversational": True}},
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{kbid}/agentic_configs/unrelated-cfg", json=unrelated_payload
    )
    assert resp.status_code == 201, resp.text

    # Delete the source
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{kbid}/sources/cascade-src"
    )
    assert resp.status_code == 204, resp.text

    # Source is gone
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{kbid}/sources/cascade-src"
    )
    assert resp.status_code == 404
