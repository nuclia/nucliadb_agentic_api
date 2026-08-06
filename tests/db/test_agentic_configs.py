from typing import Any, Dict

from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Existing CRUD tests (unchanged)
# ---------------------------------------------------------------------------


async def test_agentic_config_crud(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    payload: Dict[str, Any] = {
        "type": "perplexity",
        "description": "Dupe",
        "enabled_domains": ["example.com"],
    }

    # Create the source referenced by the agentic configuration.
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/sources/dupe-src", json=payload
    )
    assert resp.status_code == 201, resp.text
    payload = {
        "title": "Support agent",
        "smart_agent": {
            "mode": "reactive",
            "sources": ["dupe-src"],
            "history": True,
        },
        "summarize": {"conversational": True},
    }

    # Create an agentic configuration that uses the source.
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support", json=payload
    )
    assert resp.status_code == 201, resp.text

    # Retrieve the created configuration by ID.
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == payload

    # List active configurations in the knowledge box.
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"support": payload}

    updated_payload = {
        "title": "Updated support agent",
        "smart_agent": {
            "mode": "reactive",
            "sources": ["dupe-src"],
            "history": False,
        },
        "summarize": {"conversational": False},
    }
    # Update the configuration and verify the new payload persists.
    resp = await nucliadb_agentic_api_http_client.patch(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support", json=updated_payload
    )
    assert resp.status_code == 204, resp.text

    # Retrieve the updated configuration.
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == updated_payload

    # Reject source deletion while an active configuration references it.
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{knowledgebox}/sources/dupe-src"
    )
    assert resp.status_code == 409, resp.text
    assert (
        resp.json()["detail"] == "Source is in use by agentic configuration(s): support"
    )

    # Soft-delete the configuration.
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )
    assert resp.status_code == 204, resp.text

    # Soft-deleted configurations are no longer retrievable.
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )
    assert resp.status_code == 404, resp.text

    # Soft-deleted configurations are excluded from lists.
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {}

    # Recreate a soft-deleted configuration using the same ID.
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support", json=updated_payload
    )
    assert resp.status_code == 201, resp.text

    # Soft-delete the recreated configuration before deleting its source.
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )
    assert resp.status_code == 204, resp.text

    # The source is deletable once no active configuration references it.
    resp = await nucliadb_agentic_api_http_client.delete(
        f"/api/v1/kb/{knowledgebox}/sources/dupe-src"
    )
    assert resp.status_code == 204, resp.text


async def test_agentic_config_not_found(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/missing"
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Source-reference validation tests
# ---------------------------------------------------------------------------


async def test_agentic_config_create_rejects_unknown_source_id(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    """Creating a config that references a non-existent source_id returns 422."""
    payload = {
        "title": "Agent",
        "smart_agent": {
            "sources": ["nono"],
            "history": True,
        },
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/agent1", json=payload
    )
    assert resp.status_code == 422, resp.text
    assert "Source(s) not found: nono" in resp.json()["detail"]


async def test_agentic_config_patch_rejects_unknown_source_id(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    """Patching a config to reference a non-existent source_id returns 422."""

    payload: Dict[str, Any] = {
        "type": "perplexity",
        "description": "Dupe",
        "enabled_domains": ["example.com"],
    }

    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/sources/dupe-src", json=payload
    )
    assert resp.status_code == 201, resp.text
    payload = {
        "title": "Agent",
        "smart_agent": {
            "mode": "reactive",
            "sources": ["dupe-src"],
            "history": True,
        },
    }
    await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/agent1", json=payload
    )

    # Patch to an invalid source_id
    bad_patch = {
        "title": "Agent",
        "smart_agent": {
            "mode": "reactive",
            "sources": ["bad-source"],
        },
    }
    resp = await nucliadb_agentic_api_http_client.patch(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/agent1", json=bad_patch
    )
    assert resp.status_code == 422, resp.text
