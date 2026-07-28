from typing import Any, Dict

from httpx import AsyncClient


async def test_agentic_configs_api(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):

    payload: Dict[str, Any] = {
        "type": "nucliadb",
        "description": "Nucliadb",
    }
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

    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support", json=payload
    )
    assert resp.status_code == 201, resp.text

    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == payload

    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"support": payload}

    updated_payload = {
        "title": "Updated support agent",
        "summarize": {"conversational": False},
    }
    resp = await nucliadb_agentic_api_http_client.patch(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support", json=updated_payload
    )
    assert resp.status_code == 204, resp.text

    resp = await nucliadb_agentic_api_http_client.get(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/support"
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == updated_payload


async def test_agentic_config_source_validation(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
    """Integration: source_id references in smart_agent sources are validated."""

    # 1. Attempt to create a config that references a non-existent source → 422
    bad_payload = {
        "title": "Invalid agent",
        "smart_agent": {
            "sources": ["bad-source"],
        },
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/bad-agent", json=bad_payload
    )
    assert resp.status_code == 422, resp.text
    assert "Source(s) not found: bad-source" in resp.json()["detail"]

    # 2. Create the source that will be referenced
    source_payload = {
        "type": "nucliadb",
        "title": "Validation source",
        "labels": ["ok"],
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/sources/valid-src", json=source_payload
    )
    assert resp.status_code == 201, resp.text

    # 3. Now the config with a valid source_id is accepted → 201
    good_payload = {
        "title": "Valid agent",
        "smart_agent": {
            "sources": ["valid-src"],
        },
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/good-agent", json=good_payload
    )
    assert resp.status_code == 201, resp.text

    # 4. Patch also validates → 422 when trying to use a non-existent source
    resp = await nucliadb_agentic_api_http_client.patch(
        f"/api/v1/kb/{knowledgebox}/agentic_configs/good-agent",
        json={
            "title": "Patched",
            "smart_agent": {
                "sources": ["no-such-source"],
            },
        },
    )
    assert resp.status_code == 422, resp.text
