from httpx import AsyncClient


async def test_agentic_configs_api(
    nucliadb_agentic_api_http_client: AsyncClient, knowledgebox: str
):
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
        "config": {"summarize": {"conversational": False}},
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
