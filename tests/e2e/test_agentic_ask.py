from httpx import AsyncClient
from nuclia.sdk import AsyncNucliaSearch
import pytest

from nucliadb_agentic_api.ask.model import AskRequest

from nucliadb_agentic_api.ask.search.ask import AskResult

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,  # Ignore localhost requests (e.g., to the test server)
    ),
    pytest.mark.asyncio,
]


async def test_agentic_ask_nucliadb(
    nucliadb_agentic_api_http_client: AsyncClient,
    article_dataset: str,
):
    # In this basic test we just want to verify that the ask endpoint is working end-to-end with a simple question, without any agentic config.
    # We use a dataset with a single article to have a predictable answer.

    payload = {
        "type": "nucliadb",
        "title": "My KB source",
        "config": {
            "filter_expression": "label=important",
            "labels": ["important", "verified"],
        },
    }

    # Create
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{article_dataset}/sources/src-nucliadb", json=payload
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "title": "Agentic config with NucliaDB source",
        "config": {
            "smart_agent": {
                "sources": [{"type": "nucliadb", "source_id": "src-nucliadb"}],
            }
        },
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{article_dataset}/agentic_configs/key-nucliadb", json=payload
    )
    assert resp.status_code == 201, resp.text

    ask_request = AskRequest(
        query="What is it about?", agentic_config_id="key-nucliadb"
    )

    client = AsyncClient(
        base_url=f"{nucliadb_agentic_api_http_client.base_url}/api/v1/kb/{article_dataset}",
        headers={"X-NUCLIADB-ROLES": "MANAGER;READER;WRITER"},
    )
    response = await client.post("/ask", json=ask_request.model_dump())
    assert response.status_code == 200, response.text
    assert b"Agents" in response.content
