import pytest
from httpx import AsyncClient
from hyperforge_nucliadb_agentic.ask.model import AskRequest

from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,  # Ignore localhost requests (e.g., to the test server)
    ),
    pytest.mark.asyncio,
]


async def test_agentic_ask_nucliadb(
    nucliadb_agentic_api_http_client: AsyncClient,
    nucliadb_agentic_api_server: NucliaDBAgenticSessionManager,
    eric_dataset: str,
):
    # In this basic test we just want to verify that the ask endpoint is working end-to-end with a simple question, without any agentic config.
    # We use a dataset with a single article to have a predictable answer.

    payload = {
        "type": "nucliadb",
        "description": "Information about movies and actors.",
    }

    # Create
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/full-kb", json=payload
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "title": "Agentic config with NucliaDB source",
        "rephrase": {},
        "smart_agent": {"mode": "reactive", "sources": ["full-kb"]},
        "summarize": {},
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/agentic_configs/key-nucliadb", json=payload
    )
    assert resp.status_code == 201, resp.text

    ask_request = AskRequest(
        query="Who is Carrie Fisher mother and what is she known for?",
        agentic_config_id="key-nucliadb",
    )

    client = AsyncClient(
        base_url=f"{nucliadb_agentic_api_http_client.base_url}/api/v1/kb/{eric_dataset}",
        headers={"X-NUCLIADB-ROLES": "MANAGER;READER;WRITER"},
    )
    response = await client.post("/ask", json=ask_request.model_dump(), timeout=1000)
    assert response.status_code == 200, response.text
    assert b"Agents" in response.content


async def test_agentic_ask_perplexity(
    nucliadb_agentic_api_http_client: AsyncClient,
    nucliadb_agentic_api_server: NucliaDBAgenticSessionManager,
    eric_dataset: str,
):
    # In this basic test we just want to verify that the ask endpoint is working end-to-end with a simple question, without any agentic config.
    # We use a dataset with a single article to have a predictable answer.

    payload = {
        "type": "nucliadb",
        "description": "Our favorite pastries recipes",
        "filter_expression": {
            "field": {
                "prop": "resource_mimetype",
                "type": "application",
                "subtype": "pdf",
            }
        },
    }

    # Create
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/recipes-kb",
        json=payload,
        headers={
            "X-NUCLIADB-ROLES": "MANAGER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "eric",
        },
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "type": "perplexity",
        "description": "Other recipes on internet",
        "domains": ["https://www.allrecipes.com"],
    }

    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/perplexity",
        json=payload,
        headers={
            "X-NUCLIADB-ROLES": "MANAGER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "eric",
        },
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "rephrase": {},
        "smart_agent": {
            "mode": "reactive",
            "sources": ["recipes-kb", "perplexity"],
            "extra_prompt": "When asked about a recipe, always recommend the most relevant one from our favorite recipes, but if asked about a very specific recipe that is not in our favorites, check the recipes from internet.",
        },
        "summarize": {},
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/agentic_configs/default",
        json=payload,
        headers={
            "X-NUCLIADB-ROLES": "MANAGER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "eric",
        },
    )
    assert resp.status_code == 201, resp.text

    ask_request = AskRequest(
        query="I want a desert that is healthy and tasty.",
        agentic_config_id="default",
    )

    client = AsyncClient(
        base_url=f"{nucliadb_agentic_api_http_client.base_url}/api/v1/kb/{eric_dataset}",
        headers={
            "X-NUCLIADB-ROLES": "MANAGER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "eric",
        },
    )
    response = await client.post("/ask", json=ask_request.model_dump(), timeout=1000)
    assert response.status_code == 200, response.text
    assert b"cookbooks" in response.content
