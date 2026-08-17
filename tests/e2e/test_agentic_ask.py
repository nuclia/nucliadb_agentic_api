from typing import Any, Dict
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from hyperforge_nucliadb_agentic.agent import NucliaDBAgent
from hyperforge_nucliadb_agentic.ask.model import AskRequest, SyncAskResponse

from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,  # Ignore localhost requests (e.g., to the test server)
        match_on=["scheme", "host", "port", "path", "nua_chat"],
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

    payload: Dict[str, Any] = {
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
        headers={
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
            "X-Synchronous": "true",
        },
    )
    response = await client.post("/ask", json=ask_request.model_dump(), timeout=1000)
    assert response.status_code == 200, response.text
    ask_response = SyncAskResponse.model_validate_json(response.content)
    assert "Debbie" in ask_response.answer
    assert ask_response.citations
    assert any(citation["context_id"] for citation in ask_response.citations.values())


async def test_agentic_ask_nucliadb_propagates_ask_request(
    nucliadb_agentic_api_http_client: AsyncClient,
    nucliadb_agentic_api_server: NucliaDBAgenticSessionManager,
    eric_dataset: str,
):
    source_payload = {
        "type": "nucliadb",
        "description": "Information about movies and actors.",
    }
    response = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/propagation-kb",
        json=source_payload,
    )
    assert response.status_code == 201, response.text

    config_payload = {
        "title": "Ask request propagation",
        "smart_agent": {
            "mode": "reactive",
            "sources": ["propagation-kb"],
        },
        "summarize": {},
    }
    response = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/agentic_configs/propagation",
        json=config_payload,
    )
    assert response.status_code == 201, response.text

    ask_request = AskRequest(
        query="Who is Carrie Fisher mother and what is she known for?",
        agentic_config_id="propagation",
        top_k=1,
        rag_strategies=[{"name": "hierarchy", "count": 12}],  # type: ignore
        rag_images_strategies=[{"name": "page_image", "count": 1}],  # type: ignore
        generative_model="gemini-2.5-flash-lite",
        debug=True,
    )

    captured_requests = []
    original_ask_agent = NucliaDBAgent.ask_agent

    async def capture_ask_agent(self, *args, **kwargs):
        # Capture the published tool boundary so VCR playback cannot hide local calls.
        base_request = AskRequest.model_validate_json(
            kwargs["memory"].arguments["ask_request"]
        )
        captured_requests.append(
            base_request.model_copy(update={"query": kwargs["question"]})
        )
        return await original_ask_agent(self, *args, **kwargs)

    async with AsyncClient(
        base_url=f"{nucliadb_agentic_api_http_client.base_url}/api/v1/kb/{eric_dataset}",
        headers={
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
            "X-Synchronous": "true",
        },
    ) as client:
        with patch.object(NucliaDBAgent, "ask_agent", capture_ask_agent):
            response = await client.post(
                "/ask", json=ask_request.model_dump(), timeout=1000
            )

        assert response.status_code == 200, response.text
        assert captured_requests
        for internal_request in captured_requests:
            assert internal_request.query
            assert internal_request.top_k == 1
            assert internal_request.rag_strategies[0].name == "hierarchy"
            assert internal_request.rag_strategies[0].count == 12
            assert internal_request.rag_images_strategies[0].name == "page_image"
            assert internal_request.rag_images_strategies[0].count == 1
            assert internal_request.generative_model == "gemini-2.5-flash-lite"
            assert internal_request.debug is True


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
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
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
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
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
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
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
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "eric",
        },
    )
    response = await client.post("/ask", json=ask_request.model_dump(), timeout=1000)
    assert response.status_code == 200, response.text
    assert b"Banana" in response.content
