import pytest
from httpx import AsyncClient
from nuclia.sdk import AsyncNucliaSearch

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,  # Ignore localhost requests (e.g., to the test server)
        match_on=["scheme", "host", "port", "path", "nua_chat", "localhost"],
    ),
    pytest.mark.asyncio,
]


async def test_basic_ask(
    nucliadb_agentic_api_http_client: AsyncClient,
    article_dataset: str,
):
    # In this basic test we just want to verify that the ask endpoint is working end-to-end with a simple question, without any agentic config.
    # We use a dataset with a single article to have a predictable answer.

    ns = AsyncNucliaSearch()

    response = await ns.ask(
        query="What is it about?",
        url=f"{nucliadb_agentic_api_http_client.base_url}/api/v1/kb/{article_dataset}",
    )

    assert b"Agents" in response.answer
