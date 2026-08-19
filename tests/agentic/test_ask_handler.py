from unittest.mock import AsyncMock, patch

import pytest
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from nuclia.lib.nua_responses import CitationsType
from nucliadb_models.search import NucliaDBClientType

from nucliadb_agentic_api.agentic.ask_handler import create_agentic_response


@pytest.mark.parametrize(
    ("citations", "expected"),
    [(None, CitationsType.LLM_FOOTNOTES), (False, False)],
)
async def test_citation_defaults(citations, expected):
    ask_request = AskRequest(query="question", citations=citations)
    ask_result = AsyncMock()
    ask_result.start.return_value = "learning-id"
    ask_result.json.return_value = "{}"

    with patch(
        "nucliadb_agentic_api.agentic.ask_handler.AgenticAskResult",
        return_value=ask_result,
    ) as result_class:
        await create_agentic_response(
            app=AsyncMock(),
            kbid="kbid",
            account="account",
            ask_request=ask_request,
            agentic_config_id="config",
            user_id="user",
            client_type=NucliaDBClientType.API,
            origin="",
            x_synchronous=True,
        )

    assert result_class.call_args.kwargs["ask_request"].citations == expected
