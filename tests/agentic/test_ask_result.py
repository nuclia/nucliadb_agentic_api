from unittest.mock import MagicMock

import pytest
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge_nucliadb_agentic.ask.model import AskRequest, SyncAskResponse
from nucliadb_models.search import NucliaDBClientType

from nucliadb_agentic_api.agentic.ask_result import AgenticAskResult


@pytest.mark.asyncio
async def test_agentic_ask_result_json_serializes_synchronously():
    """Ensure the agentic ask result supports the endpoint's sync response mode."""
    # Regression coverage for the agentic endpoint's synchronous response mode.
    result = AgenticAskResult(
        app=MagicMock(),
        kbid="kbid",
        ask_request=AskRequest(query="question"),
        agentic_config_id="config",
        account="account",
        client_type=NucliaDBClientType.API,
        origin="",
        resource=None,
    )
    await result.queue.put(AragAnswer(operation=AnswerOperation.DONE))

    response = SyncAskResponse.model_validate_json(await result.json())

    assert response.status == "success"
    assert response.answer == ""
