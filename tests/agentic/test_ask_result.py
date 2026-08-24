from unittest.mock import MagicMock

import pytest
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge_nucliadb_agentic.ask.model import AskRequest, SyncAskResponse
from nucliadb_models.search import NucliaDBClientType

from nucliadb_agentic_api.agentic.ask_result import AgenticAskResult
from nucliadb_agentic_api.agentic.ask_transform_to_interaction import (
    interaction_from_ask_request,
)


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


def test_ask_interaction_preserves_explicit_fields_only():
    omitted = interaction_from_ask_request(AskRequest(query="question"))
    omitted_request = AskRequest.model_validate_json(omitted.arguments["ask_request"])
    explicit = interaction_from_ask_request(
        AskRequest(query="question", generative_model=None, reasoning=False)
    )
    explicit_request = AskRequest.model_validate_json(explicit.arguments["ask_request"])

    assert "generative_model" not in omitted_request.model_fields_set
    assert "reasoning" not in omitted_request.model_fields_set
    assert "generative_model" in explicit_request.model_fields_set
    assert "reasoning" in explicit_request.model_fields_set
