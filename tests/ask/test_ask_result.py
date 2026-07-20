from unittest.mock import MagicMock, patch

import pytest
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge.models import ExternalUsage, Step
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from nucliadb_models.search import NucliaDBClientType

from nucliadb_agentic_api.agentic.ask_result import AgenticAskResult


@pytest.mark.asyncio
async def test_step_external_usage_is_reported():
    step = Step(
        original_question_uuid="question",
        actual_question_uuid="question",
        module="google",
        title="Search results",
        timeit=0.1,
        input_nuclia_tokens=None,
        output_nuclia_tokens=None,
        agent_path="/context/google",
        external_usage=[
            ExternalUsage(
                provider="google",
                model="gemini-2.5-flash",
                input_tokens=10,
                output_tokens=20,
            )
        ],
    )
    audit = MagicMock()

    async def messages(*args, **kwargs):
        yield AragAnswer(step=step)
        yield AragAnswer(operation=AnswerOperation.DONE)

    result = AgenticAskResult(
        kbid="kbid",
        ask_request=AskRequest(query="question"),
        agentic_config_id="config",
        account="account",
        client_type=NucliaDBClientType.API,
        app=MagicMock(),
    )

    with (
        patch(
            "nucliadb_agentic_api.agentic.ask_result.stream_response",
            new=messages,
        ),
        patch(
            "nucliadb_agentic_api.agentic.ask_result.get_audit",
            return_value=audit,
        ),
    ):
        await result.loop()

    audit.report_step_usage.assert_called_once_with(
        account_id="account",
        kbid="kbid",
        client_type=NucliaDBClientType.API,
        step=step,
    )
