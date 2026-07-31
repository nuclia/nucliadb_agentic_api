from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.interaction import AnswerOperation

from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager


@pytest.mark.parametrize(
    "ask_request",
    [
        '{"query": "question", "top_k": "invalid"}',
        "",
    ],
)
async def test_activate_rejects_invalid_ask_request_before_loading_config(ask_request):
    session = object.__new__(NucliaDBAgenticSessionManager)
    session.question_topic = MagicMock(return_value="answer-topic")
    session.callback = AsyncMock()
    session.send_message = AsyncMock()
    session.agent_manager = MagicMock()

    message = SimpleNamespace(
        account="account",
        agent_id="kbid",
        session="session",
        question_id="question",
        workflow_id="workflow",
        arguments={"ask_request": ask_request},
    )

    await session.activate(message)

    session.agent_manager.get_agent_config.assert_not_called()
    session.callback.assert_awaited_once()
    answer = session.callback.await_args.args[1]
    assert answer.operation == AnswerOperation.ERROR
    assert answer.exception.detail == "Invalid ask_request argument"
    session.send_message.assert_awaited_once()
