import asyncio
from builtins import BaseExceptionGroup, ExceptionGroup
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.interaction import AnswerOperation

from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager


@pytest.mark.parametrize(
    ("ask_request", "error_detail"),
    [
        ('{"query": "question", "top_k": "invalid"}', '"loc":["top_k"]'),
        ("", "json_invalid"),
    ],
)
async def test_activate_rejects_invalid_ask_request_before_loading_config(
    ask_request, error_detail
):
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
    assert error_detail in answer.exception.detail
    session.send_message.assert_awaited_once()


async def test_activate_reports_sanitized_failure():
    session = object.__new__(NucliaDBAgenticSessionManager)
    session.question_topic = MagicMock(return_value="answer-topic")
    session.callback = AsyncMock()
    session.send_message = AsyncMock()
    session.agent_manager = MagicMock()
    session.settings = SimpleNamespace(
        internal_nucliadb_url="",
        internal_nucliadb=False,
        external_nucliadb_url="",
        external_nucliadb_key="",
    )
    session.agent_manager.get_agent_config = AsyncMock(
        side_effect=RuntimeError("token=secret-value invalid configuration")
    )

    message = SimpleNamespace(
        account="account",
        agent_id="kbid",
        session="session",
        question_id="question",
        workflow_id="workflow",
        arguments={"ask_request": '{"query": "question"}'},
    )

    await session.activate(message)

    answer = session.callback.await_args.args[1]
    assert answer.operation == AnswerOperation.ERROR
    assert answer.exception.detail == "token=[REDACTED] invalid configuration"


async def test_answer_unwraps_exception_group_for_user():
    session = object.__new__(NucliaDBAgenticSessionManager)
    session.settings = SimpleNamespace(question_timeout_seconds=10)
    session.broker = SimpleNamespace(keepalive_seconds=10)
    session.callback = AsyncMock()

    state = SimpleNamespace(
        agent=AsyncMock(
            side_effect=ExceptionGroup(
                "task group", [RuntimeError("Authentication token expired")]
            )
        ),
        manager=SimpleNamespace(aclose=AsyncMock()),
    )
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        set_feedback_fn=MagicMock(),
        set_oauth_fn=MagicMock(),
        set_oauth_callback_fn=MagicMock(),
        session=SimpleNamespace(id="session"),
        final_answer=None,
        final_answer_citations=None,
        final_answer_urls=None,
        data_visualizations=None,
    )

    await session.answer(
        "account", "agent", "workflow", "topic", state, question_memory
    )

    answer = session.callback.await_args_list[-1].args[1]
    assert answer.operation == AnswerOperation.ERROR
    assert answer.exception.detail == "Authentication token expired"


async def test_answer_handles_mixed_cancellation_exception_group():
    session = object.__new__(NucliaDBAgenticSessionManager)
    session.settings = SimpleNamespace(question_timeout_seconds=10)
    session.broker = SimpleNamespace(keepalive_seconds=10)
    session.callback = AsyncMock()

    state = SimpleNamespace(
        agent=AsyncMock(
            side_effect=BaseExceptionGroup(
                "task group",
                [asyncio.CancelledError(), RuntimeError("provider unavailable")],
            )
        ),
        manager=SimpleNamespace(aclose=AsyncMock()),
    )
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        set_feedback_fn=MagicMock(),
        set_oauth_fn=MagicMock(),
        set_oauth_callback_fn=MagicMock(),
        session=SimpleNamespace(id="session"),
        final_answer=None,
        final_answer_citations=None,
        final_answer_urls=None,
        data_visualizations=None,
    )

    await session.answer(
        "account", "agent", "workflow", "topic", state, question_memory
    )

    answer = session.callback.await_args_list[-1].args[1]
    assert answer.operation == AnswerOperation.ERROR
    assert answer.exception.detail == "provider unavailable"
