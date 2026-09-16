from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.interaction import AnswerOperation
from hyperforge.pubsub import AgentDone

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


async def test_answer_publishes_user_facing_learning_id_and_restores_manager():
    session = object.__new__(NucliaDBAgenticSessionManager)
    session.settings = SimpleNamespace(question_timeout_seconds=1)
    session.keep_alive = AsyncMock()
    session.callback = AsyncMock()
    session.send_message = AsyncMock()
    session.process_event = MagicMock()

    question_memory = MagicMock()
    question_memory.steps = []
    question_memory.final_answer = "answer"
    question_memory.final_answer_citations = None
    question_memory.final_answer_urls = []
    question_memory.data_visualizations = []
    question_memory.original_question_uuid = "original-question"
    question_memory.actual_question_uuid = "actual-question"
    question_memory.session.id = "session"
    question_memory.save = AsyncMock()

    manager = MagicMock()
    manager.execute_raw = AsyncMock(
        side_effect=[
            (SimpleNamespace(learning_id="auxiliary-id"), 0, 0),
            (SimpleNamespace(learning_id="answer-id"), 0, 0),
        ]
    )
    original_execute_raw = manager.execute_raw
    manager.aclose = AsyncMock()

    async def run_agent(memory, current_manager):
        await current_manager.execute_raw("auxiliary")
        await current_manager.execute_raw("answer", memory=memory)

    state = SimpleNamespace(agent=run_agent, manager=manager)

    await session.answer(
        "account",
        "agent",
        "workflow",
        "topic",
        state,
        question_memory,
    )

    marker = session.callback.await_args_list[-1].args[1]
    assert marker.step.module == "_learning_id"
    assert marker.step.metadata == {"learning_id": "answer-id"}
    assert isinstance(session.send_message.await_args.args[1], AgentDone)
    assert manager.execute_raw is original_execute_raw
