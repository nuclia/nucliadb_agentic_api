from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge.manager import Manager
from hyperforge_nucliadb_agentic.ask.exceptions import NoRetrievalResultsError
from hyperforge_nucliadb_agentic.ask.learning_headers import (
    ALLOW_ACCESS_EXTERNAL_MODELS_HEADER,
    HeaderForwardingManager,
    get_learning_forwarded_headers,
    with_forwarded_headers,
)
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from hyperforge_nucliadb_agentic.ask.search.ask import ask
from mcp.server.fastmcp.exceptions import ResourceError
from nuclia.lib.nua import QueryRequest
from nuclia.lib.nua_responses import ChatModel, RerankModel
from nucliadb_models.search import NucliaDBClientType
from starlette.datastructures import Headers

from nucliadb_agentic_api.agentic.ask_handler import create_agentic_response
from nucliadb_agentic_api.agentic.ask_result import AgenticAskResult
from nucliadb_agentic_api.agentic.ask_transform_to_interaction import (
    interaction_from_ask_request,
)
from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager
from nucliadb_agentic_api.v1 import mcp_nucliadb

HEADER = ALLOW_ACCESS_EXTERNAL_MODELS_HEADER


class StopAsk(Exception):
    """Raised by the mocked ask() to stop right after capturing its arguments"""


def test_get_learning_forwarded_headers_is_case_insensitive():
    assert get_learning_forwarded_headers(
        {"X-Allow-Access-External-Models": "true"}
    ) == {HEADER: "true"}
    assert get_learning_forwarded_headers(
        Headers({"X-Allow-Access-External-Models": "false"})
    ) == {HEADER: "false"}


def test_get_learning_forwarded_headers_omits_missing_and_unrelated_headers():
    assert get_learning_forwarded_headers({}) == {}
    assert get_learning_forwarded_headers({"x-other": "1", "x-stf-user": "u"}) == {}


def test_with_forwarded_headers_returns_same_manager_without_headers():
    manager = MagicMock(spec=Manager)

    assert with_forwarded_headers(manager, None) is manager
    assert with_forwarded_headers(manager, {}) is manager


async def test_forwarding_manager_injects_headers_in_every_learning_call():
    inner = MagicMock(spec=Manager)
    for name in (
        "query_predict",
        "rephrase",
        "generate_stream",
        "rerank",
        "tokens_predict",
    ):
        setattr(inner, name, AsyncMock())
    manager = with_forwarded_headers(inner, {HEADER: "true"})
    assert isinstance(manager, HeaderForwardingManager)
    assert isinstance(manager, Manager)

    query = QueryRequest(text="q")
    chat = ChatModel(question="q", user_id="u")
    rerank = RerankModel(question="q", user_id="u", context={"1": "text"})

    await manager.query_predict(query, kbid="kb")
    await manager.rephrase(MagicMock(), kbid="kb")
    await manager.generate_stream(chat, kbid="kb")
    await manager.rerank(rerank, kbid="kb")
    await manager.tokens_predict("text", kbid="kb")

    expected = {HEADER: "true"}
    assert inner.query_predict.await_args.kwargs == {
        "kbid": "kb",
        "extra_headers": expected,
    }
    assert inner.rephrase.await_args.kwargs["extra_headers"] == expected
    assert inner.generate_stream.await_args.kwargs["extra_headers"] == expected
    assert inner.rerank.await_args.kwargs["extra_headers"] == expected
    assert inner.tokens_predict.await_args.kwargs["extra_headers"] == expected


async def test_forwarding_manager_merges_per_call_headers():
    inner = MagicMock(spec=Manager)
    inner.generate_stream = AsyncMock()
    manager = with_forwarded_headers(inner, {HEADER: "true"})

    await manager.generate_stream(
        ChatModel(question="q", user_id="u"),
        extra_headers={"X-Show-Consumption": "false"},
    )

    assert inner.generate_stream.await_args.kwargs["extra_headers"] == {
        HEADER: "true",
        "X-Show-Consumption": "false",
    }


def test_forwarding_manager_delegates_other_attributes():
    inner = MagicMock(spec=Manager)
    inner.drivers = {"driver": object()}
    inner.send_rao_origin = False
    manager = with_forwarded_headers(inner, {HEADER: "true"})

    assert manager.drivers is inner.drivers
    assert manager.send_rao_origin is False


async def _run_ask(predict_manager, forwarded_headers=None, **ask_kwargs):
    return await ask(
        search_sdk=MagicMock(),
        reader_sdk=MagicMock(),
        kbid="kbid",
        predict_manager=predict_manager,
        user_id="user",
        client_type=NucliaDBClientType.API,
        origin="",
        forwarded_headers=forwarded_headers,
        **ask_kwargs,
    )


@pytest.mark.parametrize(
    ("forwarded_headers", "expected"),
    [({HEADER: "true"}, True), (None, False)],
)
async def test_ask_forwards_headers_to_every_predict_call(forwarded_headers, expected):
    inner = MagicMock(spec=Manager)
    inner.rephrase = AsyncMock(
        return_value=SimpleNamespace(rephrased_query="rephrased", use_chat_history=True)
    )
    seen: dict = {}

    async def fake_retrieval_step(**kwargs):
        # Retrieval, rerank and answer generation all use this manager
        seen["manager"] = kwargs["predict_manager"]
        raise NoRetrievalResultsError()

    ask_request = AskRequest(
        query="question", chat_history=[{"author": "USER", "text": "hi"}]
    )
    with patch(
        "hyperforge_nucliadb_agentic.ask.search.ask.retrieval_step",
        side_effect=fake_retrieval_step,
    ):
        await _run_ask(inner, forwarded_headers, ask_request=ask_request)

    rephrase_headers = inner.rephrase.await_args.kwargs.get("extra_headers")
    if expected:
        assert rephrase_headers == {HEADER: "true"}
        assert isinstance(seen["manager"], HeaderForwardingManager)
    else:
        assert rephrase_headers is None
        assert seen["manager"] is inner


def test_interaction_from_ask_request_sets_headers():
    interaction = interaction_from_ask_request(
        AskRequest(query="question"), headers={HEADER: "true"}
    )

    assert interaction.headers == {HEADER: "true"}
    assert interaction_from_ask_request(AskRequest(query="q")).headers == {}


async def test_agentic_ask_result_sends_forwarded_headers_in_interaction():
    captured = {}

    async def fake_stream_response(
        app, receiver, account, kbid, session, interaction, **kw
    ):
        captured["headers"] = dict(interaction.headers)
        yield AragAnswer(operation=AnswerOperation.DONE)

    result = AgenticAskResult(
        app=MagicMock(),
        kbid="kbid",
        ask_request=AskRequest(query="question"),
        agentic_config_id="config",
        account="account",
        client_type=NucliaDBClientType.API,
        forwarded_headers={HEADER: "true"},
    )
    with patch(
        "nucliadb_agentic_api.agentic.ask_result.stream_response",
        new=fake_stream_response,
    ):
        await result.start()
        await result.task

    assert captured["headers"] == {HEADER: "true"}


async def test_create_agentic_response_passes_forwarded_headers():
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
            ask_request=AskRequest(query="question"),
            agentic_config_id="config",
            user_id="user",
            client_type=NucliaDBClientType.API,
            origin="",
            x_synchronous=True,
            forwarded_headers={HEADER: "true"},
        )

    assert result_class.call_args.kwargs["forwarded_headers"] == {HEADER: "true"}


def _mcp_request_ctx(headers: dict[str, str]):
    return SimpleNamespace(request=SimpleNamespace(headers=Headers(headers)))


async def _call_search_tool():
    return await mcp_nucliadb.call_tool(
        predict_manager=MagicMock(),
        x_stf_account="account",
        x_nucliadb_user="user",
        x_ndb_client=NucliaDBClientType.API,
        x_forwarded_for="",
        ndb_reader=MagicMock(),
        ndb_search=MagicMock(),
        kbid="kbid",
        name="search_documents",
        arguments={"query": "question"},
    )


@pytest.mark.parametrize(
    ("request_headers", "expected"),
    [
        ({"X-Allow-Access-External-Models": "false"}, {HEADER: "false"}),
        ({}, {}),
    ],
)
async def test_mcp_search_documents_forwards_request_headers(request_headers, expected):
    ask_mock = AsyncMock(side_effect=StopAsk())
    token = mcp_nucliadb.request_ctx.set(_mcp_request_ctx(request_headers))  # type: ignore[arg-type]
    try:
        with patch.object(mcp_nucliadb, "ask", ask_mock), pytest.raises(ResourceError):
            await _call_search_tool()
    finally:
        mcp_nucliadb.request_ctx.reset(token)

    assert ask_mock.await_args.kwargs["forwarded_headers"] == expected


async def test_mcp_search_documents_without_request_context_forwards_nothing():
    ask_mock = AsyncMock(side_effect=StopAsk())
    with patch.object(mcp_nucliadb, "ask", ask_mock), pytest.raises(ResourceError):
        await _call_search_tool()

    assert ask_mock.await_args.kwargs["forwarded_headers"] == {}


async def test_activate_passes_forwarded_headers_to_get_state():
    session = object.__new__(NucliaDBAgenticSessionManager)
    session.question_topic = MagicMock(return_value="answer-topic")
    session.callback = AsyncMock()
    session.send_message = AsyncMock()
    session.agent_manager = MagicMock()
    session.agent_manager.get_agent_config = AsyncMock(
        return_value=SimpleNamespace(memory=None, rules=SimpleNamespace(rules=[]))
    )
    session.memory = {"session": MagicMock()}
    session.tasks = []
    session.cache = MagicMock()
    session.settings = SimpleNamespace(
        internal_nucliadb_url="",
        internal_nucliadb=False,
        external_nucliadb_url="",
        external_nucliadb_key="",
        internal_nua_api="",
        internal_nua=False,
        local_openai=None,
        external_nua_api_key=None,
        standalone=False,
    )
    session.answer = AsyncMock()
    message = SimpleNamespace(
        account="account",
        agent_id="kbid",
        session="session",
        question="question",
        question_id="question-id",
        workflow_id="workflow",
        arguments={},
        headers={"x-allow-access-external-models": "true", "x-other": "1"},
        streaming=True,
        chat_history=[],
    )

    with patch(
        "nucliadb_agentic_api.server.session.get_state", new=AsyncMock()
    ) as get_state:
        await session.activate(message)

    assert get_state.await_args.kwargs["extra_headers"] == {HEADER: "true"}
