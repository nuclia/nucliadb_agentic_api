import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from hyperforge_nucliadb_agentic.ask.account import (
    AccountContextMiddleware,
    get_account_id,
)
from hyperforge_nucliadb_agentic.ask.search import rpc
from nuclia.lib.nua import QueryRequest
from nuclia.lib.nua_responses import QueryInfo
from nucliadb_models.search import FindOptions, FindRequest, KnowledgeboxFindResults
from nucliadb_sdk import NucliaDBAsync
from nucliadb_utils.settings import AuditSettings, nuclia_settings
from starlette.responses import JSONResponse

from nucliadb_agentic_api import app as app_module
from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.settings import Settings


@pytest.fixture
async def account_app(monkeypatch):
    monkeypatch.setattr(nuclia_settings, "onprem", False)
    monkeypatch.setattr(
        nuclia_settings, "nuclia_inner_predict_url", "http://predict.internal"
    )
    monkeypatch.setattr(rpc, "__SDK", {})
    recorded: list[httpx.Request] = []
    sdks: list[NucliaDBAsync] = []

    async def handler(outgoing: httpx.Request) -> httpx.Response:
        recorded.append(outgoing)
        await asyncio.sleep(0)
        if "/predict/query" in outgoing.url.path:
            return httpx.Response(
                200,
                json=QueryInfo(
                    query="question",
                    visual_llm=False,
                    max_context=1000,
                    entities=None,
                    sentence=None,
                ).model_dump(),
            )
        if "/predict/chat" in outgoing.url.path:
            return httpx.Response(200, content=b'{"answer":"ok"}\n')
        if outgoing.url.path.endswith("/find"):
            return httpx.Response(
                200, json=KnowledgeboxFindResults(resources={}).model_dump()
            )
        if outgoing.url.path.endswith(("/retrieve", "/configuration")):
            return httpx.Response(200, json={})
        raise AssertionError(f"Unexpected request: {outgoing.method} {outgoing.url}")

    def nua_client(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    def ndb_client(**kwargs):
        sdk = NucliaDBAsync(_httpx_transport=httpx.MockTransport(handler), **kwargs)
        sdks.append(sdk)
        return sdk

    monkeypatch.setattr("nuclia.lib.nua.build_httpx_async_client", nua_client)
    monkeypatch.setattr(app_module, "NucliaDBAsync", ndb_client)
    monkeypatch.setattr(rpc, "NucliaDBAsync", ndb_client)
    for name in (
        "setup_telemetry",
        "start_audit_utility",
        "stop_audit_utility",
        "clean_telemetry",
    ):
        monkeypatch.setattr(app_module, name, AsyncMock())
    monkeypatch.setattr(
        app_module.RedisBroker, "from_url", Mock(return_value=AsyncMock())
    )
    monkeypatch.setattr(
        app_module.AgenticConfigs, "from_settings", AsyncMock(return_value=AsyncMock())
    )
    monkeypatch.setattr(
        app_module.Sources, "from_settings", AsyncMock(return_value=AsyncMock())
    )
    app = HTTPApplication(
        settings=Settings(
            internal_nucliadb=True,
            internal_nucliadb_url="http://{component}.internal/api",
        ),
        data_manager_settings=DataManagerSettings(postgresql_dsn="postgresql://unused"),
        audit_settings=AuditSettings(audit_hash_seed=1),
    )
    await app.startup()

    async def ask_with_internal_calls(**kwargs):
        kbid = kwargs["kbid"]
        query = kwargs["ask_request"].query
        manager = kwargs["predict_manager"]
        await manager.query_predict(QueryRequest(text=query), kbid=kbid)
        await rpc.find(
            kwargs["search_sdk"],
            kbid,
            FindRequest(query=query, features=[FindOptions.RELATIONS]),
            kwargs["client_type"],
            kwargs["user_id"],
            kwargs["origin"],
        )
        await app.arag_search.session.post(f"/v1/kb/{kbid}/retrieve", json={})
        await app.arag_reader.session.get(f"/v1/kb/{kbid}/configuration")

        async def stream():
            async with manager.nua.stream_client.stream(
                "POST", "/api/internal/predict/chat"
            ) as response:
                yield await response.aread()

        return Mock(
            nuclia_learning_id="test-learning-id",
            json=AsyncMock(return_value='{"answer":"ok"}'),
            ndjson_stream=stream,
            to_sync_response=AsyncMock(
                return_value=SimpleNamespace(
                    answer="ok", citation_footnote_to_context={}
                )
            ),
        )

    monkeypatch.setattr("nucliadb_agentic_api.v1.ask.ask", ask_with_internal_calls)
    monkeypatch.setattr(
        "nucliadb_agentic_api.v1.mcp_nucliadb.ask", ask_with_internal_calls
    )
    monkeypatch.setattr(
        rpc,
        "get_resource_uuid_from_slug",
        AsyncMock(return_value="00000000-0000-0000-0000-000000000001"),
    )
    try:
        yield app, recorded
    finally:
        await app.shutdown()
        for sdk in sdks:
            await sdk.session.aclose()


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/kb/kbid/ask",
        "/api/v1/kb/kbid/resource/00000000-0000-0000-0000-000000000001/ask",
        "/api/v1/kb/kbid/slug/resource/ask",
    ],
)
async def test_ask_sends_account_in_internal_calls(account_app, path):
    app, recorded = account_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            path,
            json={"query": "question", "features": ["relations"]},
            headers={
                "X-NUCLIADB-ROLES": "READER",
                "X-NUCLIADB-ACCOUNT": "owning-account",
                "X-STF-ACCOUNT": "other-account",
            },
        )
    assert response.status_code == 200, response.text
    assert len(recorded) == 5
    for outgoing in recorded:
        if outgoing.url.host.startswith("predict"):
            assert outgoing.headers["X-STF-ACCOUNT"] == "owning-account"
            if "/query" in outgoing.url.path:
                assert outgoing.headers["X-STF-KBID"] == "kbid"
        else:
            assert outgoing.headers["X-NUCLIADB-ACCOUNT"] == "owning-account"
    assert get_account_id() is None


async def test_concurrent_accounts_and_missing_account_do_not_leak(account_app, caplog):
    app, recorded = account_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *[
                client.post(
                    "/api/v1/kb/kbid/ask",
                    json={"query": account},
                    headers={
                        "X-NUCLIADB-ROLES": "READER",
                        "X-NUCLIADB-ACCOUNT": account,
                    },
                )
                for account in ("first-account", "second-account")
            ]
        )
        assert all(response.status_code == 200 for response in responses)
        for outgoing in recorded:
            if "/predict/query" in outgoing.url.path:
                assert (
                    outgoing.headers["X-STF-ACCOUNT"]
                    == json.loads(outgoing.content)["text"]
                )
            elif outgoing.url.path.endswith("/find"):
                assert (
                    outgoing.headers["X-NUCLIADB-ACCOUNT"]
                    == json.loads(outgoing.content)["query"]
                )
        count = len(recorded)
        with pytest.raises(ValueError, match="Missing authenticated account ID"):
            await client.post(
                "/api/v1/kb/kbid/ask",
                json={"query": "question"},
                headers={"X-NUCLIADB-ROLES": "READER"},
            )
    assert len(recorded) == count
    assert "Missing authenticated account ID" in caplog.text
    assert get_account_id() is None


async def test_cached_mcp_server_uses_current_account(account_app):
    app, recorded = account_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        for account in ("first-account", "second-account"):
            response = await client.post(
                "/api/v1/kb/kbid/mcp",
                headers={
                    "X-NUCLIADB-ROLES": "READER",
                    "X-STF-USER": "test-user",
                    "X-STF-ACCOUNT": account,
                    "X-STF-ACCOUNT-TYPE": "test",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-03-26",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "search_documents",
                        "arguments": {"query": account},
                    },
                },
            )
            assert response.status_code == 200, response.text
            assert not response.json()["result"].get("isError", False), response.text
    for outgoing in recorded:
        if "/predict/query" in outgoing.url.path:
            assert (
                outgoing.headers["X-STF-ACCOUNT"]
                == json.loads(outgoing.content)["text"]
            )
        elif outgoing.url.path.endswith("/find"):
            assert (
                outgoing.headers["X-NUCLIADB-ACCOUNT"]
                == json.loads(outgoing.content)["query"]
            )
    assert len(recorded) == 8
    assert get_account_id() is None


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/kb/kbid/ask",
        "/api/v1/kb/kbid/resource/00000000-0000-0000-0000-000000000001/ask",
        "/api/v1/kb/kbid/slug/resource/ask",
    ],
)
async def test_agentic_ask_uses_authenticated_account(account_app, monkeypatch, path):
    app, _ = account_app

    async def response(**kwargs):
        return JSONResponse({"account": kwargs["account"]})

    monkeypatch.setattr("nucliadb_agentic_api.v1.ask.create_agentic_response", response)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        result = await client.post(
            path,
            json={"query": "question", "agentic_config_id": "config"},
            headers={
                "X-NUCLIADB-ROLES": "READER",
                "X-STF-ACCOUNT": "authenticated-account",
            },
        )
    assert result.status_code == 200, result.text
    assert result.json() == {"account": "authenticated-account"}


async def test_websocket_account_is_available_to_internal_clients(account_app):
    app, recorded = account_app

    async def endpoint(scope, receive, send):
        await app.predict_manager.query_predict(
            QueryRequest(text="question"), kbid="kbid"
        )
        await rpc.get_sdk("search").session.post("/v1/kb/kbid/retrieve", json={})

    await AccountContextMiddleware(endpoint)(
        {"type": "websocket", "headers": [(b"x-stf-account", b"websocket-account")]},
        AsyncMock(),
        AsyncMock(),
    )
    assert len(recorded) == 2
    assert recorded[0].headers["X-STF-ACCOUNT"] == "websocket-account"
    assert recorded[1].headers["X-NUCLIADB-ACCOUNT"] == "websocket-account"
    assert get_account_id() is None
