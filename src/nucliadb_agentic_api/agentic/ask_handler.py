import asyncio
from typing import TYPE_CHECKING

from fastapi import Response
from hyperforge.interaction import AragAnswer
from nucliadb_models.search import (
    KnowledgeboxFindResults,
    NucliaDBClientType,
)
from starlette.responses import StreamingResponse

from nucliadb_agentic_api.agentic.ask_requester import interaction
from nucliadb_agentic_api.ask.audit import ChatAuditor
from nucliadb_agentic_api.ask.model import (
    AskRequest,
    AugmentedContext,
    PromptContext,
    PromptContextOrder,
    parse_max_tokens,
)
from nucliadb_agentic_api.ask.search import rpc
from nucliadb_agentic_api.ask.search.ask import (
    AskResult,
    handled_ask_exceptions,
)
from nucliadb_agentic_api.ask.search.metrics import AskMetrics
from nucliadb_agentic_api.v1.utils import websocket_to_ask

if TYPE_CHECKING:
    from nucliadb_agentic_api.app import HTTPApplication


@handled_ask_exceptions
async def create_agentic_response(
    app: "HTTPApplication",
    kbid: str,
    account: str,
    ask_request: AskRequest,
    agentic_config_id: str,
    user_id: str,
    client_type: NucliaDBClientType,
    origin: str,
    x_synchronous: bool,
    resource: str | None = None,
    extra_predict_headers: dict[str, str] | None = None,
) -> Response:
    ask_request.max_tokens = parse_max_tokens(ask_request.max_tokens)

    find_results = KnowledgeboxFindResults(resources={})

    queue: asyncio.Queue[AragAnswer] = asyncio.Queue()

    task = asyncio.create_task(
        interaction(
            app=app,
            agentic_config_id=agentic_config_id,
            ask_request=ask_request,
            queue=queue,
            kbid=kbid,
            account=account,
            origin=origin,
        )
    )

    ask_result = AskResult(
        kbid=kbid,
        ask_request=ask_request,
        main_results=find_results,
        nuclia_learning_id=None,
        predict_answer_stream=websocket_to_ask(queue, task),
        prequeries_results=None,
        augmented_context=AugmentedContext(),
        search_sdk=rpc.get_sdk("search"),
        debug_chat_model=None,
        best_matches=[],
        metrics=AskMetrics(),
        auditor=ChatAuditor(),
        prompt_context=PromptContext(),
        prompt_context_order=PromptContextOrder(),
    )

    headers = {
        "NUCLIA-LEARNING-ID": ask_result.nuclia_learning_id or "unknown",
        "Access-Control-Expose-Headers": "NUCLIA-LEARNING-ID",
    }
    if x_synchronous:
        return Response(
            content=await ask_result.json(),
            status_code=200,
            headers=headers,
            media_type="application/json",
        )
    else:
        return StreamingResponse(
            content=ask_result.ndjson_stream(),
            status_code=200,
            headers=headers,
            media_type="application/x-ndjson",
        )
