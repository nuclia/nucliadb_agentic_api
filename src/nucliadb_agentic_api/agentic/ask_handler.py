from typing import TYPE_CHECKING

from fastapi import Response
from hyperforge_nucliadb_agentic.ask.model import (
    AskRequest,
    parse_max_tokens,
)
from hyperforge_nucliadb_agentic.ask.search.ask import (
    handled_ask_exceptions,
)
from nucliadb_models.search import (
    NucliaDBClientType,
)
from starlette.responses import StreamingResponse

from nucliadb_agentic_api.agentic.ask_result import AgenticAskResult

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
    ask_result = AgenticAskResult(
        app=app,
        kbid=kbid,
        ask_request=ask_request,
        agentic_config_id=agentic_config_id,
        account=account,
        origin=origin,
        resource=resource,
    )

    nuclia_learning_id = await ask_result.start()

    headers = {
        "NUCLIA-LEARNING-ID": nuclia_learning_id or "unknown",
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
