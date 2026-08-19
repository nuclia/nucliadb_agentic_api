import asyncio
import json

from fastapi import Header, Query, WebSocket, WebSocketDisconnect
from hyperforge.api.authentication import requires
from hyperforge.api.v1.interaction import WebsocketReceiver, stream_response
from hyperforge.interaction import AnswerOperation, AragAnswer, ARAGException
from hyperforge_nucliadb_agentic.ask.audit import get_audit, get_trace_id
from hyperforge_nucliadb_agentic.ask.model import (
    AskRequest,
)
from hyperforge_nucliadb_agentic.ask.search import rpc
from hyperforge_nucliadb_agentic.ask.utils.responses import (
    HTTPClientError,
)
from nucliadb_models.configuration import AskConfig
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_models.search import (
    NucliaDBClientType,
)
from nucliadb_models.security import RequestSecurity
from nucliadb_utils.authentication import NucliaUser
from pydantic import ValidationError

from nucliadb_agentic_api import logger
from nucliadb_agentic_api.v1.router import router


@router.websocket("/api/v1/kb/{kbid}/ask")
@requires(NucliaDBRoles.READER)
async def websocket_endpoint(
    websocket: WebSocket,
    kbid: str,
    agentic_config_id: str = Query(
        ..., description="ID of the agentic configuration to use for this session"
    ),
    search_configuration: str | None = Query(
        default=None,
        description="Optional search configuration to use for this session",
    ),
    groups: list[str] = Query(
        default=[],
        description="List of group ids to do the request with. Overrides any security group from the Authorization header.",
    ),
    keep_open: bool = Query(
        default=False,
        description="Whether to keep the websocket open after the first question",
    ),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
    x_ndb_client: NucliaDBClientType = Header(NucliaDBClientType.API),
    x_show_consumption: bool = Header(default=False),
    x_nucliadb_user: str = Header(""),
    x_forwarded_for: str = Header(""),
):
    await websocket.accept()
    receiver = WebsocketReceiver(websocket)
    task = asyncio.create_task(receiver.run())

    current_user: NucliaUser = websocket.user
    # If present, security groups from AuthorizationBackend overrides any
    # security group of the payload
    if current_user.security_groups:
        security = RequestSecurity(groups=current_user.security_groups)
    else:
        security = RequestSecurity(groups=groups)

    item = AskRequest(query="")
    item.security = security

    if search_configuration is not None:
        search_config = await rpc.get_search_configuration(
            rpc.get_sdk("reader"), kbid, name=search_configuration
        )
        if search_config is None:
            return HTTPClientError(
                status_code=400, detail="Search configuration not found"
            )

        if not isinstance(search_config.config, AskConfig):
            return HTTPClientError(
                status_code=400,
                detail="This search configuration is not valid for `ask`",
            )

        try:
            item = AskRequest.model_validate(
                search_config.config.model_dump(exclude_unset=True)
                | item.model_dump(exclude_unset=True)
            )
        except ValidationError as e:
            detail = json.loads(e.json())
            return HTTPClientError(status_code=422, detail=detail)

    # Wait for questions
    first_question = True
    while True:
        if not keep_open and not first_question:
            break
        try:
            interaction = await receiver.receive_question()
        except WebSocketDisconnect:
            break
        except ValueError as e:
            # Wrong message type received
            await websocket.send_json(
                AragAnswer(
                    exception=ARAGException(detail=f"Unexpected message: {str(e)}"),
                    operation=AnswerOperation.ERROR,
                ).model_dump()
            )
            break

        first_question = False
        for header, header_value in websocket.headers.items():
            interaction.headers[header] = header_value

        item.query = interaction.question
        interaction.arguments["ask_request"] = item.model_dump_json()

        async for msg in stream_response(
            websocket.app,
            receiver,
            x_stf_account,
            kbid,
            "ephemeral",
            interaction,
            workflow_id=agentic_config_id,
        ):
            if msg.step and msg.step.external_usage:
                audit = get_audit()
                if audit is not None:
                    audit.report_step_usage(
                        account_id=x_stf_account,
                        kbid=kbid,
                        client_type=x_ndb_client,
                        step=msg.step,
                        trace_id=get_trace_id(),
                    )
                else:
                    logger.warning(
                        "Skipping WebSocket external usage report because audit utility is unavailable account=%s kb=%s module=%s",
                        x_stf_account,
                        kbid,
                        msg.step.module,
                    )
            try:
                await websocket.send_text(msg.model_dump_json())
            except (RuntimeError, WebSocketDisconnect):
                # WebSocket already closed
                pass

    try:
        task.cancel()
        await websocket.close()
    except RuntimeError:
        # WebSocket already closed
        pass
