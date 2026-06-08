import logging

from fastapi import Query
from starlette.requests import Request
from starlette.responses import HTMLResponse

from nucliadb_agentic_api.settings import Settings
from nucliadb_agentic_api.v1.router import router
from nucliadb_agentic_api.v1.utils import tracer

logger = logging.getLogger(__name__)

RENDER = "<html><body><h1>OAuth Completed</h1><p>You can close this window and return to the application.</p></body></html>"


@router.get(
    "/api/auth/kb/{kbid}/workflow/{workflow_id}/session/{session}/oauth/{oauth_uuid}/callback",
    status_code=200,
    description="OAuth callback endpoint for retrieval agent workflows. This endpoint is called by the RAO after the user completes the OAuth flow, and it sends the obtained credentials to the corresponding websocket.",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
async def oauth_callback(
    request: Request,
    kbid: str,
    session: str,
    workflow_id: str,
    oauth_uuid: str,
    question_id: str = Query(..., include_in_schema=False),
    state: str = Query(..., include_in_schema=False),
    account_id: str = Query(..., include_in_schema=False),
):
    """
    Callback from oauth flow on RAO that requires to send creds to websocket
    """
    settings: Settings = request.app.settings
    subject = settings.oauth_subject.format(
        account=account_id,
        agent_id=kbid,
        session=session,
        question=question_id,
        oauth_uuid=oauth_uuid,
        workflow_id=workflow_id,
    )
    # Request a question
    with tracer().start_as_current_span("Request activation"):
        logger.info(
            "OAuth callback received for agent=%s, session=%s, oauth_uuid=%s, question_id=%s",
            kbid,
            session,
            oauth_uuid,
            question_id,
        )
        await request.app.broker.send_reply(subject, state)
        logger.info(
            "OAuth callback published to stream %s",
            subject,
        )

    return HTMLResponse(content=RENDER)
