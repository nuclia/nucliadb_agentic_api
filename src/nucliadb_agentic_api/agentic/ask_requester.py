import asyncio
from typing import TYPE_CHECKING

from hyperforge.api.v1.interaction import stream_response
from hyperforge.interaction import AragAnswer

from nucliadb_agentic_api.agentic.ask_transform_to_interaction import (
    interaction_from_ask_request,
)
from nucliadb_agentic_api.ask.model import AskRequest

if TYPE_CHECKING:
    from nucliadb_agentic_api.app import HTTPApplication


async def interaction(
    ask_request: AskRequest,
    queue: asyncio.Queue[AragAnswer],
    app: "HTTPApplication",
    account: str,
    kbid: str,
    agentic_config_id: str,
    origin: str,
):

    interaction = interaction_from_ask_request(ask_request)
    async for msg in stream_response(
        app,  # type: ignore
        None,
        account,
        kbid,
        "ephemeral",
        interaction,
        workflow_id=agentic_config_id,
    ):
        try:
            await queue.put(msg)
        except (RuntimeError, asyncio.QueueFull):
            # WebSocket already closed
            pass
