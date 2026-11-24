from contextvars import ContextVar

from httpx import Request
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from hyperforge_nucliadb_agentic.ask import logger

_account_id_context = ContextVar[str | None]("account_id", default=None)


def get_account_id() -> str | None:
    return _account_id_context.get()


def _require_account_id() -> str:
    account_id = get_account_id()
    if not account_id:
        error = "Missing authenticated account ID for internal service request"
        logger.warning(error)
        raise ValueError(error)
    return account_id


async def add_predict_account_header(request: Request) -> None:
    request.headers["X-STF-ACCOUNT"] = _require_account_id()


async def add_nucliadb_account_header(request: Request) -> None:
    request.headers["X-NUCLIADB-ACCOUNT"] = _require_account_id()


class AccountContextMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        account_id = headers.get("x-nucliadb-account") or headers.get("x-stf-account")
        token = _account_id_context.set(account_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _account_id_context.reset(token)
