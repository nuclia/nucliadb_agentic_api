from collections.abc import Mapping
from typing import Any

from hyperforge.manager import Manager

ALLOW_ACCESS_EXTERNAL_MODELS_HEADER = "x-allow-access-external-models"

# Request headers that must be propagated to every learning (predict) call
LEARNING_FORWARDED_HEADERS = (ALLOW_ACCESS_EXTERNAL_MODELS_HEADER,)


def get_learning_forwarded_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Pick, case-insensitively, the headers to forward to learning. Headers
    that are not present in the request are omitted."""
    lowered = {key.lower(): value for key, value in headers.items()}
    return {
        name: lowered[name] for name in LEARNING_FORWARDED_HEADERS if name in lowered
    }


class HeaderForwardingManager(Manager):
    """Wraps a Manager so every learning call carries a set of extra headers.
    Headers explicitly passed on a call take precedence."""

    def __init__(self, manager: Manager, headers: Mapping[str, str]) -> None:
        self._manager = manager
        self._headers = dict(headers)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def _merge(self, extra_headers: dict[str, str] | None) -> dict[str, str]:
        return {**self._headers, **(extra_headers or {})}

    async def query_predict(self, request, *, extra_headers=None, **kwargs):  # type: ignore[override]
        return await self._manager.query_predict(
            request, extra_headers=self._merge(extra_headers), **kwargs
        )

    async def rephrase(self, request, *, extra_headers=None, **kwargs):  # type: ignore[override]
        return await self._manager.rephrase(
            request, extra_headers=self._merge(extra_headers), **kwargs
        )

    async def generate_stream(self, body, *, extra_headers=None, **kwargs):  # type: ignore[override]
        return await self._manager.generate_stream(
            body, extra_headers=self._merge(extra_headers), **kwargs
        )

    async def rerank(self, model, *, extra_headers=None, **kwargs):  # type: ignore[override]
        return await self._manager.rerank(
            model, extra_headers=self._merge(extra_headers), **kwargs
        )

    async def tokens_predict(self, text, model=None, extra_headers=None, **kwargs):  # type: ignore[override]
        return await self._manager.tokens_predict(
            text, model=model, extra_headers=self._merge(extra_headers), **kwargs
        )


def with_forwarded_headers(
    manager: Manager, headers: Mapping[str, str] | None
) -> Manager:
    if not headers:
        return manager
    return HeaderForwardingManager(manager, headers)
