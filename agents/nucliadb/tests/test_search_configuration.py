from unittest.mock import AsyncMock, patch

import pytest
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from hyperforge_nucliadb_agentic.ask.search import rpc
from nucliadb_models.configuration import AskConfig


async def test_ask_request_overrides_search_configuration():
    search_config = AsyncMock()
    search_config.config = AskConfig(top_k=1)

    with patch.object(
        rpc,
        "get_search_configuration",
        new=AsyncMock(return_value=search_config),
    ):
        request = await rpc.apply_ask_search_configuration(
            AsyncMock(),
            "kbid",
            AskRequest(
                query="question",
                search_configuration="compact-rag",
                top_k=7,
            ),
        )

    assert request.top_k == 7


async def test_search_configuration_supplies_unspecified_ask_parameters():
    search_config = AsyncMock()
    search_config.config = AskConfig(top_k=1)

    with patch.object(
        rpc,
        "get_search_configuration",
        new=AsyncMock(return_value=search_config),
    ):
        request = await rpc.apply_ask_search_configuration(
            AsyncMock(),
            "kbid",
            AskRequest(query="question", search_configuration="compact-rag"),
        )

    assert request.top_k == 1


@pytest.mark.parametrize(
    ("search_config", "expected_error"),
    [
        (None, rpc.SearchConfigurationNotFound),
        (AsyncMock(config=object()), rpc.InvalidAskSearchConfiguration),
    ],
)
async def test_rejects_unusable_search_configuration(
    search_config, expected_error
):
    with (
        patch.object(
            rpc,
            "get_search_configuration",
            new=AsyncMock(return_value=search_config),
        ),
        pytest.raises(expected_error),
    ):
        await rpc.apply_ask_search_configuration(
            AsyncMock(),
            "kbid",
            AskRequest(query="question", search_configuration="invalid"),
        )