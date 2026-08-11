from unittest.mock import AsyncMock, patch

import pytest

from nucliadb_agentic_api.db.transform import transform_agentic_config
from nucliadb_agentic_api.models import AgenticConfigSchema


@pytest.mark.parametrize("enabled", [True, False])
async def test_transform_propagates_conversation_options(enabled: bool):
    config = AgenticConfigSchema.model_validate(
        {
            "rephrase": {"history": enabled},
            "smart_agent": {"history": enabled},
            "summarize": {
                "history": enabled,
                "conversational": enabled,
            },
        }
    )

    with patch(
        "nucliadb_agentic_api.db.transform.RetrievalAgentConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        retrieval_config, _, _ = await transform_agentic_config(
            config,
            AsyncMock(),
            account="account",
            kbid="kbid",
        )

    assert retrieval_config["preprocess"][0]["history"] is enabled
    assert retrieval_config["context"][0]["history"] is enabled
    assert retrieval_config["generation"][0]["history"] is enabled
    assert retrieval_config["generation"][0]["conversational"] is enabled
    assert retrieval_config["generation"][0]["citations"] is True
    assert retrieval_config["generation"][0]["force_chunk_level_citations"] is False


async def test_conversation_options_default_to_enabled():
    config = AgenticConfigSchema.model_validate(
        {
            "rephrase": {},
            "smart_agent": {},
            "summarize": {},
        }
    )

    assert config.rephrase is not None and config.rephrase.history is True
    assert config.smart_agent is not None and config.smart_agent.history is True
    assert config.summarize is not None and config.summarize.history is True
    assert config.summarize.conversational is True


async def test_history_omission():
    config = AgenticConfigSchema.model_validate(
        {
            "rephrase": {"history": None},
            "smart_agent": {"history": None},
            "summarize": {"history": None},
        }
    )

    with patch(
        "nucliadb_agentic_api.db.transform.RetrievalAgentConfig",
        side_effect=lambda **kwargs: kwargs,
    ):
        retrieval_config, _, _ = await transform_agentic_config(
            config,
            AsyncMock(),
            account="account",
            kbid="kbid",
        )

    assert "history" not in retrieval_config["preprocess"][0]
    assert "history" not in retrieval_config["context"][0]
    assert "history" not in retrieval_config["generation"][0]


def test_history_can_be_disabled_or_unsupported():
    disabled = AgenticConfigSchema.model_validate(
        {
            "rephrase": {"history": False},
            "smart_agent": {"history": False},
            "summarize": {"history": False},
        }
    )
    unsupported = AgenticConfigSchema.model_validate(
        {
            "rephrase": {"history": None},
            "smart_agent": {"history": None},
            "summarize": {"history": None},
        }
    )

    assert disabled.rephrase is not None and disabled.rephrase.history is False
    assert disabled.smart_agent is not None and disabled.smart_agent.history is False
    assert disabled.summarize is not None and disabled.summarize.history is False
    assert unsupported.rephrase is not None and unsupported.rephrase.history is None
    assert unsupported.smart_agent is not None and unsupported.smart_agent.history is None
    assert unsupported.summarize is not None and unsupported.summarize.history is None