from unittest.mock import AsyncMock

import pytest
from hyperforge_google.config import GoogleAgentConfig
from hyperforge_mcp.config import MCPAgentConfig
from hyperforge_mcp.config_driver import MCPHTTPDriverConfig
from hyperforge_nucliadb_agentic.config import NucliaDBAgentConfig
from hyperforge_nucliadb_agentic.internal_driver import InternalNucliaDBConfig
from hyperforge_perplexity.config import PerplexityAgentConfig
from hyperforge_rephrase.config import RephraseAgentConfig
from hyperforge_smart.config import SmartAgentConfig
from hyperforge_summarize.config import SummarizeAgentConfig

from nucliadb_agentic_api.db.transform import transform_agentic_config
from nucliadb_agentic_api.models import (
    AgenticConfigSchema,
    AgenticRephraseConfiguration,
    AgenticSmartAgentConfiguration,
    AgenticSmartAgentMode,
    AgenticSmartAgentModels,
    AgenticSummarizeConfiguration,
    GoogleSourceSchema,
    MCPSourceSchema,
    NucliaDBSourceSchema,
    PerplexitySourceSchema,
)


@pytest.mark.parametrize("enabled", [True, False])
async def test_transform_propagates_conversation_options(enabled: bool):
    config = AgenticConfigSchema(
        rephrase=AgenticRephraseConfiguration(history=enabled),
        smart_agent=AgenticSmartAgentConfiguration(history=enabled),
        summarize=AgenticSummarizeConfiguration(
            history=enabled, conversational=enabled
        ),
    )

    retrieval_config, _, _ = await transform_agentic_config(
        config,
        AsyncMock(),
        account="account",
        kbid="kbid",
    )

    rephrase = retrieval_config.preprocess[0]
    smart = retrieval_config.context[0]
    summarize = retrieval_config.generation[0]
    assert isinstance(rephrase, RephraseAgentConfig)
    assert isinstance(smart, SmartAgentConfig)
    assert isinstance(summarize, SummarizeAgentConfig)
    assert rephrase.history is enabled
    assert smart.history is enabled
    assert summarize.history is enabled
    assert summarize.conversational is enabled
    assert summarize.citations is True
    assert summarize.force_chunk_level_citations is False


async def test_conversation_options_default_to_enabled():
    config = AgenticConfigSchema(
        rephrase=AgenticRephraseConfiguration(),
        smart_agent=AgenticSmartAgentConfiguration(),
        summarize=AgenticSummarizeConfiguration(),
    )

    assert config.rephrase is not None and config.rephrase.history is True
    assert config.smart_agent is not None and config.smart_agent.history is True
    assert config.summarize is not None and config.summarize.history is True
    assert config.summarize.conversational is True


async def test_history_omission():
    config = AgenticConfigSchema(
        rephrase=AgenticRephraseConfiguration(history=None),
        smart_agent=AgenticSmartAgentConfiguration(history=None),
        summarize=AgenticSummarizeConfiguration(history=None),
    )

    retrieval_config, _, _ = await transform_agentic_config(
        config,
        AsyncMock(),
        account="account",
        kbid="kbid",
    )

    rephrase = retrieval_config.preprocess[0]
    smart = retrieval_config.context[0]
    summarize = retrieval_config.generation[0]
    assert isinstance(rephrase, RephraseAgentConfig)
    assert isinstance(smart, SmartAgentConfig)
    assert isinstance(summarize, SummarizeAgentConfig)
    assert rephrase.history is False
    assert smart.history is False
    assert summarize.history is False


async def test_transform_mcp_source_references_its_driver(load_agents_nucliadb_agentic):
    config = AgenticConfigSchema(
        smart_agent=AgenticSmartAgentConfiguration(sources=["mcp-source"])
    )
    source_manager = AsyncMock()
    source_manager.get_source.return_value = MCPSourceSchema(
        uri="https://example.com/mcp"
    )

    retrieval_config, drivers, _ = await transform_agentic_config(
        config,
        source_manager,
        account="account",
        kbid="kbid",
    )

    smart = retrieval_config.context[0]
    assert isinstance(smart, SmartAgentConfig)
    mcp_config = smart.registered_agents[0]
    assert isinstance(mcp_config, MCPAgentConfig)
    assert mcp_config.source in drivers


async def test_transform_applies_explicit_configuration(load_agents_nucliadb_agentic):
    config = AgenticConfigSchema(
        title="Configured",
        rephrase=AgenticRephraseConfiguration(
            model="rephrase-model", prompt="Rephrase this", ask_to="other-kb"
        ),
        smart_agent=AgenticSmartAgentConfiguration(
            mode=AgenticSmartAgentMode.PLAN_EXECUTE,
            extra_prompt="Use these tools",
            models=AgenticSmartAgentModels(
                context_validation="validation-model",
                planner="planner-model",
                executor="executor-model",
            ),
        ),
        summarize=AgenticSummarizeConfiguration(
            model="summary-model",
            user_prompt="Answer briefly",
            system_prompt="Be precise",
        ),
    )

    retrieval_config, _, _ = await transform_agentic_config(
        config, AsyncMock(), account="account", kbid="kbid"
    )

    rephrase = retrieval_config.preprocess[0]
    smart = retrieval_config.context[0]
    summarize = retrieval_config.generation[0]
    assert isinstance(rephrase, RephraseAgentConfig)
    assert isinstance(smart, SmartAgentConfig)
    assert isinstance(summarize, SummarizeAgentConfig)
    assert (rephrase.title, rephrase.model, rephrase.rules, rephrase.kb) == (
        "Configured - Retrieval",
        "rephrase-model",
        ["Rephrase this"],
        "other-kb",
    )
    assert (
        smart.title,
        smart.planning_mode,
        smart.extra_prompt,
        smart.context_validation_model,
        smart.planner_model,
        smart.executor_model,
    ) == (
        "Configured - Smart Agent",
        "plan_execute",
        "Use these tools",
        "validation-model",
        "planner-model",
        "executor-model",
    )
    assert (
        summarize.title,
        summarize.model,
        summarize.prompt,
        summarize.system_prompt,
    ) == (
        "Configured - Summarize Agent",
        "summary-model",
        "Answer briefly",
        "Be precise",
    )


async def test_transform_uses_runtime_model_defaults():
    config = AgenticConfigSchema(
        rephrase=AgenticRephraseConfiguration(),
        smart_agent=AgenticSmartAgentConfiguration(),
        summarize=AgenticSummarizeConfiguration(),
    )

    retrieval_config, _, _ = await transform_agentic_config(
        config, AsyncMock(), account="account", kbid="kbid"
    )

    rephrase = retrieval_config.preprocess[0]
    smart = retrieval_config.context[0]
    summarize = retrieval_config.generation[0]
    assert isinstance(rephrase, RephraseAgentConfig)
    assert isinstance(smart, SmartAgentConfig)
    assert isinstance(summarize, SummarizeAgentConfig)
    assert rephrase.model == RephraseAgentConfig().model
    assert smart.context_validation_model == SmartAgentConfig().context_validation_model
    assert smart.planner_model == SmartAgentConfig().planner_model
    assert smart.executor_model == SmartAgentConfig().executor_model
    assert summarize.model == SummarizeAgentConfig().model


async def test_transform_wires_sources_and_internal_nucliadb_driver(
    load_agents_nucliadb_agentic,
):
    config = AgenticConfigSchema(
        title="Sources",
        smart_agent=AgenticSmartAgentConfiguration(
            sources=["kb", "mcp", "google", "pplx"]
        ),
    )
    source_manager = AsyncMock()
    source_manager.get_source.side_effect = [
        NucliaDBSourceSchema.model_validate(
            {
                "filter_expression": {
                    "field": {
                        "prop": "resource_mimetype",
                        "type": "application",
                        "subtype": "pdf",
                    }
                },
                "resource_filters": ["resource-id"],
                "search_config": "legal-rag",
            }
        ),
        MCPSourceSchema(
            uri="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
            tool_choice_model="tool-model",
            valid_headers=["X-Request-ID"],
        ),
        GoogleSourceSchema(),
        PerplexitySourceSchema(enabled_domains=["example.com"]),
    ]

    retrieval_config, drivers, global_drivers = await transform_agentic_config(
        config,
        source_manager,
        account="account",
        kbid="kbid",
        internal_nucliadb_url="http://{component}.internal",
    )

    smart = retrieval_config.context[0]
    assert isinstance(smart, SmartAgentConfig)
    kb_agent, mcp_agent, google_agent, perplexity_agent = smart.registered_agents
    assert isinstance(kb_agent, NucliaDBAgentConfig)
    assert isinstance(mcp_agent, MCPAgentConfig)
    assert isinstance(google_agent, GoogleAgentConfig)
    assert isinstance(perplexity_agent, PerplexityAgentConfig)
    assert kb_agent.sources[0] in drivers
    assert kb_agent.search_config == "legal-rag"
    assert mcp_agent.source in drivers
    kb_driver = drivers[kb_agent.sources[0]]
    mcp_driver = drivers[mcp_agent.source]
    assert isinstance(kb_driver, InternalNucliaDBConfig)
    assert isinstance(mcp_driver, MCPHTTPDriverConfig)
    assert kb_driver.config.url == "http://{component}.internal"
    assert kb_driver.config.kbid == "kbid"
    assert kb_driver.config.filters == ["resource-id"]
    assert kb_driver.config.filter_expression is not None
    assert mcp_driver.config.uri == "https://example.com/mcp"
    assert mcp_driver.config.headers == {"Authorization": "Bearer token"}
    assert mcp_agent.tool_choice_model == "tool-model"
    assert mcp_agent.valid_headers == ["X-Request-ID"]
    assert perplexity_agent.domain == ["example.com"]
    assert global_drivers == ["google", "perplexity"]


async def test_transform_uses_external_nucliadb_driver(load_agents_nucliadb_agentic):
    config = AgenticConfigSchema(
        smart_agent=AgenticSmartAgentConfiguration(sources=["kb"])
    )
    source_manager = AsyncMock()
    source_manager.get_source.return_value = NucliaDBSourceSchema()

    retrieval_config, drivers, _ = await transform_agentic_config(
        config,
        source_manager,
        account="account",
        kbid="kbid",
        internal_nucliadb=False,
        external_nucliadb_url="https://external.example.com",
        external_nucliadb_key="secret",
    )

    smart = retrieval_config.context[0]
    assert isinstance(smart, SmartAgentConfig)
    kb_agent = smart.registered_agents[0]
    assert isinstance(kb_agent, NucliaDBAgentConfig)
    driver = drivers[kb_agent.sources[0]]
    assert driver.provider == "nucliadb"
    assert driver.config.url == "https://external.example.com"
    assert driver.config.key == "secret"


async def test_transform_rejects_nucliadb_source_without_configured_url():
    config = AgenticConfigSchema(
        smart_agent=AgenticSmartAgentConfiguration(sources=["kb"])
    )
    source_manager = AsyncMock()
    source_manager.get_source.return_value = NucliaDBSourceSchema()

    with pytest.raises(ValueError, match="No NucliaDB URL"):
        await transform_agentic_config(
            config, source_manager, account="account", kbid="kbid"
        )


def test_history_can_be_disabled_or_unsupported():
    disabled = AgenticConfigSchema(
        rephrase=AgenticRephraseConfiguration(history=False),
        smart_agent=AgenticSmartAgentConfiguration(history=False),
        summarize=AgenticSummarizeConfiguration(history=False),
    )
    unsupported = AgenticConfigSchema(
        rephrase=AgenticRephraseConfiguration(history=None),
        smart_agent=AgenticSmartAgentConfiguration(history=None),
        summarize=AgenticSummarizeConfiguration(history=None),
    )

    assert disabled.rephrase is not None and disabled.rephrase.history is False
    assert disabled.smart_agent is not None and disabled.smart_agent.history is False
    assert disabled.summarize is not None and disabled.summarize.history is False
    assert unsupported.rephrase is not None and unsupported.rephrase.history is None
    assert (
        unsupported.smart_agent is not None and unsupported.smart_agent.history is None
    )
    assert unsupported.summarize is not None and unsupported.summarize.history is None
