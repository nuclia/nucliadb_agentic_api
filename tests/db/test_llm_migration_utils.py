import sqlalchemy as sa
from hyperforge.llm_config import LLMConfig

from nucliadb_agentic_api.db.agentic_configs import (
    CACHE as AGENTIC_CACHE,
    AgenticConfigs,
)
from nucliadb_agentic_api.db.llm_migration_utils import (
    migrate_llm_models,
    walk_and_replace,
)
from nucliadb_agentic_api.db.sources import CACHE as SOURCE_CACHE, Sources
from nucliadb_agentic_api.models import (
    AgenticConfigSchema,
    AgenticSmartAgentConfiguration,
    AgenticSmartAgentModels,
    AgenticSummarizeConfiguration,
    MCPSourceSchema,
)


def test_walk_and_replace_only_updates_plain_llm_config_models():
    config = {
        "planner": {"_type": "llm_config", "model_id": "claude-4-5-sonnet"},
        "steps": [
            {"_type": "llm_config", "model_id": "chatgpt-azure-4o-mini"},
            {
                "_type": "llm_config",
                "model_id": "chatgpt-azure-4o/custom-id",
            },
            {"model_id": "claude-4-5-sonnet"},
        ],
    }

    assert walk_and_replace(
        config,
        {
            "claude-4-5-sonnet": "claude-5-sonnet",
            "chatgpt-azure-4o": "chatgpt-azure-5.6-terra",
            "chatgpt-azure-4o-mini": "chatgpt-azure-5.6-luna",
        },
    )
    assert config["planner"]["model_id"] == "claude-5-sonnet"
    assert config["steps"][0]["model_id"] == "chatgpt-azure-5.6-luna"
    assert config["steps"][1]["model_id"] == "chatgpt-azure-4o/custom-id"
    assert config["steps"][2]["model_id"] == "claude-4-5-sonnet"


async def test_migrate_llm_models_updates_stored_agent_and_source_configs(
    agentic_pg_dsn, nucliadb_agentic_data_manager_settings
):
    agent_manager = await AgenticConfigs.from_settings(
        nucliadb_agentic_data_manager_settings
    )
    source_manager = await Sources.from_settings(nucliadb_agentic_data_manager_settings)
    await agent_manager.initialize()
    await source_manager.initialize()
    await agent_manager.create_agentic_config(
        "account",
        "kbid",
        "agent-id",
        AgenticConfigSchema(
            title="Agent",
            smart_agent=AgenticSmartAgentConfiguration(
                models=AgenticSmartAgentModels(
                    context_validation=LLMConfig(model_id="claude-4-5-sonnet"),
                    planner=LLMConfig(model_id="chatgpt-azure-4o/custom-id"),
                    executor=LLMConfig(model_id="chatgpt-azure-4o-mini"),
                )
            ),
            summarize=AgenticSummarizeConfiguration(
                model=LLMConfig(model_id="chatgpt-azure-4o")
            ),
        ),
    )
    await source_manager.create_source(
        "account",
        "kbid",
        "source-id",
        MCPSourceSchema(
            uri="https://mcp.example.com",
            tool_choice_model=LLMConfig(model_id="chatgpt-azure-4o-mini"),
        ),
    )

    engine = sa.create_engine(agentic_pg_dsn)
    with engine.connect() as conn:
        migrate_llm_models(
            conn,
            {
                "claude-4-5-sonnet": "claude-5-sonnet",
                "chatgpt-azure-4o": "chatgpt-azure-5.6-terra",
                "chatgpt-azure-4o-mini": "chatgpt-azure-5.6-luna",
            },
        )
        conn.commit()

    AGENTIC_CACHE.clear()
    SOURCE_CACHE.clear()
    stored_agent = await agent_manager.get_agentic_config("account", "kbid", "agent-id")
    stored_source = await source_manager.get_source("account", "kbid", "source-id")

    assert stored_agent.smart_agent is not None
    assert stored_agent.smart_agent.models is not None
    assert stored_agent.smart_agent.models.context_validation is not None
    assert (
        stored_agent.smart_agent.models.context_validation.model_id == "claude-5-sonnet"
    )
    assert stored_agent.smart_agent.models.planner is not None
    assert stored_agent.smart_agent.models.planner.model_id == (
        "chatgpt-azure-4o/custom-id"
    )
    assert stored_agent.smart_agent.models.executor is not None
    assert stored_agent.smart_agent.models.executor.model_id == "chatgpt-azure-5.6-luna"
    assert stored_agent.summarize is not None
    assert stored_agent.summarize.model is not None
    assert stored_agent.summarize.model.model_id == "chatgpt-azure-5.6-terra"
    assert isinstance(stored_source, MCPSourceSchema)
    assert stored_source.tool_choice_model is not None
    assert stored_source.tool_choice_model.model_id == "chatgpt-azure-5.6-luna"

    await source_manager.finalize()
    await agent_manager.finalize()
