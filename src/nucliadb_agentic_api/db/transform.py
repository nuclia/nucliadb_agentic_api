from typing import Dict, Tuple
from uuid import uuid4

from hyperforge.agent import AgentConfig
from hyperforge.context.config import ContextAgentConfig
from hyperforge.driver import DriverConfig
from hyperforge.models import MemoryConfig, Rules
from hyperforge.retrieval.config import RetrievalAgentConfig
from hyperforge.workflows import WorkflowData
from hyperforge_google.config import GoogleAgentConfig
from hyperforge_mcp.config import MCPAgentConfig, Transport
from hyperforge_mcp.config_driver import (
    MCPHTTPDriverConfig,
    MCPHTTPInnerConfig,
)
from hyperforge_nucliadb.driver_config import (
    NucliaDBConfig,
    NucliaDBConnection,
)
from hyperforge_nucliadb_agentic.config import NucliaDBAgentConfig
from hyperforge_nucliadb_agentic.internal_driver import (
    InternalNucliaDBConfig,
    InternalNucliaDBConnection,
)
from hyperforge_perplexity.config import PerplexityAgentConfig
from hyperforge_rephrase.config import RephraseAgentConfig
from hyperforge_smart.config import SmartAgentConfig
from hyperforge_summarize.config import SummarizeAgentConfig

from nucliadb_agentic_api.db.sources import Sources
from nucliadb_agentic_api.models import AgenticConfigSchema

# Kept for the dormant AskRequest configuration path below.
# from hyperforge_nucliadb_agentic.ask.model import AskRequest


async def transform_agentic_config(
    agentic_config: AgenticConfigSchema,
    source_manager: Sources,
    account: str,
    internal_nucliadb: bool = True,
    internal_nucliadb_url: str | None = None,
    external_nucliadb_url: str | None = None,
    external_nucliadb_key: str | None = None,
    # The runtime AskRequest is consumed from QuestionMemory by NucliaDBAgent.
    # Keep this dormant configuration parameter visible until its intended use is known.
    # ask_request: AskRequest | None = None,
    kbid: str = "",
) -> Tuple[RetrievalAgentConfig, Dict[str, DriverConfig], list[str]]:
    drivers: Dict[str, DriverConfig] = {}
    global_driver = []

    title = agentic_config.title if agentic_config.title else "Default Agentic Config"
    preprocess: list[AgentConfig] = []
    context: list[AgentConfig] = []
    generation: list[AgentConfig] = []

    if agentic_config.rephrase:
        rephrase = agentic_config.rephrase
        rephrase_config = RephraseAgentConfig(
            title=f"{title} - Retrieval",
            rules=[rephrase.prompt] if rephrase.prompt else None,
            kb=rephrase.ask_to,
            history=rephrase.history if rephrase.history is not None else False,
        )
        if rephrase.model:
            rephrase_config.model = rephrase.model
        preprocess.append(rephrase_config)

    if agentic_config.smart_agent:
        smart_agent = agentic_config.smart_agent
        registered_agents: list[ContextAgentConfig] = []
        for source in smart_agent.sources:
            source_obj = await source_manager.get_source(account, kbid, source)
            source_title = (
                f"{title} - Smart Agent - {source_obj.type.capitalize()} Source"
            )

            if source_obj.type == "nucliadb":
                # Same KB different
                uid = uuid4().hex

                ndb_driver_config: DriverConfig
                if internal_nucliadb and internal_nucliadb_url:
                    ndb_driver_config = InternalNucliaDBConfig(
                        identifier=uid,
                        name="NucliaDB",
                        provider="nucliadb_internal",
                        config=InternalNucliaDBConnection(
                            url=internal_nucliadb_url,
                            key=None,
                            manager="",
                            description="",
                            kbid=kbid,
                            filter_expression=source_obj.filter_expression,
                            filters=source_obj.resource_filters
                            if source_obj.resource_filters
                            else [],
                        ),
                    )
                elif external_nucliadb_url:
                    ndb_driver_config = NucliaDBConfig(
                        identifier=uid,
                        name="NucliaDB",
                        provider="nucliadb",
                        config=NucliaDBConnection(
                            url=external_nucliadb_url,
                            key=external_nucliadb_key,
                            manager="",
                            description="",
                            kbid=kbid,
                            filter_expression=source_obj.filter_expression,
                            filters=source_obj.resource_filters
                            if source_obj.resource_filters
                            else [],
                        ),
                    )
                else:
                    raise ValueError(
                        "No NucliaDB URL configured for internal or external access"
                    )
                drivers[uid] = ndb_driver_config
                registered_agents.append(
                    NucliaDBAgentConfig(
                        id=f"nucliadb-agent-{source}", title=source_title, sources=[uid]
                    )
                )
            elif source_obj.type == "mcp":
                uid = uuid4().hex
                mcp_driver_config = MCPHTTPDriverConfig(
                    identifier=uid,
                    name="MCPHTTP",
                    provider="mcphttp",
                    config=MCPHTTPInnerConfig(
                        uri=source_obj.uri,  # TODO: pass real URL if needed
                        headers=source_obj.headers if source_obj.headers else {},
                    ),
                )  # TODO: pass real config if needed
                drivers[uid] = mcp_driver_config
                registered_agents.append(
                    MCPAgentConfig(
                        id=f"mcp-agent-{source}",
                        title=source_title,
                        source=uid,
                        transport=Transport.HTTP,
                        tool_choice_model=source_obj.tool_choice_model or "chatgpt-4.1",
                        valid_headers=source_obj.valid_headers or [],
                    )
                )

            elif source_obj.type == "google":
                global_driver.append("google")
                registered_agents.append(
                    GoogleAgentConfig(id=f"google-agent-{source}", title=source_title)
                )

            elif source_obj.type == "perplexity":
                global_driver.append("perplexity")
                registered_agents.append(
                    PerplexityAgentConfig(
                        id=f"perplexity-agent-{source}",
                        title=source_title,
                        domain=source_obj.enabled_domains or [],
                    )
                )

        models = smart_agent.models
        smart_config = SmartAgentConfig(
            title=f"{title} - Smart Agent",
            planning_mode=smart_agent.mode.value,
            extra_prompt=smart_agent.extra_prompt,
            history=smart_agent.history if smart_agent.history is not None else False,
            registered_agents=[agent.model_dump() for agent in registered_agents],
        )
        # Only set the models if they are provided; otherwise, leave them as None to use hyperforge's default models. This allows for flexibility in configuration without enforcing specific model choices.
        # TODO: Provide nucliadb agentic api level defaults for models
        if models:
            if models.context_validation:
                smart_config.context_validation_model = models.context_validation
            if models.planner:
                smart_config.planner_model = models.planner
            if models.executor:
                smart_config.executor_model = models.executor
        context.append(smart_config)

    if agentic_config.summarize:
        summarize = agentic_config.summarize
        summarize_config = SummarizeAgentConfig(
            title=f"{title} - Summarize Agent",
            prompt=summarize.user_prompt,
            system_prompt=summarize.system_prompt,
            conversational=summarize.conversational,
            citations=True,
            force_chunk_level_citations=False,
            history=summarize.history if summarize.history is not None else False,
        )
        if summarize.model:
            summarize_config.model = summarize.model
        generation.append(summarize_config)

    return (
        RetrievalAgentConfig(
            drivers=[],
            rules=Rules(),
            memory=MemoryConfig(),
            workflow=WorkflowData(
                id="default",
                name="Default Workflow",
                description="Default workflow for agentic config transformation",
                parameters=None,
            ),
            preprocess=preprocess,
            context=context,
            generation=generation,
            postprocess=[],
        ),
        drivers,
        global_driver,
    )
