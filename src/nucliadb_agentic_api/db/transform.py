from typing import Any, Dict, Tuple
from uuid import uuid4

from hyperforge.driver import DriverConfig
from hyperforge.models import MemoryConfig, Rules
from hyperforge.retrieval.config import RetrievalAgentConfig
from hyperforge.workflows import WorkflowData
from hyperforge_mcp.config import Transport
from hyperforge_mcp.config_driver import (
    MCPHTTPDriverConfig,
    MCPHTTPInnerConfig,
)
from hyperforge_nucliadb.driver_config import (
    NucliaDBConfig,
    NucliaDBConnection,
)
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from hyperforge_nucliadb_agentic.internal_driver import (
    InternalNucliaDBConfig,
    InternalNucliaDBConnection,
)

from nucliadb_agentic_api.db.sources import Sources
from nucliadb_agentic_api.models import AgenticConfigSchema


async def transform_agentic_config(
    agentic_config: AgenticConfigSchema,
    source_manager: Sources,
    account: str,
    internal_nucliadb: bool = True,
    internal_nucliadb_url: str | None = None,
    external_nucliadb_url: str | None = None,
    external_nucliadb_key: str | None = None,
    ask_request: AskRequest | None = None,
    kbid: str = "",
) -> Tuple[RetrievalAgentConfig, Dict[str, DriverConfig], list[str]]:
    drivers: Dict[str, DriverConfig] = {}
    global_driver = []

    title = agentic_config.title if agentic_config.title else "Default Agentic Config"

    if agentic_config.rephrase:
        config: Dict[str, Any] = {
            "title": f"{title} - Retrieval",
        }
        if agentic_config.rephrase.model:
            config["model"] = agentic_config.rephrase.model
        if agentic_config.rephrase.prompt:
            config["rules"] = [agentic_config.rephrase.prompt]
        if agentic_config.rephrase.ask_to:
            # Same KB with different layers
            config["kbid"] = agentic_config.rephrase.ask_to

        config["module"] = "rephrase"

        preprocess = [config]
    else:
        preprocess = []

    if agentic_config.smart_agent:
        config = {
            "title": f"{title} - Smart Agent",
        }
        if agentic_config.smart_agent.extra_prompt:
            config["extra_prompt"] = agentic_config.smart_agent.extra_prompt
        if agentic_config.smart_agent.models:
            if agentic_config.smart_agent.models.context_validation:
                config["context_validation_model"] = (
                    agentic_config.smart_agent.models.context_validation
                )
            if agentic_config.smart_agent.models.planner:
                config["planner_model"] = agentic_config.smart_agent.models.planner
            if agentic_config.smart_agent.models.executor:
                config["executor_model"] = agentic_config.smart_agent.models.executor
        registered_agents = []
        for source in agentic_config.smart_agent.sources:
            source_obj = await source_manager.get_source(account, kbid, source)
            source_config: Dict[str, Any] = {
                "title": f"{title} - Smart Agent - {source_obj.type.capitalize()} Source",
            }

            if source_obj.type == "nucliadb":
                # Same KB different
                uid = uuid4().hex

                source_config["sources"] = [uid]
                source_config["module"] = "nucliadb_agent"

                if internal_nucliadb and internal_nucliadb_url:
                    ndb_driver_config = InternalNucliaDBConfig(
                        identifier=uid,
                        name="NucliaDB",
                        provider="nucliadb_internal",
                        config=InternalNucliaDBConnection(
                            url="internal",  # Placeholder, actual URL is handled by the internal driver
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
                registered_agents.append(source_config)
            elif source_obj.type == "mcp":
                uid = uuid4().hex
                source_config["sources"] = [uid]
                source_config["transport"] = Transport.HTTP
                source_config["module"] = "mcp"
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
                registered_agents.append(source_config)

            elif source_obj.type == "google":
                source_config["module"] = "google"
                global_driver.append("google")
                registered_agents.append(source_config)

            elif source_obj.type == "perplexity":
                source_config["module"] = "perplexity"
                global_driver.append("perplexity")
                registered_agents.append(source_config)

        config["registered_agents"] = registered_agents
        config["module"] = "smart"
        context = [config]
    else:
        context = []

    if agentic_config.summarize:
        config = {
            "title": f"{title} - Summarize Agent",
        }
        if agentic_config.summarize.model:
            config["model"] = agentic_config.summarize.model
        if agentic_config.summarize.user_prompt:
            config["user_prompt"] = agentic_config.summarize.user_prompt
        if agentic_config.summarize.system_prompt:
            config["system_prompt"] = agentic_config.summarize.system_prompt
        if agentic_config.summarize.conversational:
            config["conversational"] = agentic_config.summarize.conversational

        config["module"] = "summarize"

        generation = [config]
    else:
        generation = []

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
            preprocess=preprocess,  # type: ignore
            context=context,  # type: ignore
            generation=generation,  # type: ignore
            postprocess=[],
        ),
        drivers,
        global_driver,
    )
