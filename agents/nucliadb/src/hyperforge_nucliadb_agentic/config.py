from typing import List, Literal, Optional, Tuple

from hyperforge.context.agent import ContextAgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class NucliaDBAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Knowledge Box Agent")
    module: Literal["nucliadb_agent"] = "nucliadb_agent"
    search_config: Optional[str] = Field(
        default=None,
        description="ID of an ask search configuration applied to this agent",
    )
    sources: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "show_in_node": True,
        },
    )
    generative_model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
        description="Model used to generate answers",
    )
    published_functions: Optional[Tuple[str, ...]] = Field(
        default=(
            "search_by_title",
            "ask_labels_list",
            "ask_by_title",
            "ask_agent",
            "ask_labels",
            "facets_count",
            "facets_search",
            "catalog_search",
            "all_images_by_title",
            "search_images",
        ),
        title="Published functions",
        description="List of functions published by this agent to be used by other agents in the chain",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    generate_inner_answer: bool = Field(
        default=True,
        title="Generate inner answer",
        description="If true, the agent will generate an inner answer to be used by other agents in the chain. If false, the agent will only generate a reasoning and a list of resources.",
    )
