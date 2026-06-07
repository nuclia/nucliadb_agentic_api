from typing import Annotated, Any, Literal, Self

from nucliadb_models import (
    DateTime,
    FieldTypeName,
    FilterExpression,
    RequestSecurity,
    SearchParamDefaults,
)
from nucliadb_models.resource import ExtractedDataTypeName
from nucliadb_models.search import (
    ANSWER_JSON_SCHEMA_EXAMPLE,
    AuditMetadataBase,
    ChatOptions,
    Filter,
    FindRequest,
    Image,
    KnowledgeboxFindResults,
    MinScore,
    RankFusion,
    RankFusionName,
    Relations,
    ResourceProperties,
    TextPosition,
    _validate_resource_filter,
)
from enum import Enum
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema


class RerankerName(str, Enum):
    """Rerankers

    - Predict reranker: after retrieval, send the results to Predict API to
      rerank it. This method uses a reranker model, so one can expect better
      results at the expense of more latency.

      This will be the new default

    - No-operation (noop) reranker: maintain order and do not rerank the results
      after retrieval

    """

    PREDICT_RERANKER = "predict"
    NOOP = "noop"


class _BaseReranker(BaseModel):
    name: str

    @model_validator(mode="after")
    def set_discriminator(self) -> Self:
        # Ensure discriminator is explicitly set so it's always serialized
        self.name = self.name
        return self


class PredictReranker(_BaseReranker):
    name: Literal[RerankerName.PREDICT_RERANKER] = RerankerName.PREDICT_RERANKER
    window: int | None = Field(
        default=None,
        le=200,
        title="Reranker window",
        description="Number of elements reranker will use. Window must be greater or equal to top_k. Greater values will improve results at cost of retrieval and reranking time. By default, this reranker uses a default of 2 times top_k",
    )


Reranker = Annotated[PredictReranker, Field(discriminator="name")]


class Author(str, Enum):
    NUCLIA = "NUCLIA"
    USER = "USER"


class ChatContextMessage(BaseModel):
    author: Author
    text: str


# For bw compatibility
Message = ChatContextMessage


class UserPrompt(BaseModel):
    prompt: str


class MaxTokens(BaseModel):
    context: int | None = Field(
        default=None,
        title="Maximum context tokens",
        description="Use to limit the amount of tokens used in the LLM context",
    )
    answer: int | None = Field(
        default=None,
        title="Maximum answer tokens",
        description="Use to limit the amount of tokens used in the LLM answer",
    )


def parse_max_tokens(max_tokens: int | MaxTokens | None) -> MaxTokens | None:
    if isinstance(max_tokens, int):
        # If the max_tokens is an integer, it is interpreted as the max_tokens value for the generated answer.
        # The max tokens for the context is set to None to use the default value for the model (comes in the
        # NUA's query endpoint response).
        return MaxTokens(answer=max_tokens, context=None)
    return max_tokens


class Reasoning(BaseModel):
    display: bool = Field(
        default=True,
        description="Whether to display the reasoning steps in the response.",
    )
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="medium",
        description=(
            "Level of reasoning effort. Used by OpenAI models to control the depth of reasoning. "
            "This parameter will be automatically mapped to budget_tokens "
            "if the chosen model does not support effort."
        ),
    )
    budget_tokens: int = Field(
        default=15_000,
        description=(
            "Token budget for reasoning. Used by Anthropic or Google models to limit the number of "
            "tokens used for reasoning. This parameter will be automatically mapped to effort "
            "if the chosen model does not support budget_tokens."
        ),
    )


class CitationsType(str, Enum):
    NONE = "none"
    DEFAULT = "default"
    LLM_FOOTNOTES = "llm_footnotes"


class ChatModel(BaseModel):
    """
    This is the model for the predict request payload on the chat endpoint
    """

    question: str = Field(description="Question to ask the generative model")
    user_id: str
    retrieval: bool = True
    system: str | None = Field(
        default=None,
        title="System prompt",
        description="Optional system prompt input by the user",
    )
    query_context: dict[str, str] = Field(
        default={},
        description="The information retrieval context for the current query",
    )
    query_context_order: dict[str, int] | None = Field(
        default=None,
        description="The order of the query context elements. This is used to sort the context elements by relevance before sending them to the generative model",
    )
    chat_history: list[ChatContextMessage] = Field(
        default=[], description="The chat conversation history"
    )
    truncate: bool = Field(
        default=True,
        description="Truncate the chat context in case it doesn't fit the generative input",
    )
    user_prompt: UserPrompt | None = Field(
        default=None, description="Optional custom prompt input by the user"
    )
    citations: bool | None | CitationsType = Field(
        default=None,
        description="Whether to include citations in the response. "
        "If set to None or False, no citations will be computed. "
        "If set to True or 'default', citations will be computed after answer generation and send as a separate `CitationsGenerativeResponse` chunk. "
        "If set to 'llm_footnotes', citations will be included in the LLM's response as markdown-styled footnotes. A `FootnoteCitationsGenerativeResponse` chunk will also be sent to map footnote ids to context keys in the `query_context`.",
    )
    citation_threshold: float | None = Field(
        default=None,
        description="If citations is set to True or 'default', this will be the similarity threshold. Value between 0 and 1, lower values will produce more citations. If not set, it will be set to the optimized threshold found by Nuclia.",
        ge=0.0,
        le=1.0,
    )
    generative_model: str | None = Field(
        default=None,
        title="Generative model",
        description="The generative model to use for the predict chat endpoint. If not provided, the model configured for the Knowledge Box is used.",
    )

    max_tokens: int | None = Field(
        default=None, description="Maximum characters to generate"
    )

    query_context_images: dict[str, Image] = Field(
        default={},
        description="The information retrieval context for the current query, each image is a base64 encoded string",
    )

    prefer_markdown: bool = Field(
        default=False,
        description="If set to true, the response will be in markdown format",
    )
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="The JSON schema to use for the generative model answers",
    )
    rerank_context: bool = Field(
        default=False,
        description="Whether to reorder the query context based on a reranker",
    )
    top_k: int | None = Field(
        default=None, description="Number of best elements to get from"
    )

    format_prompt: bool = Field(
        default=True,
        description="If set to false, the prompt given as `user_prompt` will be used as is, without any formatting for question or context. If set to true, the prompt must contain the placeholders {question} and {context} to be replaced by the question and context respectively",
    )
    seed: int | None = Field(
        default=None,
        description="Seed use for the generative model for a deterministic output.",
    )
    reasoning: Reasoning | bool = Field(
        title="Reasoning options",
        default=False,
        description=(
            "Reasoning options for the generative model. "
            "Set to True to enable default reasoning, False to disable, or provide a Reasoning object for custom options."
        ),
    )


class RephraseModel(BaseModel):
    question: str
    chat_history: list[ChatContextMessage] = []
    user_id: str
    user_context: list[str] = []
    generative_model: str | None = Field(
        default=None,
        title="Generative model",
        description="The generative model to use for the rephrase endpoint. If not provided, the model configured for the Knowledge Box is used.",
    )
    chat_history_relevance_threshold: (
        Annotated[
            float,
            Field(
                ge=0.0,
                le=1.0,
                description="Threshold to determine if the past chat history is relevant to rephrase the user's question. "
                "0 - Always treat previous messages as relevant (always rephrase)."
                "1 - Always treat previous messages as irrelevant (never rephrase)."
                "Values in between adjust the sensitivity.",
            ),
        ]
        | None
    ) = None


class RagStrategyName:
    FIELD_EXTENSION = "field_extension"
    FULL_RESOURCE = "full_resource"
    HIERARCHY = "hierarchy"
    NEIGHBOURING_PARAGRAPHS = "neighbouring_paragraphs"
    METADATA_EXTENSION = "metadata_extension"
    PREQUERIES = "prequeries"
    CONVERSATION = "conversation"
    GRAPH = "graph_beta"


class ImageRagStrategyName:
    PAGE_IMAGE = "page_image"
    TABLES = "tables"
    PARAGRAPH_IMAGE = "paragraph_image"


class RagStrategy(BaseModel):
    name: Any

    @model_validator(mode="after")
    def set_discriminator(self) -> Self:
        # Ensure discriminator is explicitly set so it's always serialized
        self.name = self.name
        return self


class ImageRagStrategy(BaseModel):
    name: Any

    @model_validator(mode="after")
    def set_discriminator(self) -> Self:
        # Ensure discriminator is explicitly set so it's always serialized
        self.name = self.name
        return self


ALLOWED_FIELD_TYPES: dict[str, str] = {
    "t": "text",
    "f": "file",
    "u": "link",
    "c": "conversation",
    "a": "generic",
}


class FieldExtensionStrategy(RagStrategy):
    name: Literal["field_extension"] = "field_extension"
    fields: list[str] = Field(
        default=[],
        title="Fields",
        description="List of field ids to extend the context with. It will try to extend the retrieval context with the specified fields in the matching resources. The field ids have to be in the format `{field_type}/{field_name}`, like 'a/title', 'a/summary' for title and summary fields or 't/amend' for a text field named 'amend'.",
    )
    data_augmentation_field_prefixes: list[str] = Field(
        default=[],
        description="List of prefixes for data augmentation added fields to extend the context with. For example, if the prefix is 'simpson', all fields that are a result of data augmentation with that prefix will be used to extend the context.",
    )

    @model_validator(mode="after")
    def field_extension_strategy_validator(self) -> Self:
        # We also accept /{field_type}/{field_name} for legacy/usability but we
        # remove it so we don't deal with it
        self.fields = [field.strip("/") for field in self.fields]

        # Check that the fields are in the format {field_type}/{field_name}
        for field in self.fields:
            try:
                field_type, _ = field.split("/")
            except ValueError:
                raise ValueError(
                    f"Field '{field}' is not in the format {{field_type}}/{{field_name}}"
                )
            if field_type not in ALLOWED_FIELD_TYPES:
                allowed_field_types_part = ", ".join(
                    [
                        f"'{fid}' for '{fname}' fields"
                        for fid, fname in ALLOWED_FIELD_TYPES.items()
                    ]
                )
                raise ValueError(
                    f"Field '{field}' does not have a valid field type. "
                    f"Valid field types are: {allowed_field_types_part}."
                )
        return self


class FullResourceApplyTo(BaseModel):
    exclude: list[str] = Field(
        default_factory=list,
        title="Labels to exclude from full resource expansion",
        description="Resources from matches containing any of these labels won't expand to the full resource. This may be useful to exclude long and not interesting resources and expend less tokens",
    )


class FullResourceStrategy(RagStrategy):
    name: Literal["full_resource"] = "full_resource"
    count: int | None = Field(
        default=None,
        title="Count",
        description="Maximum number of full documents to retrieve. If not specified, all matching documents are retrieved.",
        ge=1,
    )
    include_remaining_text_blocks: bool = Field(
        default=False,
        title="Include remaining text blocks",
        description="Whether to include the remaining text blocks after the maximum number of resources has been reached.",
    )
    apply_to: FullResourceApplyTo | None = Field(
        default=None,
        title="Apply to certain resources only",
        description="Define which resources to exclude from serialization",
    )


class HierarchyResourceStrategy(RagStrategy):
    name: Literal["hierarchy"] = "hierarchy"
    count: int = Field(
        default=0,
        title="Count",
        description="Number of extra characters that are added to each matching paragraph when adding to the context.",
        ge=0,
        le=1024,
    )


class NeighbouringParagraphsStrategy(RagStrategy):
    name: Literal["neighbouring_paragraphs"] = "neighbouring_paragraphs"
    before: int = Field(
        default=2,
        title="Before",
        description="Number of previous neighbouring paragraphs to add to the context, for each matching paragraph in the retrieval step.",
        ge=0,
    )
    after: int = Field(
        default=2,
        title="After",
        description="Number of following neighbouring paragraphs to add to the context, for each matching paragraph in the retrieval step.",
        ge=0,
    )


class MetadataExtensionType(str, Enum):
    ORIGIN = "origin"
    CLASSIFICATION_LABELS = "classification_labels"
    NERS = "ners"
    EXTRA_METADATA = "extra_metadata"


class MetadataExtensionStrategy(RagStrategy):
    """
    RAG strategy to enrich the context with metadata of the matching paragraphs or its resources.
    This strategy can be combined with any of the other strategies.
    """

    name: Literal["metadata_extension"] = "metadata_extension"
    types: list[MetadataExtensionType] = Field(
        min_length=1,
        title="Types",
        description="""
List of resource metadata types to add to the context.
  - 'origin': origin metadata of the resource.
  - 'classification_labels': classification labels of the resource.
  - 'ner': Named Entity Recognition entities detected for the resource.
  - 'extra_metadata': extra metadata of the resource.

Types for which the metadata is not found at the resource are ignored and not added to the context.
""",
        examples=[
            ["origin", "classification_labels"],
            ["ners"],
        ],
    )


class ConversationalStrategy(RagStrategy):
    name: Literal["conversation"] = "conversation"
    attachments_text: bool = Field(
        default=False,
        title="Add attachments on context",
        description="Add attachments on context retrieved on conversation",
    )
    attachments_images: bool = Field(
        default=False,
        title="Add attachments images on context",
        description="Add attachments images on context retrieved on conversation if they are mime type image and using a visual LLM",
    )
    full: bool = Field(
        default=False,
        title="Add all conversation",
        description="Add all conversation fields on matched blocks",
    )
    max_messages: int = Field(
        default=15,
        title="Max messages",
        description="Max messages to append in case its not full field",
        ge=0,
    )


class PreQuery(BaseModel):
    request: "FindRequest" = Field(
        title="Request",
        description="The request to be executed before the main query.",
    )
    weight: float = Field(
        default=1.0,
        title="Weight",
        description=(
            "Weight of the prequery in the context. The weight is used to scale the results of the prequery before adding them to the context."
            "The weight should be a positive number, and they are normalized so that the sum of all weights for all prequeries is 1."
        ),
        ge=0,
    )
    id: str | None = Field(
        default=None,
        title="Prequery id",
        min_length=1,
        max_length=100,
        description="Identifier of the prequery. If not specified, it is autogenerated based on the index of the prequery in the list (prequery_0, prequery_1, ...).",
        examples=[
            "title_prequery",
            "summary_prequery",
            "prequery_1",
        ],
    )
    prefilter: bool = Field(
        default=False,
        title="Prefilter",
        description=(
            "If set to true, the prequery results are used to filter the scope of the remaining queries. "
            "The resources of the most relevant paragraphs of the prefilter queries are used as resource "
            "filters for the main query and other prequeries with the prefilter flag set to false."
        ),
    )


class PreQueriesStrategy(RagStrategy):
    """
    This strategy allows to run a set of queries before the main query and add the results to the context.
    It allows to give more importance to some queries over others by setting the weight of each query.
    The weight of the main query can also be set with the `main_query_weight` parameter.
    """

    name: Literal["prequeries"] = "prequeries"
    queries: list[PreQuery] = Field(
        title="Queries",
        description="List of queries to run before the main query. The results are added to the context with the specified weights for each query. There is a limit of 10 prequeries per request.",
        min_length=1,
        max_length=15,
    )
    main_query_weight: float = Field(
        default=1.0,
        title="Main query weight",
        description="Weight of the main query in the context. Use this to control the importance of the main query in the context.",
        ge=0,
    )


PreQueryResult = tuple[PreQuery, "KnowledgeboxFindResults"]


class RelationRanking(str, Enum):
    RERANKER = "reranker"
    GENERATIVE = "generative"


class QueryEntityDetection(str, Enum):
    PREDICT = "predict"
    SUGGEST = "suggest"


class GraphStrategy(RagStrategy):
    """
    This strategy retrieves context pieces by exploring the Knowledge Graph, starting from the entities present in the query.
    It works best if the Knowledge Box has a user-defined Graph Extraction agent enabled.
    """

    name: Literal["graph_beta"] = "graph_beta"
    hops: int = Field(
        default=3,
        title="Number of hops",
        description="""Number of hops to take when exploring the graph for relevant context.
For example,
- hops=1 will explore the neighbors of the starting entities.
- hops=2 will explore the neighbors of the neighbors of the starting entities.
And so on.
Bigger values will discover more intricate relationships but will also take more time to compute.""",
        ge=1,
        le=10,
    )
    # Here we ingore mypy because the default value is set dynamically in the model_validator
    top_k: int = Field(  # type: ignore
        default=None,
        title="Top k",
        description="Number of relationships to keep after each hop after ranking them by relevance to the query. This number correlates to more paragraphs being sent as context. If not set, this number will be set to 30 if `relation_text_as_paragraphs` is set to false or 200 if `relation_text_as_paragraphs` is set to true.",
        ge=1,
        le=300,
    )
    exclude_processor_relations: bool = Field(
        default=True,
        title="Do not use relations extracted by processor.",
        description="If set to true, only relationships extracted from a graph extraction agent are considered for context expansion.",
        validation_alias=AliasChoices(
            "agentic_graph_only", "exclude_processor_relations"
        ),
    )
    relation_text_as_paragraphs: bool = Field(
        default=False,
        title="Use relation text as context",
        description="If set to true, the text of the relationships is to create context paragraphs, this enables to use bigger top K values without running into the generative model's context limits. If set to false, the paragraphs that contain the relationships are used as context.",
    )
    relation_ranking: RelationRanking = Field(
        default=RelationRanking.RERANKER,
        title="Method to rank relationships",
        description="""Method to rank relationships.
- `reranker` uses the reranker model to rank relationships.
- `generative` uses first the reranker to first lower the amount of relationships and then the generative model to rank relationships.
The generative model is slower and consumes more tokens, but can provide better results.""",
    )
    query_entity_detection: QueryEntityDetection = Field(
        default=QueryEntityDetection.PREDICT,
        title="Method to detect entities in the query",
        description="""Method to detect entities in the query.
- `predict` uses NUA to detect entities in the query, slower and more accurate but requires an exact text match between Knowledge Box entities and entities in the query.
- `suggest` uses fuzzy search to detect entities. It's faster and more flexible but might have trouble matching entities composed of multiple words. It will fallback to Predict if no entities are detected.""",
    )
    weight: float = Field(
        default=3.0,
        title="Weight",
        description=(
            "Weight of the graph strategy in the context. The weight is used to scale the results of the strategy before adding them to the context."
            "The weight should be a positive number."
        ),
        ge=0,
    )

    @model_validator(mode="before")
    def set_dynamic_defaults(cls, values):
        if values.get("top_k") is None:
            values["top_k"] = 200 if values.get("relation_text_as_paragraphs") else 30
        return values


class TableImageStrategy(ImageRagStrategy):
    name: Literal["tables"] = "tables"


class PageImageStrategy(ImageRagStrategy):
    name: Literal["page_image"] = "page_image"
    count: int | None = Field(
        default=None,
        title="Count",
        description="Maximum number of page images to retrieve. By default, at most 5 images are retrieved.",
    )


class ParagraphImageStrategy(ImageRagStrategy):
    name: Literal["paragraph_image"] = "paragraph_image"


RagStrategies = Annotated[
    FieldExtensionStrategy
    | FullResourceStrategy
    | HierarchyResourceStrategy
    | NeighbouringParagraphsStrategy
    | MetadataExtensionStrategy
    | ConversationalStrategy
    | PreQueriesStrategy
    | GraphStrategy,
    Field(discriminator="name"),
]
RagImagesStrategies = Annotated[
    PageImageStrategy | ParagraphImageStrategy | TableImageStrategy,
    Field(discriminator="name"),
]
PromptContext = dict[str, str]
PromptContextOrder = dict[str, int]
PromptContextImages = dict[str, Image]


class CustomPrompt(BaseModel):
    system: str | None = Field(
        default=None,
        title="System prompt",
        description="System prompt given to the generative model responsible of generating the answer. This can help customize the behavior of the model when generating the answer. If not specified, the default model provider's prompt is used.",
        min_length=1,
        examples=[
            "You are a medical assistant, use medical terminology",
            "You are an IT expert, express yourself like one",
            "You are a very friendly customer service assistant, be polite",
            "You are a financial expert, use correct terms",
        ],
    )
    user: str | None = Field(
        default=None,
        title="User prompt",
        description="User prompt given to the generative model responsible of generating the answer. Use the words {context} and {question} in brackets where you want those fields to be placed, in case you want them in your prompt. Context will be the data returned by the retrieval step and question will be the user's query.",
        min_length=1,
        examples=[
            "Taking into account our previous conversation, and this context: {context} answer this {question}",
            "Give a detailed answer to this {question} in a list format. If you do not find an answer in this context: {context}, say that you don't have enough data.",
            "Given this context: {context}. Answer this {question} in a concise way using the provided context",
            "Given this context: {context}. Answer this {question} using the provided context. Please, answer always in French",
        ],
    )
    rephrase: str | None = Field(
        default=None,
        title="Rephrase",
        description=(
            "Rephrase prompt given to the generative model responsible for rephrasing the query for a more effective retrieval step. "
            "This is only used if the `rephrase` flag is set to true in the request.\n"
            "If not specified, Nuclia's default prompt is used. It must include the {question} placeholder. "
            "The placeholder will be replaced with the original question"
        ),
        min_length=1,
        examples=[
            """Rephrase this question so its better for retrieval, and keep the rephrased question in the same language as the original.
QUESTION: {question}
Please return ONLY the question without any explanation. Just the rephrased question.""",
            """Rephrase this question so its better for retrieval, identify any part numbers and append them to the end of the question separated by a commas.
            QUESTION: {question}
            Please return ONLY the question without any explanation.""",
        ],
    )


class TokensDetail(BaseModel):
    input: float
    output: float
    image: float


class Consumption(BaseModel):
    normalized_tokens: TokensDetail
    customer_key_tokens: TokensDetail


class ConsumptionGenerative(Consumption):
    type: Literal["consumption"] = "consumption"


class AskRequest(AuditMetadataBase):
    query: str = SearchParamDefaults.chat_query.to_pydantic_field()
    agentic_config_id: str | None = Field(
        default=None,
        title="Agentic configuration ID",
        description=(
            "The ID of the agentic configuration to use for this request. If not provided, the default retrieval and generation parameters of the Knowledge Box will be used. "
            "If provided, the parameters in the agentic configuration will override the parameters in the request. Note that the `query` field in the agentic configuration will be ignored, and the query in the request will be used instead."
        ),
    )
    top_k: int = Field(
        default=20,
        title="Top k",
        ge=1,
        le=200,
        description="The top most relevant results to fetch at the retrieval step. The maximum number of results allowed is 200.",
    )
    filter_expression: FilterExpression | None = (
        SearchParamDefaults.filter_expression.to_pydantic_field()
    )
    fields: list[str] = SearchParamDefaults.fields.to_pydantic_field()
    filters: list[str] | list[Filter] = Field(
        default=[],
        title="Search Filters",
        description="The list of filters to apply. Filtering examples can be found here: https://docs.nuclia.dev/docs/rag/advanced/search-filters",
    )
    keyword_filters: list[str] | list[Filter] = Field(
        default=[],
        title="Keyword filters",
        description=(
            "List of keyword filter expressions to apply to the retrieval step. "
            "The text block search will only be performed on the documents that contain the specified keywords. "
            "The filters are case-insensitive, and only alphanumeric characters and spaces are allowed. "
            "Filtering examples can be found here: https://docs.nuclia.dev/docs/rag/advanced/search-filters"
        ),
        examples=[
            ["NLP", "BERT"],
            [Filter(all=["NLP", "BERT"])],
            ["Friedrich Nietzsche", "Immanuel Kant"],
        ],
    )
    vectorset: str | None = SearchParamDefaults.vectorset.to_pydantic_field()
    min_score: float | MinScore | None = Field(
        default=None,
        title="Minimum score",
        description="Minimum score to filter search results. Results with a lower score will be ignored. Accepts either a float or a dictionary with the minimum scores for the bm25 and vector indexes. If a float is provided, it is interpreted as the minimum score for vector index search.",
    )
    features: list[ChatOptions] = SearchParamDefaults.chat_features.to_pydantic_field()
    range_creation_start: DateTime | None = (
        SearchParamDefaults.range_creation_start.to_pydantic_field()
    )
    range_creation_end: DateTime | None = (
        SearchParamDefaults.range_creation_end.to_pydantic_field()
    )
    range_modification_start: DateTime | None = (
        SearchParamDefaults.range_modification_start.to_pydantic_field()
    )
    range_modification_end: DateTime | None = (
        SearchParamDefaults.range_modification_end.to_pydantic_field()
    )
    show: list[ResourceProperties] = SearchParamDefaults.show.to_pydantic_field()
    field_type_filter: list[FieldTypeName] = (
        SearchParamDefaults.field_type_filter.to_pydantic_field()
    )
    extracted: list[ExtractedDataTypeName] = (
        SearchParamDefaults.extracted.to_pydantic_field()
    )
    context: list[ChatContextMessage] | None = (
        SearchParamDefaults.chat_context.to_pydantic_field()
    )
    chat_history: list[ChatContextMessage] | None = (
        SearchParamDefaults.chat_history.to_pydantic_field()
    )
    extra_context: list[str] | None = Field(
        default=None,
        title="Extra query context",
        description="""Additional context that is added to the retrieval context sent to the LLM.
        It allows extending the chat feature with content that may not be in the Knowledge Box.""",
    )
    extra_context_images: list[Image] | None = Field(
        default=None,
        title="Extra query context images",
        description="""Additional images added to the retrieval context sent to the LLM."
        It allows extending the chat feature with content that may not be in the Knowledge Box.""",
    )
    query_image: Image | None = Field(
        default=None,
        title="Query image",
        description="Image that will be used together with the query text for retrieval and then sent to the LLM as part of the context. "
        "If a query image is provided, the `extra_context_images` and `rag_images_strategies` will be disabled.",
    )

    # autofilter is deprecated and its logic was removed. We're just keeping it in the model definition to
    # avoid breaking changes in the python sdks. Please remove on a future major release.
    autofilter: SkipJsonSchema[bool] = False

    highlight: bool = SearchParamDefaults.highlight.to_pydantic_field()
    resource_filters: list[str] = (
        SearchParamDefaults.resource_filters.to_pydantic_field()
    )
    prompt: str | CustomPrompt | None = Field(
        default=None,
        title="Prompts",
        description="Use to customize the prompts given to the generative model. Both system and user prompts can be customized. If a string is provided, it is interpreted as the user prompt.",
    )
    rank_fusion: RankFusionName | RankFusion = (
        SearchParamDefaults.rank_fusion.to_pydantic_field()
    )
    reranker: RerankerName | Reranker = SearchParamDefaults.reranker.to_pydantic_field()
    citations: bool | None | CitationsType = Field(
        default=None,
        description="Whether to include citations in the response. "
        "If set to None or False, no citations will be computed. "
        "If set to True or 'default', citations will be computed after answer generation and send as a separate `CitationsGenerativeResponse` chunk. "
        "If set to 'llm_footnotes', citations will be included in the LLM's response as markdown-styled footnotes. A `FootnoteCitationsGenerativeResponse` chunk will also be sent to map footnote ids to context keys in the `query_context`.",
    )
    citation_threshold: float | None = Field(
        default=None,
        description="If citations is set to True or 'default', this will be the similarity threshold. Value between 0 and 1, lower values will produce more citations. If not set, it will be set to the optimized threshold found by Nuclia.",
        ge=0.0,
        le=1.0,
    )
    security: RequestSecurity | None = SearchParamDefaults.security.to_pydantic_field()
    show_hidden: bool = SearchParamDefaults.show_hidden.to_pydantic_field()
    rag_strategies: list[RagStrategies] = Field(
        default=[],
        title="RAG context building strategies",
        description=(
            """Options for tweaking how the context for the LLM model is crafted:
- `full_resource` will add the full text of the matching resources to the context. This strategy cannot be combined with `hierarchy`, `neighbouring_paragraphs`, or `field_extension`.
- `field_extension` will add the text of the matching resource's specified fields to the context.
- `hierarchy` will add the title and summary text of the parent resource to the context for each matching paragraph.
- `neighbouring_paragraphs` will add the sorrounding paragraphs to the context for each matching paragraph.
- `metadata_extension` will add the metadata of the matching paragraphs or its resources to the context.
- `prequeries` allows to run multiple retrieval queries before the main query and add the results to the context. The results of specific queries can be boosted by the specifying weights.

If empty, the default strategy is used, which simply adds the text of the matching paragraphs to the context.
"""
        ),
        examples=[
            [{"name": "full_resource", "count": 2}],
            [
                {"name": "field_extension", "fields": ["t/amend", "a/title"]},
            ],
            [{"name": "hierarchy", "count": 2}],
            [{"name": "neighbouring_paragraphs", "before": 2, "after": 2}],
            [
                {
                    "name": "metadata_extension",
                    "types": ["origin", "classification_labels"],
                }
            ],
            [
                {
                    "name": "prequeries",
                    "queries": [
                        {
                            "request": {
                                "query": "What is the capital of France?",
                                "features": ["keyword"],
                            },
                            "weight": 0.5,
                        },
                        {
                            "request": {
                                "query": "What is the capital of Germany?",
                            },
                            "weight": 0.5,
                        },
                    ],
                }
            ],
        ],
    )
    rag_images_strategies: list[RagImagesStrategies] = Field(
        default=[],
        title="RAG image context building strategies",
        description=(
            "Options for tweaking how the image based context for the LLM model is crafted:\n"
            "- `page_image` will add the full page image of the matching resources to the context.\n"
            "- `tables` will send the table images for the paragraphs that contain tables and matched the retrieval query.\n"
            "- `paragraph_image` will add the images of the paragraphs that contain images (images for tables are not included).\n"
            "No image strategy is used by default. Note that this is only available for LLM models that support visual inputs. If the model does not support visual inputs, the image strategies will be ignored."
        ),
    )
    debug: bool = SearchParamDefaults.debug.to_pydantic_field()

    generative_model: str | None = Field(
        default=None,
        title="Generative model",
        description="The generative model to use for the chat endpoint. If not provided, the model configured for the Knowledge Box is used.",
    )
    generative_model_seed: int | None = Field(
        default=None,
        title="Seed for the generative model",
        description="The seed to use for the generative model for deterministic generation. Only supported by some models.",
    )

    max_tokens: int | MaxTokens | None = Field(
        default=None,
        title="Maximum LLM tokens to use for the request",
        description="Use to limit the amount of tokens used in the LLM context and/or for generating the answer. If not provided, the default maximum tokens of the generative model will be used. If an integer is provided, it is interpreted as the maximum tokens for the answer.",
    )

    rephrase: bool = Field(
        default=False,
        description=(
            "Rephrase the query for a more efficient retrieval. This will consume LLM tokens and make the request slower."
        ),
    )
    chat_history_relevance_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold to determine if the past chat history is relevant to rephrase the user's question. "
            "0 - Always treat previous messages as relevant (always rephrase)."
            "1 - Always treat previous messages as irrelevant (never rephrase)."
            "Values in between adjust the sensitivity."
        ),
    )

    prefer_markdown: bool = Field(
        default=False,
        title="Prefer markdown",
        description="If set to true, the response will be in markdown format",
    )

    answer_json_schema: dict[str, Any] | None = Field(
        default=None,
        title="Answer JSON schema",
        description="""Desired JSON schema for the LLM answer.
This schema is passed to the LLM so that it answers in a scructured format following the schema. If not provided, textual response is returned.
Note that when using this parameter, the answer in the generative response will not be returned in chunks, the whole response text will be returned instead.
Using this feature also disables the `citations` parameter. For maximal accuracy, please include a `description` for each field of the schema.
""",
        examples=[ANSWER_JSON_SCHEMA_EXAMPLE],
    )

    generate_answer: bool = Field(
        default=True,
        description="Whether to generate an answer using the generative model. If set to false, the response will only contain the retrieval results.",
    )

    search_configuration: str | None = Field(
        default=None,
        description="Load ask parameters from this configuration. Parameters in the request override parameters from the configuration.",
    )

    reasoning: Reasoning | bool = Field(
        default=False,
        title="Reasoning options",
        description=(
            "Reasoning options for the generative model. "
            "Set to True to enable default reasoning, False to disable, or provide a Reasoning object for custom options."
        ),
    )

    @field_validator("rag_strategies", mode="before")
    @classmethod
    def validate_rag_strategies(
        cls, rag_strategies: list[RagStrategies]
    ) -> list[RagStrategies]:
        strategy_names: set[str] = set()
        for strategy in rag_strategies or []:
            if isinstance(strategy, dict):
                obj = strategy
            elif isinstance(strategy, BaseModel):
                obj = strategy.model_dump()
            else:
                raise ValueError(
                    "RAG strategies must be defined using a valid RagStrategy object or a dictionary"
                )
            strategy_name = obj.get("name")
            if strategy_name is None:
                raise ValueError(f"Invalid strategy '{strategy}'")
            strategy_names.add(strategy_name)

        if len(strategy_names) != len(rag_strategies):
            raise ValueError("There must be at most one strategy of each type")

        for not_allowed_combination in (
            {RagStrategyName.FULL_RESOURCE, RagStrategyName.HIERARCHY},
            {RagStrategyName.FULL_RESOURCE, RagStrategyName.NEIGHBOURING_PARAGRAPHS},
            {RagStrategyName.FULL_RESOURCE, RagStrategyName.FIELD_EXTENSION},
        ):
            if not_allowed_combination.issubset(strategy_names):
                raise ValueError(
                    f"The following strategies cannot be combined in the same request: {', '.join(sorted(not_allowed_combination))}"
                )
        return rag_strategies

    @model_validator(mode="before")
    @classmethod
    def fix_legacy_rank_fusion(cls, values):
        """Dirty fix to allow passing "legacy" as rank fusion algorithm but
        convert it to RRF"""
        if isinstance(values, dict):
            rank_fusion = values.get("rank_fusion")
            if isinstance(rank_fusion, str) and rank_fusion == "legacy":
                values["rank_fusion"] = RankFusionName.RECIPROCAL_RANK_FUSION
        return values

    @model_validator(mode="after")
    def rename_context_to_chat_history(self) -> Self:
        """Bw/c rename from `context` to `chat_history`"""
        if self.context is not None and self.chat_history is not None:
            raise ValueError(
                "`context` and `chat_history` are the same, please, use the latter"
            )
        elif self.context is not None:
            self.chat_history = self.context
            self.context = None
        return self

    @field_validator("resource_filters", mode="after")
    def validate_resource_filters(cls, values: list[str]) -> list[str]:
        if values is not None:
            for v in values:
                _validate_resource_filter(v)
        return values


class TextBlockAugmentationType(str, Enum):
    NEIGHBOURING_PARAGRAPHS = "neighbouring_paragraphs"
    CONVERSATION = "conversation"
    HIERARCHY = "hierarchy"
    FULL_RESOURCE = "full_resource"
    FIELD_EXTENSION = "field_extension"
    METADATA_EXTENSION = "metadata_extension"


class AugmentedTextBlock(BaseModel):
    id: str = Field(
        description="The id of the augmented text bloc. It can be a paragraph id or a field id."
    )
    text: str = Field(
        description="The text of the augmented text block. It may include additional metadata to enrich the context"
    )
    position: TextPosition | None = Field(
        default=None,
        description="Metadata about the position of the text block in the original document.",
    )
    parent: str | None = Field(
        default=None, description="The parent text block that was augmented for."
    )
    augmentation_type: TextBlockAugmentationType = Field(
        description="Type of augmentation."
    )


class AugmentedContext(BaseModel):
    paragraphs: dict[str, AugmentedTextBlock] = Field(
        default={},
        description="Paragraphs added to the context as a result of using the `rag_strategies` parameter, typically the neighbouring_paragraphs or the conversation strategies",
    )
    fields: dict[str, AugmentedTextBlock] = Field(
        default={},
        description="Field extracted texts added to the context as a result of using the `rag_strategies` parameter, typically the hierarcy or full_resource strategies.",
    )


class AskTokens(BaseModel):
    input: int = Field(
        title="Input tokens",
        description="Number of LLM tokens used for the context in the query",
    )
    output: int = Field(
        title="Output tokens",
        description="Number of LLM tokens used for the answer",
    )
    input_nuclia: float | None = Field(
        title="Input Nuclia tokens",
        description="Number of Nuclia LLM tokens used for the context in the query",
        default=None,
    )
    output_nuclia: float | None = Field(
        title="Output Nuclia tokens",
        description="Number of Nuclia LLM tokens used for the answer",
        default=None,
    )


class AskTimings(BaseModel):
    generative_first_chunk: float | None = Field(
        default=None,
        title="Generative first chunk",
        description="Time the LLM took to generate the first chunk of the answer",
    )
    generative_total: float | None = Field(
        default=None,
        title="Generative total",
        description="Total time the LLM took to generate the answer",
    )


class SyncAskMetadata(BaseModel):
    tokens: AskTokens | None = Field(
        default=None,
        title="Tokens",
        description="Number of tokens used in the LLM context and answer",
    )
    timings: AskTimings | None = Field(
        default=None,
        title="Timings",
        description="Timings of the generative model",
    )


class AskRetrievalMatch(BaseModel):
    id: str = Field(
        title="Id",
        description="Id of the matching text block",
    )


class SyncAskResponse(BaseModel):
    answer: str = Field(
        title="Answer",
        description="The generative answer to the query",
    )
    reasoning: str | None = Field(
        default=None,
        title="Reasoning steps",
        description="The reasoning steps followed by the LLM to generate the answer. This is returned only if the reasoning feature is enabled in the request.",
    )
    answer_json: dict[str, Any] | None = Field(
        default=None,
        title="Answer JSON",
        description="The generative JSON answer to the query. This is returned only if the answer_json_schema parameter is provided in the request.",
    )
    status: str = Field(
        title="Status",
        description="The status of the query execution. It can be 'success', 'error', 'no_context' or 'no_retrieval_data'",
    )
    retrieval_results: KnowledgeboxFindResults = Field(
        title="Retrieval results",
        description="The retrieval results of the query",
    )
    retrieval_best_matches: list[AskRetrievalMatch] = Field(
        default=[],
        title="Retrieval best matches",
        description="Sorted list of best matching text blocks in the retrieval step. This includes the main query and prequeries results, if any.",
    )
    prequeries: dict[str, KnowledgeboxFindResults] | None = Field(
        default=None,
        title="Prequeries",
        description="The retrieval results of the prequeries",
    )
    learning_id: str = Field(
        default="",
        title="Learning id",
        description="The id of the learning request. This id can be used to provide feedback on the learning process.",
    )
    relations: Relations | None = Field(
        default=None,
        title="Relations",
        description="The detected relations of the answer",
    )
    citations: dict[str, Any] = Field(
        default_factory=dict,
        title="Citations",
        description="The citations of the answer. List of references to the resources used to generate the answer.",
    )
    citation_footnote_to_context: dict[str, str] = Field(
        default_factory=dict,
        title="Citation footnote to context",
        description="""Maps ids in the footnote citations to query_context keys (normally paragraph ids)""",
    )
    augmented_context: AugmentedContext | None = Field(
        default=None,
        description=(
            "Augmented text blocks that were sent to the LLM as part of the RAG strategies "
            "applied on the retrieval results in the request."
        ),
    )
    prompt_context: list[str] | None = Field(
        default=None,
        title="Prompt context",
        description="The prompt context used to generate the answer. Returned only if the debug flag is set to true",
    )
    predict_request: dict[str, Any] | None = Field(
        default=None,
        title="Predict request",
        description="The internal predict request used to generate the answer. Returned only if the debug flag is set to true",
    )
    metadata: SyncAskMetadata | None = Field(
        default=None,
        title="Metadata",
        description="Metadata of the query execution. This includes the number of tokens used in the LLM context and answer, and the timings of the generative model.",
    )
    consumption: Consumption | None = Field(
        default=None,
        title="Consumption",
        description=(
            "The consumption of the query execution. Return only if"
            " 'X-show-consumption' header is set to true in the request."
        ),
    )
    error_details: str | None = Field(
        default=None,
        title="Error details",
        description="Error details message in case there was an error",
    )
    debug: dict[str, Any] | None = Field(
        default=None,
        title="Debug information",
        description=(
            "Debug information about the ask operation. "
            "The metadata included in this field is subject to change and should not be used in production. "
            "Note that it is only available if the `debug` parameter is set to true in the request."
        ),
    )


class RetrievalAskResponseItem(BaseModel):
    type: Literal["retrieval"] = "retrieval"
    results: KnowledgeboxFindResults
    best_matches: list[AskRetrievalMatch] = Field(
        default=[],
        title="Best matches",
        description="Sorted list of best matching text blocks in the retrieval step. This includes the main query and prequeries results, if any.",
    )


class PrequeriesAskResponseItem(BaseModel):
    type: Literal["prequeries"] = "prequeries"
    results: dict[str, KnowledgeboxFindResults] = {}


class AnswerAskResponseItem(BaseModel):
    type: Literal["answer"] = "answer"
    text: str


class ReasoningAskResponseItem(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    text: str


class JSONAskResponseItem(BaseModel):
    type: Literal["answer_json"] = "answer_json"
    object: dict[str, Any]


class MetadataAskResponseItem(BaseModel):
    type: Literal["metadata"] = "metadata"
    tokens: AskTokens
    timings: AskTimings


class ConsumptionResponseItem(BaseModel):
    type: Literal["consumption"] = "consumption"
    normalized_tokens: TokensDetail
    customer_key_tokens: TokensDetail


class AugmentedContextResponseItem(BaseModel):
    type: Literal["augmented_context"] = "augmented_context"
    augmented: AugmentedContext = Field(
        description=(
            "Augmented text blocks that were sent to the LLM as part of the RAG strategies "
            "applied on the retrieval results in the request."
        )
    )


class CitationsAskResponseItem(BaseModel):
    type: Literal["citations"] = "citations"
    citations: dict[str, Any]


class FootnoteCitationsAskResponseItem(BaseModel):
    type: Literal["footnote_citations"] = "footnote_citations"
    footnote_to_context: dict[str, str] = Field(
        description="""Maps ids in the footnote citations to query_context keys (normally paragraph ids)
e.g.,
{ "block-AA": "f44f4e8acbfb1d48de3fd3c2fb04a885/f/f44f4e8acbfb1d48de3fd3c2fb04a885/73758-73972", ... }
If the query_context is a list, it will map to 1-based indices as strings
e.g., { "block-AA": "1", "block-AB": "2", ... }
"""
    )


class StatusAskResponseItem(BaseModel):
    type: Literal["status"] = "status"
    code: str
    status: str
    details: str | None = None


class ErrorAskResponseItem(BaseModel):
    type: Literal["error"] = "error"
    error: str


class RelationsAskResponseItem(BaseModel):
    type: Literal["relations"] = "relations"
    relations: Relations


class DebugAskResponseItem(BaseModel):
    type: Literal["debug"] = "debug"
    metadata: dict[str, Any]
    metrics: dict[str, Any]


AskResponseItemType = (
    AnswerAskResponseItem
    | ReasoningAskResponseItem
    | JSONAskResponseItem
    | MetadataAskResponseItem
    | AugmentedContextResponseItem
    | CitationsAskResponseItem
    | FootnoteCitationsAskResponseItem
    | StatusAskResponseItem
    | ErrorAskResponseItem
    | RetrievalAskResponseItem
    | RelationsAskResponseItem
    | DebugAskResponseItem
    | PrequeriesAskResponseItem
    | ConsumptionResponseItem
)


class AskResponseItem(BaseModel):
    item: AskResponseItemType = Field(..., discriminator="type")


def parse_custom_prompt(item: AskRequest) -> CustomPrompt:
    prompt = CustomPrompt()
    if item.prompt is not None:
        if isinstance(item.prompt, str):
            # If the prompt is a string, it is interpreted as the user prompt
            prompt.user = item.prompt
        else:
            prompt.user = item.prompt.user
            prompt.system = item.prompt.system
            prompt.rephrase = item.prompt.rephrase
    return prompt


def parse_rephrase_prompt(item: AskRequest) -> str | None:
    prompt = parse_custom_prompt(item)
    return prompt.rephrase


def parse_max_tokens(max_tokens: int | MaxTokens | None) -> MaxTokens | None:
    if isinstance(max_tokens, int):
        # If the max_tokens is an integer, it is interpreted as the max_tokens value for the generated answer.
        # The max tokens for the context is set to None to use the default value for the model (comes in the
        # NUA's query endpoint response).
        return MaxTokens(answer=max_tokens, context=None)
    return max_tokens
