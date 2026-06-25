import asyncio
import json
from asyncio import Event, Queue, Task, create_task
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from hyperforge.api.v1.interaction import stream_response
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge_nucliadb_agentic.agent import JSON_OBJECT_ID
from hyperforge_nucliadb_agentic.ask.model import (
    AnswerAskResponseItem,
    AskRequest,
    AskResponseItemType,
    AskRetrievalMatch,
    AskTimings,
    AskTokens,
    CitationsAskResponseItem,
    ConsumptionResponseItem,
    FootnoteCitationsAskResponseItem,
    JSONAskResponseItem,
    MetadataAskResponseItem,
    ReasoningAskResponseItem,
    RelationsAskResponseItem,
    RetrievalAskResponseItem,
    StatusAskResponseItem,
    TokensDetail,
)
from hyperforge_nucliadb_agentic.ask.predict import AnswerStatusCode
from hyperforge_nucliadb_agentic.ask.search.ask import AskResult, RetrievalMatch
from hyperforge_nucliadb_agentic.ask.search.metrics import (
    AskMetrics,
)
from nuclia_models.common.consumption import (
    Consumption,
)
from nuclia_models.common.consumption import (
    TokensDetail as ConsumptionTokensDetail,
)
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    FootnoteCitationsGenerativeResponse,
    JSONGenerativeResponse,
    MetaGenerativeResponse,
    ReasoningGenerativeResponse,
    StatusGenerativeResponse,
    TextGenerativeResponse,
)
from nucliadb_models.search import KnowledgeboxFindResults, Relations
from nucliadb_sdk.v2.exceptions import UnprocessableEntity
from typing_extensions import assert_never

from nucliadb_agentic_api import logger
from nucliadb_agentic_api.agentic.ask_transform_to_interaction import (
    interaction_from_ask_request,
)

if TYPE_CHECKING:
    from nucliadb_agentic_api.app import HTTPApplication


class AgenticAskResult(AskResult):
    def __init__(
        self,
        *,
        kbid: str,
        ask_request: AskRequest,
        agentic_config_id: str,
        account: str,
        app: "HTTPApplication",
        origin: str | None = None,
        generate_inner_answer: bool = True,
    ):
        # Initial attributes
        self.kbid = kbid
        self.ask_request = ask_request
        self.agentic_config_id = agentic_config_id
        self.account = account
        self.origin = origin
        self.metrics = AskMetrics()
        self.app = app
        self.nuclia_learning_id: str = ""
        self.event_learning_id = Event()
        self.queue: Queue[AragAnswer] = Queue()
        self.task: Task | None = None
        self.main_results = KnowledgeboxFindResults(resources={})
        self.generate_inner_answer = generate_inner_answer

        self._answer_text = ""
        self._reasoning_text: str | None = None

        self._object: JSONGenerativeResponse | None = None
        self._status: StatusGenerativeResponse | None = None
        self._citations: CitationsGenerativeResponse | None = None
        self._footnote_citations: FootnoteCitationsGenerativeResponse | None = None
        self._metadata: MetaGenerativeResponse | None = None
        self._relations: Relations | None = None
        self._consumption: Consumption | None = None

        self.best_matches: list[RetrievalMatch] = []

    async def loop(self):

        interaction = interaction_from_ask_request(self.ask_request)
        msg: AragAnswer
        async for msg in stream_response(
            self.app,  # type: ignore
            None,
            self.account,
            self.kbid,
            "ephemeral",
            interaction,
            workflow_id=self.agentic_config_id,
        ):
            try:
                if msg.
                await self.queue.put(msg)
            except (RuntimeError, asyncio.QueueFull):
                # WebSocket already closed
                pass

    async def start(self) -> str:
        self.task = create_task(self.loop())

        await self.event_learning_id.wait()
        return self.nuclia_learning_id

    async def _stream(self) -> AsyncGenerator[AskResponseItemType, None]:
        # First, stream out the predict answer
        first_chunk_yielded = False
        first_reasoning_chunk_yielded = False
        with self.metrics.time("stream_websocket_answer"):
            async for answer_chunk in self.websocket_to_ask():
                if isinstance(answer_chunk, TextGenerativeResponse):
                    yield AnswerAskResponseItem(text=answer_chunk.text)
                    if not first_chunk_yielded:
                        self.metrics.record_first_chunk_yielded()
                        first_chunk_yielded = True
                elif isinstance(answer_chunk, ReasoningGenerativeResponse):
                    yield ReasoningAskResponseItem(text=answer_chunk.text)
                    if not first_reasoning_chunk_yielded:
                        self.metrics.record_first_reasoning_chunk_yielded()
                        first_reasoning_chunk_yielded = True
                else:
                    assert_never(answer_chunk)

        if self._object is not None:
            yield JSONAskResponseItem(object=self._object.object)
            if not first_chunk_yielded:
                # When there is a JSON generative response, we consider the first chunk yielded
                # to be the moment when the JSON object is yielded, not the text
                self.metrics.record_first_chunk_yielded()
                first_chunk_yielded = True

        yield RetrievalAskResponseItem(
            results=self.main_results,
            best_matches=[
                AskRetrievalMatch(
                    id=match.paragraph.id,
                )
                for match in self.best_matches
            ],
        )

        # Then the status
        if self.status_code == AnswerStatusCode.ERROR:
            # If predict yielded an error status, we yield it too and halt the stream immediately
            yield StatusAskResponseItem(
                code=self.status_code.value,
                status=self.status_code.prettify(),
                details=self.status_error_details or "Unknown error",
            )
            return

        yield StatusAskResponseItem(
            code=self.status_code.value,
            status=self.status_code.prettify(),
        )

        # Stream out the citations
        if self._citations is not None:
            yield CitationsAskResponseItem(
                citations=self._citations.citations,
            )
        # Stream out the footnote citations mapping
        if self._footnote_citations is not None:
            yield FootnoteCitationsAskResponseItem(
                footnote_to_context=self._footnote_citations.footnote_to_context,
            )

        # Stream out generic metadata about the answer
        if self._metadata is not None:
            yield MetadataAskResponseItem(
                tokens=AskTokens(
                    input=self._metadata.input_tokens,
                    output=self._metadata.output_tokens,
                    input_nuclia=self._metadata.input_nuclia_tokens,
                    output_nuclia=self._metadata.output_nuclia_tokens,
                ),
                timings=AskTimings(
                    generative_first_chunk=self._metadata.timings.get(
                        "generative_first_chunk"
                    ),
                    generative_total=self._metadata.timings.get("generative"),
                ),
            )

        if self._consumption is not None:
            yield ConsumptionResponseItem(
                normalized_tokens=TokensDetail(
                    input=self._consumption.normalized_tokens.input,
                    output=self._consumption.normalized_tokens.output,
                    image=self._consumption.normalized_tokens.image,
                ),
                customer_key_tokens=TokensDetail(
                    input=self._consumption.customer_key_tokens.input,
                    output=self._consumption.customer_key_tokens.output,
                    image=self._consumption.customer_key_tokens.image,
                ),
            )

        # Stream out the relations results
        should_query_relations = (
            self.ask_request_with_relations
            and self.status_code == AnswerStatusCode.SUCCESS
        )
        if should_query_relations:
            relations = await self.get_relations_results()
            yield RelationsAskResponseItem(relations=relations)

    async def websocket_to_ask(
        self,
    ) -> AsyncGenerator[TextGenerativeResponse | ReasoningGenerativeResponse, None]:

        output_nuclia_tokens = 0.0
        input_nuclia_tokens = 0.0
        timings = {}
        self._answer_text = ""

        while True:
            answer: AragAnswer = await self.queue.get()
            if answer.operation == AnswerOperation.DONE:
                break

            if answer.operation == AnswerOperation.ERROR:
                raise UnprocessableEntity(
                    message=answer.exception.detail
                    if answer.exception
                    else "Unknown error in agent execution"
                )

            if answer.operation == AnswerOperation.REASONING and answer.reasoning:
                if self._reasoning_text is None:
                    self._reasoning_text = answer.reasoning.text
                else:
                    self._reasoning_text += answer.reasoning.text
                yield ReasoningGenerativeResponse(text=answer.reasoning.text)

            if (
                answer.operation == AnswerOperation.ANSWER_CHUNK
                and answer.streaming_response_chunk
            ):
                self._answer_text += answer.streaming_response_chunk.text
                yield TextGenerativeResponse(
                    text=answer.streaming_response_chunk.text
                    if answer.streaming_response_chunk
                    else ""
                )

            if answer.operation == AnswerOperation.ANSWER:
                if answer.step:
                    if answer.step.module == "rephrase":
                        logger.debug("Received rephrase step, recording rephrase")

                    if answer.step.module == "smart":
                        logger.debug(
                            "Received smart step, recording: %s", answer.step.value
                        )
                    if answer.step.module == "basic_ask":
                        logger.debug(
                            "Received Basic Ask step %s with value %s",
                            answer.step.agent_path,
                            answer.step.value,
                        )
                    input_nuclia_tokens += (
                        answer.step.input_nuclia_tokens
                        if answer.step.input_nuclia_tokens is not None
                        else 0.0
                    )
                    output_nuclia_tokens += (
                        answer.step.output_nuclia_tokens
                        if answer.step.output_nuclia_tokens is not None
                        else 0.0
                    )
                    timings[answer.step.module] = answer.step.timeit
                    self.event_learning_id.set()

                if answer.possible_answer:
                    # Not usefull for ask endpoint, more for a agent playground where we want to show the final answer separately
                    pass

                # Context(id='100d70a657884c2c8fdc80738dda71b9', original_question_uuid='201701ca9cd7412ba3153bcbdeb1f07e', actual_question_uuid='201701ca9cd7412ba3153bcbdeb1f07e', question='Provide dessert options that are both healthy and delicious.', chunks=[Chunk(chunk_id='catalog_search_result-0', title=None, source=None, text='Here are some healthy and delicious dessert recipes from the catalog search results:\n\n1. **Carrot Cake** - A classic dessert that can be made healthier by using whole grain flour and reducing sugar.\n   - [Download Carrot Cake Recipe](carrot-cake-A4.pdf)\n\n2. **Chocolate Chip Cookies** - You can make these healthier by using dark chocolate and whole grain flour.\n   - [Download Chocolate Chip Cookies Recipe](chocolate-chip-cookies-A4.pdf)\n\n3. **Zucchini Bread** - A great way to incorporate vegetables into a sweet treat, often made with whole grains and less sugar.\n   - [Download Zucchini Bread Recipe](Zucchini-bread-A4.pdf)\n\n4. **Banana Bread** - A delicious option that can be made healthier by using ripe bananas for sweetness and whole grain flour.\n   - [Download Banana Bread Recipe](banana-bread-A4.pdf)\n\n5. **Gingerbread Cake** - A spiced cake that can be made with healthier ingredients like whole wheat flour and less sugar.\n   - [Download Gingerbread Cake Recipe](Gingerbread-Cake-A4.pdf)\n\n6. **Sugar Cookies** - These can be made healthier by using natural sweeteners and whole grain flour.\n   - [Download Sugar Cookies Recipe](Sugar-Cookies-A4.pdf)\n\nFeel free to explore these recipes for a healthier dessert option!', labels=[], url=[], metadata=None, action="catalog_search of 4c9b0b15-de46-4a15-849b-82fe502fa5cb with parameters {'question': 'healthy and delicious dessert recipes'}", origin_url=None, origin_agent='basic_ask')], images={}, prompts=[], structured=[], source='smart_agent', agent='smart_agent', summary='Here are some healthy and delicious dessert recipes:\n\n1. **Carrot Cake** - Made healthier by using whole grain flour and reducing sugar.\n2. **Chocolate Chip Cookies** - Made healthier by using dark chocolate and whole grain flour.\n3. **Zucchini Bread** - Incorporates vegetables, often made with whole grains and less sugar.\n4. **Banana Bread** - Made healthier by using ripe bananas for sweetness and whole grain flour.\n5. **Gingerbread Cake** - Made with healthier ingredients like whole wheat flour and less sugar.\n6. **Sugar Cookies** - Made healthier by using natural sweeteners and whole grain flour.', agent_id='b6d72f8c-2e09-4ee1-bba0-b0ec711894a1', title='Default Agentic Config - Smart Agent', missing=None, citations=['catalog_search_result-0'], citations_id=None, image_urls=[])
                if answer.context:
                    try:
                        for structured in answer.context.json_objects:
                            if structured.id == JSON_OBJECT_ID:
                                self.main_results = (
                                    KnowledgeboxFindResults.model_validate(
                                        structured.json_object
                                    )
                                )
                    except Exception as e:
                        logger.error("Error validating structured data: %s", e)

                if answer.generated_text:
                    # Not usefull for ask endpoint, more for a agent playground where we want to show the final answer separately
                    pass

                yield TextGenerativeResponse(
                    text=answer.answer if answer.answer else ""
                )
                if answer.answer_citations:
                    citations = {}
                    for key, value in answer.answer_citations.metadata.items():
                        citations[key] = {
                            "context_id": value.context_id,
                            "origin_urls": value.origin_urls,
                            "chunk_index": value.chunk_index,
                        }
                    self._citations = CitationsGenerativeResponse(citations=citations)

                if answer.generated_text:
                    # TODO: This is a bit of a hack, but we need to parse the generated text as JSON and store it in the _object attribute. This is because the AskResult class expects a JSONGenerativeResponse object to be returned from the websocket_to_ask method, but we don't have that yet. Once we have a proper JSONGenerativeResponse object, we can remove this hack.
                    try:
                        self._object = JSONGenerativeResponse(
                            object=json.loads(answer.generated_text)
                        )
                    except Exception as e:
                        logger.error(f"Error processing generated text: {e}")
                # elif isinstance(item, FootnoteCitationsGenerativeResponse):
                #     self._footnote_citations = item

            if answer.operation == AnswerOperation.AGENT_REQUEST:
                # Not supported by Ask endpoint
                pass

            if answer.operation == AnswerOperation.START:
                logger.debug("Received start message")

        self._consumption = Consumption(
            normalized_tokens=ConsumptionTokensDetail(
                input=input_nuclia_tokens,
                output=output_nuclia_tokens,
                image=self.metrics.get("normalized_image_tokens") or 0,
            ),
            customer_key_tokens=ConsumptionTokensDetail(
                input=self.metrics.get("customer_key_input_tokens") or 0,
                output=self.metrics.get("customer_key_output_tokens") or 0,
                image=self.metrics.get("customer_key_image_tokens") or 0,
            ),
        )

        self._metadata = MetaGenerativeResponse(
            input_tokens=0,
            output_tokens=0,
            input_nuclia_tokens=input_nuclia_tokens,
            output_nuclia_tokens=output_nuclia_tokens,
            timings=timings,
            learning_id=self.nuclia_learning_id,
            model_name=None,
            trace_id=None,
        )

        self._status = StatusGenerativeResponse(
            code=AnswerStatusCode.SUCCESS.value,
            details=None,
        )

        if self.task:
            self.task.cancel()
