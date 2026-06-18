from asyncio import Event, Queue, Task, create_task
from collections.abc import AsyncGenerator
import json

from nucliadb_agentic_api.agentic.ask_transform_to_interaction import (
    interaction_from_ask_request,
)
from typing_extensions import assert_never
from hyperforge_nucliadb_agentic.ask.model import (
    AnswerAskResponseItem,
    AskRequest,
    AskResponseItemType,
    AskRetrievalMatch,
    AskTimings,
    AskTokens,
    AugmentedContextResponseItem,
    CitationsAskResponseItem,
    ConsumptionResponseItem,
    DebugAskResponseItem,
    FootnoteCitationsAskResponseItem,
    JSONAskResponseItem,
    MetadataAskResponseItem,
    PrequeriesAskResponseItem,
    ReasoningAskResponseItem,
    RelationsAskResponseItem,
    RetrievalAskResponseItem,
    StatusAskResponseItem,
    TokensDetail,
)
from hyperforge_nucliadb_agentic.ask.predict import AnswerStatusCode
from hyperforge_nucliadb_agentic.ask.search.ask import AskResult
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    GenerativeChunk,
    MetaGenerativeResponse,
    StatusGenerativeResponse,
)
from hyperforge.interaction import AnswerOperation, AragAnswer
from nucliadb_sdk.v2.exceptions import UnprocessableEntity
from nuclia_models.predict.generative_responses import (
    ReasoningGenerativeResponse,
    TextGenerativeResponse,
)
import asyncio
from typing import TYPE_CHECKING

from hyperforge.api.v1.interaction import stream_response
from hyperforge_nucliadb_agentic.ask.search.metrics import (
    AskMetrics,
)
from hyperforge_nucliadb_agentic.ask.search.retrieval import (
    sorted_prompt_context_list,
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
        self.queue = Queue()
        self.task: Task | None = None

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
                print("Putting message in queue:", msg)
                await self.queue.put(msg)
                self.event_learning_id.set()
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

        if len(self.prequeries_results) > 0:
            item = PrequeriesAskResponseItem()
            for index, (prequery, result) in enumerate(self.prequeries_results):
                prequery_id = prequery.id or f"prequery_{index}"
                item.results[prequery_id] = result
            yield item

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

        # Audit the answer
        if self._object is None:
            audit_answer = self._answer_text.encode("utf-8")
        else:
            audit_answer = json.dumps(self._object.object).encode("utf-8")
        self.auditor.audit(
            text_answer=audit_answer,
            text_reasoning=self._reasoning_text,
            generative_answer_time=self.metrics["stream_predict_answer"],
            generative_answer_first_chunk_time=self.metrics.get_first_chunk_time() or 0,
            generative_reasoning_first_chunk_time=self.metrics.get_first_reasoning_chunk_time(),
            rephrase_time=self.metrics.get("rephrase"),
            status_code=self.status_code,
        )

        yield AugmentedContextResponseItem(augmented=self.augmented_context)

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

        # Stream out debug information
        if self.ask_request_with_debug_flag:
            predict_request = None
            if self.debug_chat_model:
                predict_request = self.debug_chat_model.model_dump(mode="json")
            yield DebugAskResponseItem(
                metadata={
                    "prompt_context": sorted_prompt_context_list(
                        self.prompt_context, self.prompt_context_order
                    ),
                    "predict_request": predict_request,
                },
                metrics=self.metrics.dump(),
            )

    async def websocket_to_ask(
        self,
    ) -> AsyncGenerator[TextGenerativeResponse | ReasoningGenerativeResponse, None]:

        meta = MetaGenerativeResponse(
            input_tokens=0,
            output_tokens=0,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
            timings={},
            learning_id=self.nuclia_learning_id,
            model_name=None,
            trace_id=None,
        )

        while True:
            answer: AragAnswer = await self.queue.get()
            breakpoint()
            if answer.operation == AnswerOperation.DONE:
                breakpoint()
                break

            if answer.operation == AnswerOperation.ERROR:
                breakpoint()
                raise UnprocessableEntity(
                    message=answer.exception.detail
                    if answer.exception
                    else "Unknown error in agent execution"
                )

            if answer.operation == AnswerOperation.REASONING and answer.reasoning:
                breakpoint()
                if self._reasoning_text is None:
                    self._reasoning_text = answer.reasoning.text
                else:
                    self._reasoning_text += answer.reasoning.text
                yield ReasoningGenerativeResponse(text=answer.reasoning.text)

            if (
                answer.operation == AnswerOperation.ANSWER_CHUNK
                and answer.streaming_response_chunk
            ):
                breakpoint()
                self._answer_text += answer.streaming_response_chunk.text
                yield TextGenerativeResponse(
                    text=answer.streaming_response_chunk.text
                    if answer.streaming_response_chunk
                    else ""
                )

            if answer.operation == AnswerOperation.ANSWER:
                breakpoint()
                if answer.step:
                    # We need to collect tokens
                    pass

                if answer.possible_answer:
                    # Not usefull for ask endpoint, more for a agent playground where we want to show the final answer separately
                    pass

                if answer.context:
                    # To fill the find payload
                    for chunk in answer.context.chunks:
                        # TODO Transform to find results
                        KnowledgeboxFindResults(resources={})
                        find_results.resources.setdefault(chunk.chunk_id, []).append(
                            chunk
                        )
                    pass

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
                # self._footnote_citations = item
                # elif isinstance(item, JSONGenerativeResponse):
                #     self._object = item
                # elif isinstance(item, StatusGenerativeResponse):
                #     self._status = item
                # elif isinstance(item, CitationsGenerativeResponse):
                #     self._citations = item
                # elif isinstance(item, FootnoteCitationsGenerativeResponse):
                #     self._footnote_citations = item
                # elif isinstance(item, MetaGenerativeResponse):
                #     self._metadata = item
                # elif isinstance(item, Consumption):
                #     self._consumption = item

            if answer.operation == AnswerOperation.AGENT_REQUEST:
                # Not supported by Ask endpoint
                pass

            if answer.operation == AnswerOperation.START:
                yield GenerativeChunk(
                    chunk=StatusGenerativeResponse(
                        code="", details="Agent execution started"
                    )
                )

        if self.task:
            self.task.cancel()
