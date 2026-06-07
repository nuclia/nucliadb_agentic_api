from asyncio import Queue
import asyncio
from collections.abc import AsyncGenerator

from nucliadb_models.search import KnowledgeboxFindResults
from nucliadb_telemetry.utils import get_telemetry
from opentelemetry import trace

from nucliadb_agentic_api import SERVICE_NAME
from hyperforge.interaction import AnswerOperation, AragAnswer
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    GenerativeChunk,
    MetaGenerativeResponse,
    ReasoningGenerativeResponse,
    StatusGenerativeResponse,
    TextGenerativeResponse,
)
from nucliadb_sdk.v2.exceptions import UnprocessableEntity


def tracer():
    provider = get_telemetry(SERVICE_NAME)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()


async def websocket_to_ask(
    queue: Queue, task: asyncio.Task
) -> AsyncGenerator[GenerativeChunk, None]:

    meta = MetaGenerativeResponse(
        input_tokens=0,
        output_tokens=0,
        input_nuclia_tokens=0,
        output_nuclia_tokens=0,
        timings={},
    )

    while True:
        answer: AragAnswer = await queue.get()
        if answer.operation == AnswerOperation.DONE:
            break

        if answer.operation == AnswerOperation.ERROR:
            raise UnprocessableEntity(
                message=answer.exception.detail
                if answer.exception
                else "Unknown error in agent execution"
            )

        if answer.operation == AnswerOperation.REASONING:
            yield GenerativeChunk(
                chunk=ReasoningGenerativeResponse(
                    text=answer.reasoning.text if answer.reasoning else ""
                )
            )

        if answer.operation == AnswerOperation.ANSWER_CHUNK:
            yield GenerativeChunk(
                chunk=TextGenerativeResponse(
                    text=answer.streaming_response_chunk.text
                    if answer.streaming_response_chunk
                    else ""
                )
            )

        if answer.operation == AnswerOperation.ANSWER:
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
                    find_results.resources.setdefault(chunk.chunk_id, []).append(chunk)
                pass

            if answer.generated_text:
                # Not usefull for ask endpoint, more for a agent playground where we want to show the final answer separately
                pass

            yield GenerativeChunk(
                chunk=TextGenerativeResponse(
                    text=answer.answer if answer.answer else ""
                )
            )
            if answer.answer_citations:
                citations = {}
                for key, value in answer.answer_citations.metadata.items():
                    citations[key] = {
                        "context_id": value.context_id,
                        "origin_urls": value.origin_urls,
                        "chunk_index": value.chunk_index,
                    }
                yield GenerativeChunk(
                    chunk=CitationsGenerativeResponse(citations=citations)
                )

        if answer.operation == AnswerOperation.AGENT_REQUEST:
            # Not supported by Ask endpoint
            pass

        if answer.operation == AnswerOperation.START:
            yield GenerativeChunk(
                chunk=StatusGenerativeResponse(
                    code="", details="Agent execution started"
                )
            )

    yield GenerativeChunk(chunk=meta)
    task.cancel()
