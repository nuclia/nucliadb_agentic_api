import asyncio
from asyncio import Queue
from collections.abc import AsyncGenerator

from hyperforge.interaction import AnswerOperation, AragAnswer
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    GenerativeChunk,
    MetaGenerativeResponse,
    ReasoningGenerativeResponse,
    StatusGenerativeResponse,
    TextGenerativeResponse,
)
from nucliadb_models.search import KnowledgeboxFindResults
from nucliadb_sdk.v2.exceptions import UnprocessableEntity
from nucliadb_telemetry.utils import get_telemetry
from opentelemetry import trace

from nucliadb_agentic_api import SERVICE_NAME


def tracer():
    provider = get_telemetry(SERVICE_NAME)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()
