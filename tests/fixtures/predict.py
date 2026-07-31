import json
import os
import random
from collections.abc import AsyncIterator

import pytest
from nuclia.lib.nua import (
    PredictQueryRequest,
    PredictRephraseRequest,
    PredictRephraseResponse,
)
from nuclia.lib.nua_responses import ChatModel, Token, Tokens
from nuclia_models.predict.generative_responses import GenerativeChunk
from nucliadb_models.internal.predict import (
    Ner,
    QueryInfo,
    RerankModel,
    RerankResponse,
    SentenceSearch,
    TokenSearch,
)
from nucliadb_protos.utils_pb2 import RelationNode

DUMMY_RELATION_NODE = [
    RelationNode(value="Ferran", ntype=RelationNode.NodeType.ENTITY, subtype="PERSON"),
    RelationNode(
        value="Joan Antoni", ntype=RelationNode.NodeType.ENTITY, subtype="PERSON"
    ),
]

DUMMY_REPHRASE_QUERY = "This is a rephrased query"
DUMMY_LEARNING_ID = "00"
DUMMY_LEARNING_MODEL = "chatgpt"

_predict: "DummyPredictManager | None" = None


def get_predict() -> "DummyPredictManager":
    assert _predict is not None
    return _predict


class DummyPredictManager:
    default_semantic_threshold = 0.7

    def __init__(self):
        self.calls = []
        self.ndjson_reasoning = [
            b'{"chunk": {"type": "reasoning", "text": "dummy "}}\n',
            b'{"chunk": {"type": "reasoning", "text": "reasoning"}}\n',
        ]
        self.ndjson_answer = [
            b'{"chunk": {"type": "text", "text": "valid "}}\n',
            b'{"chunk": {"type": "text", "text": "answer "}}\n',
            b'{"chunk": {"type": "text", "text": "to"}}\n',
            b'{"chunk": {"type": "status", "code": "0"}}\n',
        ]
        self.max_context = 1000

    async def aclose(self):
        pass

    async def predict_rephrase(
        self, request: PredictRephraseRequest, *, kbid: str | None = None, **kwargs
    ) -> PredictRephraseResponse:
        self.calls.append(("rephrase_query", request))
        return PredictRephraseResponse(
            rephrased_query=DUMMY_REPHRASE_QUERY, use_chat_history=None
        )

    async def predict_chat_stream(
        self, item: ChatModel, *, kbid: str | None = None, **kwargs
    ) -> tuple[str, str, AsyncIterator[GenerativeChunk]]:
        self.calls.append(("chat_query_ndjson", item))

        async def generate():
            if item.reasoning is not False:
                for chunk in self.ndjson_reasoning:
                    yield GenerativeChunk.model_validate_json(chunk)
            for chunk in self.ndjson_answer:
                yield GenerativeChunk.model_validate_json(chunk)

        return DUMMY_LEARNING_ID, DUMMY_LEARNING_MODEL, generate()

    async def predict_query(
        self, item: PredictQueryRequest, *, kbid: str | None = None, **kwargs
    ) -> QueryInfo:
        assert item.text is not None
        self.calls.append(("query", item))
        response = QueryInfo(
            query=item.text,
            rephrased_query=f"Rephrased: {item.text}"
            if item.rephrase or item.query_image
            else None,
            language="en",
            visual_llm=True,
            max_context=self.max_context,
            sentence=SentenceSearch(),
            entities=TokenSearch(
                tokens=[Ner(text="text", ner="PERSON", start=0, end=2)], time=0.0
            ),
        )
        assert response.sentence is not None
        response.sentence.vectors["en-2024-04-24"] = [
            random.random() for _ in range(768)
        ]
        response.sentence.timings["en-2024-04-24"] = 0.5
        response.sentence.vectors["multilingual-2024-05-06"] = [
            random.random() for _ in range(1024)
        ]
        response.sentence.timings["multilingual-2024-05-06"] = 0.7
        return response

    async def detect_entities(
        self, kbid: str | None, sentence: str
    ) -> list[RelationNode]:
        self.calls.append(("detect_entities", sentence))
        dummy_data = os.environ.get("TEST_RELATIONS")
        if dummy_data is not None:  # pragma: no cover
            return [
                RelationNode(
                    value=token["text"],
                    ntype=RelationNode.NodeType.ENTITY,
                    subtype=token["ner"],
                )
                for token in json.loads(dummy_data)["tokens"]
            ]
        return DUMMY_RELATION_NODE

    async def predict_tokens(
        self, sentence: str, *, kbid: str | None = None, **kwargs
    ) -> Tokens:
        entities = await self.detect_entities(kbid, sentence)
        return Tokens(
            tokens=[
                Token(text=e.value, ner=e.subtype, start=0, end=0) for e in entities
            ],
            time=0.0,
        )

    async def predict_rerank(
        self, item: RerankModel, *, kbid: str | None = None, **kwargs
    ) -> RerankResponse:
        self.calls.append(("rerank", (kbid, item)))
        return RerankResponse(
            context_scores={
                paragraph_id: i for i, paragraph_id in enumerate(item.context.keys())
            }
        )


@pytest.fixture(scope="function")
async def dummy_predict() -> AsyncIterator[DummyPredictManager]:
    global _predict
    _predict = DummyPredictManager()
    try:
        yield _predict
    finally:
        _predict = None
