"""
tests/ask/test_model.py — unit tests for hyperforge_nucliadb_agentic.ask.model

Covers:
- Pydantic model construction and defaults
- Field validators (rag_strategies, resource_filters)
- Model validators (rename context → chat_history, rank_fusion legacy fix)
- CitationsType enum
- RAG strategy discriminated-union round-trips
- AskRequest validation of illegal strategy combinations
- SyncAskResponse structure
"""

import pytest
from pydantic import ValidationError

from hyperforge_nucliadb_agentic.ask.model import (
    AskRequest,
    AskTimings,
    AskTokens,
    Author,
    ChatContextMessage,
    CitationsType,
    CustomPrompt,
    FieldExtensionStrategy,
    FullResourceStrategy,
    GraphStrategy,
    HierarchyResourceStrategy,
    MaxTokens,
    MetadataExtensionStrategy,
    MetadataExtensionType,
    NeighbouringParagraphsStrategy,
    Reasoning,
    SyncAskMetadata,
    SyncAskResponse,
    parse_custom_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_FIND_RESULTS = {
    "resources": {},
    "query": "",
    "total": 0,
    "min_score": 0.0,
    "shards": [],
}


def _sync_ask_response(**extra) -> SyncAskResponse:
    return SyncAskResponse(
        answer="test answer",
        status="success",
        retrieval_results=MINIMAL_FIND_RESULTS,  # type: ignore
        **extra,
    )


# ---------------------------------------------------------------------------
# ChatContextMessage
# ---------------------------------------------------------------------------


class TestChatContextMessage:
    def test_user_message(self):
        msg = ChatContextMessage(author=Author.USER, text="hello")
        assert msg.author == Author.USER
        assert msg.text == "hello"

    def test_nuclia_message(self):
        msg = ChatContextMessage(author=Author.NUCLIA, text="response")
        assert msg.author == Author.NUCLIA

    def test_message_is_alias_of_chat_context_message(self):
        from hyperforge_nucliadb_agentic.ask.model import Message

        assert Message is ChatContextMessage


# ---------------------------------------------------------------------------
# CitationsType
# ---------------------------------------------------------------------------


class TestCitationsType:
    def test_none_value(self):
        assert CitationsType.NONE == "none"

    def test_default_value(self):
        assert CitationsType.DEFAULT == "default"

    def test_llm_footnotes_value(self):
        assert CitationsType.LLM_FOOTNOTES == "llm_footnotes"


# ---------------------------------------------------------------------------
# MaxTokens
# ---------------------------------------------------------------------------


class TestMaxTokens:
    def test_defaults_are_none(self):
        mt = MaxTokens()
        assert mt.context is None
        assert mt.answer is None

    def test_set_values(self):
        mt = MaxTokens(context=1000, answer=500)
        assert mt.context == 1000
        assert mt.answer == 500


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_defaults(self):
        r = Reasoning()
        assert r.display is True
        assert r.effort == "medium"
        assert r.budget_tokens == 15_000

    def test_custom_effort(self):
        r = Reasoning(effort="high", budget_tokens=30_000)
        assert r.effort == "high"
        assert r.budget_tokens == 30_000


# ---------------------------------------------------------------------------
# RAG strategy models
# ---------------------------------------------------------------------------


class TestFieldExtensionStrategy:
    def test_valid_fields(self):
        s = FieldExtensionStrategy(fields=["a/title", "t/body"])
        assert "a/title" in s.fields

    def test_strips_leading_slash(self):
        s = FieldExtensionStrategy(fields=["/a/title"])
        assert s.fields == ["a/title"]

    def test_invalid_field_type_raises(self):
        with pytest.raises(ValidationError):
            FieldExtensionStrategy(fields=["z/unknown"])

    def test_invalid_format_no_slash_raises(self):
        with pytest.raises(ValidationError):
            FieldExtensionStrategy(fields=["notafield"])

    def test_empty_fields_ok(self):
        s = FieldExtensionStrategy(fields=[])
        assert s.fields == []


class TestFullResourceStrategy:
    def test_default_count_is_none(self):
        s = FullResourceStrategy()
        assert s.count is None

    def test_set_count(self):
        s = FullResourceStrategy(count=3)
        assert s.count == 3

    def test_count_must_be_positive(self):
        with pytest.raises(ValidationError):
            FullResourceStrategy(count=0)


class TestNeighbouringParagraphsStrategy:
    def test_defaults(self):
        s = NeighbouringParagraphsStrategy()
        assert s.before == 2
        assert s.after == 2

    def test_custom_values(self):
        s = NeighbouringParagraphsStrategy(before=5, after=5)
        assert s.before == 5
        assert s.after == 5

    def test_negative_before_raises(self):
        with pytest.raises(ValidationError):
            NeighbouringParagraphsStrategy(before=-1)


class TestMetadataExtensionStrategy:
    def test_requires_at_least_one_type(self):
        with pytest.raises(ValidationError):
            MetadataExtensionStrategy(types=[])

    def test_valid_types(self):
        s = MetadataExtensionStrategy(
            types=[
                MetadataExtensionType.ORIGIN,
                MetadataExtensionType.CLASSIFICATION_LABELS,
            ]
        )
        assert MetadataExtensionType.ORIGIN in s.types

    def test_string_types_accepted(self):
        s = MetadataExtensionStrategy(types=["origin"])  # type: ignore
        assert s.types[0] == MetadataExtensionType.ORIGIN


class TestHierarchyResourceStrategy:
    def test_default_count_zero(self):
        s = HierarchyResourceStrategy()
        assert s.count == 0

    def test_count_max(self):
        s = HierarchyResourceStrategy(count=1024)
        assert s.count == 1024

    def test_count_exceeds_max_raises(self):
        with pytest.raises(ValidationError):
            HierarchyResourceStrategy(count=1025)


class TestGraphStrategy:
    def test_defaults(self):
        s = GraphStrategy()
        assert s.hops == 3
        assert s.exclude_processor_relations is True
        assert s.relation_text_as_paragraphs is False

    def test_dynamic_top_k_default_without_relation_text(self):
        s = GraphStrategy(relation_text_as_paragraphs=False)
        assert s.top_k == 30

    def test_dynamic_top_k_with_relation_text(self):
        s = GraphStrategy(relation_text_as_paragraphs=True)
        assert s.top_k == 200

    def test_explicit_top_k_overrides_dynamic_default(self):
        s = GraphStrategy(top_k=50)
        assert s.top_k == 50


# ---------------------------------------------------------------------------
# AskRequest
# ---------------------------------------------------------------------------


class TestAskRequest:
    def test_minimal_valid_request(self):
        req = AskRequest(query="what is ML?")
        assert req.query == "what is ML?"

    def test_default_top_k(self):
        req = AskRequest(query="q")
        assert req.top_k == 20

    def test_top_k_lower_bound(self):
        with pytest.raises(ValidationError):
            AskRequest(query="q", top_k=0)

    def test_top_k_upper_bound(self):
        with pytest.raises(ValidationError):
            AskRequest(query="q", top_k=201)

    def test_context_renamed_to_chat_history(self):
        history = [{"author": "USER", "text": "hi"}]
        req = AskRequest(query="q", context=history)  # type: ignore
        assert req.chat_history is not None
        assert req.context is None

    def test_cannot_set_both_context_and_chat_history(self):
        history = [{"author": "USER", "text": "hi"}]
        with pytest.raises(ValidationError):
            AskRequest(query="q", context=history, chat_history=history)  # type: ignore

    def test_legacy_rank_fusion_converted_to_rrf(self):
        req = AskRequest(query="q", rank_fusion="legacy")  # type: ignore
        from hyperforge_nucliadb_agentic.ask.model import RankFusionName

        assert req.rank_fusion == RankFusionName.RECIPROCAL_RANK_FUSION

    def test_rag_strategy_full_resource(self):
        req = AskRequest(
            query="q",
            rag_strategies=[{"name": "full_resource", "count": 2}],  # type: ignore
        )
        assert req.rag_strategies[0].name == "full_resource"

    def test_rag_strategy_duplicates_raise(self):
        with pytest.raises(ValidationError):
            AskRequest(
                query="q",
                rag_strategies=[  # type: ignore
                    {"name": "full_resource"},
                    {"name": "full_resource"},
                ],
            )

    def test_illegal_strategy_combination_full_resource_and_hierarchy(self):
        with pytest.raises(ValidationError):
            AskRequest(
                query="q",
                rag_strategies=[  # type: ignore
                    {"name": "full_resource"},
                    {"name": "hierarchy"},
                ],
            )

    def test_illegal_strategy_combination_full_resource_and_field_extension(self):
        with pytest.raises(ValidationError):
            AskRequest(
                query="q",
                rag_strategies=[  # type: ignore
                    {"name": "full_resource"},
                    {"name": "field_extension", "fields": ["a/title"]},
                ],
            )

    def test_illegal_strategy_combination_full_resource_and_neighbouring(self):
        with pytest.raises(ValidationError):
            AskRequest(
                query="q",
                rag_strategies=[  # type: ignore
                    {"name": "full_resource"},
                    {"name": "neighbouring_paragraphs"},
                ],
            )

    def test_rag_strategy_invalid_object_raises(self):
        with pytest.raises(ValidationError):
            AskRequest(query="q", rag_strategies=["not-a-dict"])  # type: ignore

    def test_resource_filters_validation(self):
        # Valid UUID-like filter
        req = AskRequest(
            query="q", resource_filters=["a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"]
        )
        assert len(req.resource_filters) == 1


# ---------------------------------------------------------------------------
# SyncAskResponse
# ---------------------------------------------------------------------------


class TestSyncAskResponse:
    def test_minimal_response(self):
        resp = _sync_ask_response()
        assert resp.answer == "test answer"
        assert resp.status == "success"
        assert resp.citations == {}
        assert resp.citation_footnote_to_context == {}

    def test_with_reasoning(self):
        resp = _sync_ask_response(reasoning="step 1: ...")
        assert resp.reasoning == "step 1: ..."

    def test_answer_json_optional(self):
        resp = _sync_ask_response()
        assert resp.answer_json is None

    def test_with_metadata(self):
        metadata = SyncAskMetadata(
            tokens=AskTokens(
                input=100, output=50, input_nuclia=0.5, output_nuclia=0.25
            ),
            timings=AskTimings(generative_first_chunk=0.3, generative_total=1.2),
        )
        resp = _sync_ask_response(metadata=metadata)
        assert resp.metadata.tokens.input == 100  # type: ignore
        assert resp.metadata.timings.generative_total == 1.2  # type: ignore

    def test_citations_dict(self):
        resp = _sync_ask_response(citations={"block-AA": "rid/f/field/0-100"})
        assert "block-AA" in resp.citations

    def test_augmented_context_default_empty(self):
        resp = _sync_ask_response()
        assert resp.augmented_context is None


# ---------------------------------------------------------------------------
# CustomPrompt / parse_custom_prompt
# ---------------------------------------------------------------------------


class TestCustomPrompt:
    def test_defaults_all_none(self):
        cp = CustomPrompt()
        assert cp.system is None
        assert cp.user is None
        assert cp.rephrase is None

    def test_set_system(self):
        cp = CustomPrompt(system="You are an expert.")
        assert cp.system == "You are an expert."


class TestParseCustomPrompt:
    def test_string_prompt_becomes_user_prompt(self):
        req = AskRequest(query="q", prompt="Use {context} to answer {question}")
        result = parse_custom_prompt(req)
        assert result.user == "Use {context} to answer {question}"
        assert result.system is None

    def test_custom_prompt_object_preserved(self):
        cp = CustomPrompt(system="You are an expert.", user="Answer {question}")
        req = AskRequest(query="q", prompt=cp)
        result = parse_custom_prompt(req)
        assert result.system == "You are an expert."
        assert result.user == "Answer {question}"

    def test_none_prompt_returns_empty_custom_prompt(self):
        req = AskRequest(query="q")
        result = parse_custom_prompt(req)
        assert result.system is None
        assert result.user is None
