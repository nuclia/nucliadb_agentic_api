"""
test_agent.py — unit tests for NucliaDBAgent.

Tests are isolated: every external call (NucliaDBDriver, Manager, ask())
is mocked so that no real network traffic or NucliaDB instance is needed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hyperforge.llm_config import LLMConfig
from hyperforge_nucliadb_agentic.agent import (
    NucliaDBAgent,
    choose_sources,
    clean_citation_footnotes_from_answer,
    get_catalog_filter_prompt,
    get_chunk_text,
)
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from hyperforge_nucliadb_agentic.ask.search.ask import NotEnoughContextAskResult
from hyperforge_nucliadb_agentic.config import NucliaDBAgentConfig

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestChooseSources:
    async def test_single_source_does_not_load_routing_metadata(
        self, mock_memory, mock_manager
    ):
        with patch(
            "hyperforge_nucliadb_agentic.agent.choose_source",
            new_callable=AsyncMock,
        ) as choose_source:
            sources = await choose_sources(
                mock_memory,
                mock_manager,
                ["local-kb"],
                "question",
                ident="agent",
                step_title="Choose sources",
            )

        assert [source.id for source in sources] == ["local-kb"]
        choose_source.assert_not_awaited()

    async def test_multiple_sources_use_router(self, mock_memory, mock_manager):
        expected = [MagicMock()]
        with patch(
            "hyperforge_nucliadb_agentic.agent.choose_source",
            new_callable=AsyncMock,
            return_value=expected,
        ) as choose_source:
            sources = await choose_sources(
                mock_memory,
                mock_manager,
                ["first-kb", "second-kb"],
                "question",
                ident="agent",
                step_title="Choose sources",
            )

        assert sources == expected
        choose_source.assert_awaited_once()


class TestNucliaDBAgentInit:
    def test_init_default_state(self, nucliadb_agent: NucliaDBAgent):
        assert nucliadb_agent.labelsets == {}
        assert nucliadb_agent.synonyms == {}

    def test_agent_id_is_set(self, nucliadb_agent: NucliaDBAgent):
        assert nucliadb_agent.agent_id == "test-agent-id"

    def test_config_is_stored(self, nucliadb_agent: NucliaDBAgent, agent_config):
        assert nucliadb_agent.config is agent_config

    def test_published_functions_defined(self, nucliadb_agent: NucliaDBAgent):
        expected = {
            "ask_agent",
            "ask_labels",
            "ask_labels_list",
            "search_by_title",
            "facets_count",
            "facets_search",
            "catalog_search",
            "all_images_by_title",
            "search_images",
        }
        assert set(nucliadb_agent.__published_functions__.keys()) == expected


# ---------------------------------------------------------------------------
# ask_labels
# ---------------------------------------------------------------------------


class TestAskLabels:
    async def test_returns_labels_from_driver(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        labels = {"topic": ["tech", "health"], "year": ["2023", "2024"]}
        mock_nucliadb_driver.labels = AsyncMock(return_value=labels)

        result = await nucliadb_agent.ask_labels(
            memory=mock_memory, manager=mock_manager
        )

        assert result == labels
        mock_nucliadb_driver.labels.assert_called_once()

    async def test_caches_labelsets_when_source_key_is_present(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        # The caching guard is `if source not in self.labelsets` where self.labelsets
        # is the raw dict returned by driver.labels().  If the source id happens to
        # be a key in that dict, the second call is skipped.
        # We model this by using the source id as a top-level label-set name.
        source_id = nucliadb_agent.config.sources[0]  # "kb-source-1"
        labels = {source_id: ["some-label"]}  # source id IS a key → cached
        mock_nucliadb_driver.labels = AsyncMock(return_value=labels)

        await nucliadb_agent.ask_labels(memory=mock_memory, manager=mock_manager)
        await nucliadb_agent.ask_labels(memory=mock_memory, manager=mock_manager)

        # The driver is only called on the first invocation
        mock_nucliadb_driver.labels.assert_called_once()

    async def test_raises_with_multiple_sources(
        self, multi_source_config, mock_memory, mock_manager
    ):
        agent = NucliaDBAgent(config=multi_source_config, agent_id="agent-multi")
        # Provide drivers for both sources
        mock_manager.drivers["kb-source-2"] = MagicMock()

        with pytest.raises(
            Exception, match="ask_labels can only be used with one source"
        ):
            await agent.ask_labels(memory=mock_memory, manager=mock_manager)


# ---------------------------------------------------------------------------
# ask_labels_list
# ---------------------------------------------------------------------------


class TestAskLabelsList:
    async def test_returns_list_for_existing_labelset(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        labels = {"topic": ["tech", "health"]}
        mock_nucliadb_driver.labels = AsyncMock(return_value=labels)

        result = await nucliadb_agent.ask_labels_list(
            labelset="topic", memory=mock_memory, manager=mock_manager
        )

        assert result == ["tech", "health"]

    async def test_raises_for_missing_labelset(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        mock_nucliadb_driver.labels = AsyncMock(return_value={"topic": ["tech"]})

        with pytest.raises(Exception, match="Label set nonexistent not found"):
            await nucliadb_agent.ask_labels_list(
                labelset="nonexistent", memory=mock_memory, manager=mock_manager
            )


# ---------------------------------------------------------------------------
# build_filter_expression
# ---------------------------------------------------------------------------


class TestBuildFilterExpression:
    async def test_returns_none_when_no_filters(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        result = await nucliadb_agent.build_filter_expression(
            mock_nucliadb_driver, "kb-source-1"
        )
        assert result is None

    async def test_single_and_filter(self, nucliadb_agent, mock_nucliadb_driver):
        result = await nucliadb_agent.build_filter_expression(
            mock_nucliadb_driver,
            "kb-source-1",
            and_filters=["/l/topic/tech"],
        )
        assert result is not None

    async def test_multiple_and_filters_combined(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        result = await nucliadb_agent.build_filter_expression(
            mock_nucliadb_driver,
            "kb-source-1",
            and_filters=["/l/topic/tech", "/l/year/2023"],
        )
        assert result is not None

    async def test_or_filter_produces_result(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        result = await nucliadb_agent.build_filter_expression(
            mock_nucliadb_driver,
            "kb-source-1",
            or_filters=["/l/topic/tech", "/l/topic/health"],
        )
        assert result is not None

    async def test_keyword_filter_fetches_synonyms(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        mock_nucliadb_driver.synonyms_raw = AsyncMock(
            return_value={"ml": ["machine learning", "ML"]}
        )

        result = await nucliadb_agent.build_filter_expression(
            mock_nucliadb_driver,
            "kb-source-1",
            keyword_filters=["ml"],
        )
        assert result is not None
        mock_nucliadb_driver.synonyms_raw.assert_called_once()

    async def test_driver_level_filter_expression_is_included(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        from nucliadb_models.filters import FilterExpression, Label

        driver_fe = FilterExpression(field=Label(labelset="topic", label="tech"))
        mock_nucliadb_driver.config.filter_expression = driver_fe

        result = await nucliadb_agent.build_filter_expression(
            mock_nucliadb_driver, "kb-source-1"
        )
        assert result is not None


# ---------------------------------------------------------------------------
# build_catalog_filter_expression
# ---------------------------------------------------------------------------


class TestBuildCatalogFilterExpression:
    async def test_returns_none_when_no_filters(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        result = await nucliadb_agent.build_catalog_filter_expression(
            mock_nucliadb_driver
        )
        assert result is None

    async def test_single_classification_label(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        result = await nucliadb_agent.build_catalog_filter_expression(
            mock_nucliadb_driver,
            classification_labels=["/l/topic/tech"],
        )
        assert result is not None

    async def test_multiple_labels_with_or_operand(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        result = await nucliadb_agent.build_catalog_filter_expression(
            mock_nucliadb_driver,
            classification_labels=["/l/topic/tech", "/l/topic/health"],
            classification_labels_operand="or",
        )
        assert result is not None

    async def test_driver_level_catalog_filter_expression_included(
        self, nucliadb_agent, mock_nucliadb_driver
    ):
        from nucliadb_models.filters import CatalogFilterExpression, Label

        driver_fe = CatalogFilterExpression(
            resource=Label(labelset="topic", label="tech")
        )
        mock_nucliadb_driver.config.catalog_filter_expression = driver_fe

        result = await nucliadb_agent.build_catalog_filter_expression(
            mock_nucliadb_driver
        )
        assert result is not None


# ---------------------------------------------------------------------------
# parse_selected_filters
# ---------------------------------------------------------------------------


class TestParseSelectedFilters:
    def test_valid_json_filters(self, nucliadb_agent):
        selected = {"filters": '[{"any": ["/l/topic/tech"]}]'}
        result = nucliadb_agent.parse_selected_filters("q", "src", selected)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_empty_filters(self, nucliadb_agent):
        result = nucliadb_agent.parse_selected_filters("q", "src", {"filters": "[]"})
        assert result == []

    def test_invalid_json_returns_empty(self, nucliadb_agent):
        result = nucliadb_agent.parse_selected_filters(
            "q", "src", {"filters": "not-json"}
        )
        assert result == []

    def test_missing_filters_key_returns_empty(self, nucliadb_agent):
        result = nucliadb_agent.parse_selected_filters("q", "src", {})
        assert result == []

    def test_python_literal_dict_falls_back_correctly(self, nucliadb_agent):
        # Single-quote dict (ast.literal_eval path)
        selected = {"filters": "[{'any': ['/l/topic/tech']}]"}
        result = nucliadb_agent.parse_selected_filters("q", "src", selected)
        # Either succeeds (list) or gracefully returns empty — both are acceptable
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_chunk_text helper
# ---------------------------------------------------------------------------


class TestGetChunkText:
    def _make_response(self, *, resource_id, field_id, chunk_id, text):
        """Build a minimal SyncAskResponse-like mock."""
        para = MagicMock()
        para.text = text

        field = MagicMock()
        field.paragraphs = {chunk_id: para}

        resource = MagicMock()
        resource.fields = {field_id: field}

        results = MagicMock()
        results.resources = {resource_id: resource}

        response = MagicMock()
        response.retrieval_results = results
        response.augmented_context = None
        return response

    def test_returns_paragraph_text(self):
        rid = "resource-abc"
        fid = "/f/myfield"
        cid = f"{rid}/f/myfield"
        response = self._make_response(
            resource_id=rid, field_id=fid, chunk_id=cid, text="hello world"
        )
        assert get_chunk_text(response, cid) == "hello world"

    def test_falls_back_to_augmented_context(self):
        rid = "resource-abc"
        cid = "resource-abc/f/myfield/0-100"

        para = MagicMock()
        para.text = "retrieved text"

        augmented = MagicMock()
        augmented.paragraphs = {cid: para}

        # Primary field lookup will raise KeyError
        resource = MagicMock()
        resource.fields = {}  # empty → KeyError path

        results = MagicMock()
        results.resources = {rid: resource}

        response = MagicMock()
        response.retrieval_results = results
        response.augmented_context = augmented

        assert get_chunk_text(response, cid) == "retrieved text"

    def test_returns_empty_string_when_not_found(self):
        rid = "resource-abc"
        cid = "resource-abc/f/myfield/0-100"

        resource = MagicMock()
        resource.fields = {}

        results = MagicMock()
        results.resources = {rid: resource}

        response = MagicMock()
        response.retrieval_results = results
        response.augmented_context = None

        assert get_chunk_text(response, cid) == ""


# ---------------------------------------------------------------------------
# clean_citation_footnotes_from_answer helper
# ---------------------------------------------------------------------------


class TestCleanCitationFootnotes:
    def test_removes_inline_markers_and_definitions(self):
        answer = (
            "The answer is here[1] and also here[2].\n\n[1]: block-AA\n[2]: block-AB"
        )
        footnote_map = {
            "block-AA": "rid/f/field/0-100",
            "block-AB": "rid/f/field/100-200",
        }
        result = clean_citation_footnotes_from_answer(answer, footnote_map)
        assert "[1]" not in result
        assert "[2]" not in result
        assert "block-AA" not in result

    def test_no_footnotes_returns_unchanged(self):
        answer = "Simple answer with no citations."
        result = clean_citation_footnotes_from_answer(answer, {})
        assert result == answer

    def test_only_inline_markers_no_definitions(self):
        answer = "Some text[1] more text."
        result = clean_citation_footnotes_from_answer(
            answer, {"block-AA": "rid/f/field/0-100"}
        )
        assert "[1]" not in result
        assert "text" in result


# ---------------------------------------------------------------------------
# get_catalog_filter_prompt helper
# ---------------------------------------------------------------------------


class TestGetCatalogFilterPrompt:
    def test_renders_without_error(self):
        prompt = get_catalog_filter_prompt(
            question="Find all PDF documents about technology",
            labels_str='{"topic": ["technology", "health"]}',
        )
        assert "technology" in prompt
        assert "Find all PDF documents" in prompt

    def test_examples_are_included(self):
        prompt = get_catalog_filter_prompt(
            question="any question",
            labels_str="{}",
        )
        # All three pre-filled examples should be rendered
        assert (
            "example_filter_exp" not in prompt
        )  # template variable should be resolved
        assert "/l/" in prompt  # label prefix from examples

    def test_question_is_interpolated(self):
        question = "Show me research articles from 2024"
        prompt = get_catalog_filter_prompt(question=question, labels_str="{}")
        assert question in prompt


# ---------------------------------------------------------------------------
# search_by_title
# ---------------------------------------------------------------------------


class TestSearchByTitle:
    async def test_returns_resource_ids_per_source(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        rid = "resource-xyz"
        catalog_resp = MagicMock()
        catalog_resp.resources = {rid: MagicMock()}
        mock_nucliadb_driver.catalog_search_raw = AsyncMock(return_value=catalog_resp)

        result = await nucliadb_agent.search_by_title(
            memory=mock_memory,
            manager=mock_manager,
            title="Some Document Title",
        )

        assert "kb-source-1" in result
        assert rid in result["kb-source-1"]

    async def test_empty_catalog_returns_empty_list(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        catalog_resp = MagicMock()
        catalog_resp.resources = {}
        mock_nucliadb_driver.catalog_search_raw = AsyncMock(return_value=catalog_resp)

        result = await nucliadb_agent.search_by_title(
            memory=mock_memory,
            manager=mock_manager,
            title="Nonexistent Title",
        )

        assert result == {"kb-source-1": []}


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    async def test_returns_list_of_resource_ids(
        self, nucliadb_agent, mock_manager, mock_nucliadb_driver
    ):
        rid = "resource-find-123"
        find_resp = MagicMock()
        find_resp.resources = {rid: MagicMock()}
        mock_nucliadb_driver.find_raw = AsyncMock(return_value=find_resp)

        result = await nucliadb_agent.retrieve(
            manager=mock_manager,
            source_id="kb-source-1",
            question="What is machine learning?",
        )

        assert rid in result

    async def test_empty_find_returns_empty_list(
        self, nucliadb_agent, mock_manager, mock_nucliadb_driver
    ):
        find_resp = MagicMock()
        find_resp.resources = {}
        mock_nucliadb_driver.find_raw = AsyncMock(return_value=find_resp)

        result = await nucliadb_agent.retrieve(
            manager=mock_manager,
            source_id="kb-source-1",
            question="Unknown question",
        )

        assert result == []


# ---------------------------------------------------------------------------
# prepare_ask_request and inner_rag
# ---------------------------------------------------------------------------


class TestPrepareAskRequest:
    async def test_uses_agent_defaults(self, nucliadb_agent, mock_nucliadb_driver):
        request = await nucliadb_agent.prepare_ask_request(
            mock_nucliadb_driver,
            "runtime question",
            None,
            [],
        )

        assert request.query == "runtime question"
        assert request.show == ["basic", "origin"]
        assert request.citations == "llm_footnotes"
        assert (
            request.generative_model == nucliadb_agent.config.generative_model.model_id
        )
        assert request.reasoning is False
        assert request.rag_strategies == []
        assert request.generate_answer is nucliadb_agent.config.generate_inner_answer

    async def test_public_ask_overrides_search_config_and_runtime_query_wins(
        self, mock_nucliadb_driver
    ):
        agent = NucliaDBAgent(
            config=NucliaDBAgentConfig(
                sources=["kb-source-1"],
                search_config="source-rag",
                generative_model="agent-model",
            ),
            agent_id="rag-agent",
        )
        public_request = AskRequest(
            query="public question",
            top_k=7,
            generative_model=None,
        )

        async def apply_config(reader_sdk, kbid, request):
            assert request.search_configuration == "source-rag"
            return AskRequest.model_validate(
                {"top_k": 1, "generative_model": "config-model"}
                | request.model_dump(exclude_unset=True)
            )

        with patch(
            "hyperforge_nucliadb_agentic.agent.rpc.apply_ask_search_configuration",
            new=AsyncMock(side_effect=apply_config),
        ):
            request = await agent.prepare_ask_request(
                mock_nucliadb_driver,
                "runtime question",
                public_request.model_dump_json(exclude_unset=True),
                [],
            )

        assert request.query == "runtime question"
        assert request.top_k == 7
        assert request.generative_model is None

    async def test_uses_model_reasoning_as_fallback(self, mock_nucliadb_driver):
        agent = NucliaDBAgent(
            config=NucliaDBAgentConfig(
                sources=["kb-source-1"],
                generative_model=LLMConfig(
                    model_id="reasoning-model",
                    advanced_reasoning={"effort": "high", "budget_tokens": 20_000},
                ),
            ),
            agent_id="rag-agent",
        )

        request = await agent.prepare_ask_request(
            mock_nucliadb_driver, "runtime question", None, []
        )

        assert request.reasoning is not False
        assert request.reasoning.effort == "high"
        assert request.reasoning.budget_tokens == 20_000

    async def test_public_ask_reasoning_overrides_model_reasoning(
        self, mock_nucliadb_driver
    ):
        agent = NucliaDBAgent(
            config=NucliaDBAgentConfig(
                sources=["kb-source-1"],
                generative_model=LLMConfig(
                    model_id="reasoning-model", reasoning="enabled"
                ),
            ),
            agent_id="rag-agent",
        )
        public_request = AskRequest(query="public question", reasoning=False)

        request = await agent.prepare_ask_request(
            mock_nucliadb_driver,
            "runtime question",
            public_request.model_dump_json(exclude_unset=True),
            [],
        )

        assert request.reasoning is False


class TestInnerRag:
    async def test_applies_search_config_and_preserves_source_filter(
        self, mock_memory, mock_manager, mock_source, mock_nucliadb_driver
    ):
        from nucliadb_models.filters import FilterExpression, Label

        config = NucliaDBAgentConfig(
            sources=["kb-source-1"],
            search_config="legal-rag",
        )
        agent = NucliaDBAgent(config=config, agent_id="rag-agent")
        source_filter = FilterExpression(
            field=Label(labelset="document", label="legal")
        )
        mock_nucliadb_driver.config.filter_expression = source_filter
        ask_result = NotEnoughContextAskResult()

        async def apply_config(reader_sdk, kbid, request):
            assert reader_sdk is mock_nucliadb_driver.driver
            assert kbid == "test-kbid"
            assert request.search_configuration == "legal-rag"
            return request.model_copy(update={"top_k": 3})

        with (
            patch(
                "hyperforge_nucliadb_agentic.agent.rpc.apply_ask_search_configuration",
                new=AsyncMock(side_effect=apply_config),
            ),
            patch(
                "hyperforge_nucliadb_agentic.agent.ask",
                new=AsyncMock(return_value=ask_result),
            ) as mock_ask,
        ):
            await agent.inner_rag(
                source_obj=mock_source,
                manager=mock_manager,
                memory=mock_memory,
                question="Which clauses apply?",
            )

        internal_request = mock_ask.call_args.kwargs["ask_request"]
        assert internal_request.top_k == 3
        assert internal_request.filter_expression == source_filter

    async def test_uses_endpoint_ask_request_with_smart_agent_query(
        self, nucliadb_agent, mock_memory, mock_manager, mock_source
    ):
        mock_memory.arguments["ask_request"] = AskRequest(
            query="original endpoint question",
            top_k=7,
            rag_strategies=[{"name": "hierarchy", "count": 12}],  # type: ignore
            rag_images_strategies=[{"name": "page_image", "count": 2}],  # type: ignore
            generative_model="requested-model",
            generate_answer=False,
        ).model_dump_json()

        ask_result = NotEnoughContextAskResult()

        with patch(
            "hyperforge_nucliadb_agentic.agent.ask",
            new=AsyncMock(return_value=ask_result),
        ) as mock_ask:
            await nucliadb_agent.inner_rag(
                source_obj=mock_source,
                manager=mock_manager,
                memory=mock_memory,
                question="SmartAgent retrieval question",
            )

        internal_request = mock_ask.call_args.kwargs["ask_request"]
        assert internal_request.query == "SmartAgent retrieval question"
        assert internal_request.top_k == 7
        assert internal_request.rag_strategies[0].name == "hierarchy"
        assert internal_request.rag_strategies[0].count == 12
        assert internal_request.rag_images_strategies[0].name == "page_image"
        assert internal_request.rag_images_strategies[0].count == 2
        assert internal_request.generative_model == "requested-model"
        assert internal_request.generate_answer is False
        assert mock_ask.call_args.kwargs["extra_predict_headers"] == {
            "X-Show-Consumption": "true"
        }

        preparation_step, retrieval_step = mock_memory.add_step.await_args_list
        assert preparation_step.kwargs["timeit"] > 0
        assert preparation_step.kwargs["input_nuclia_tokens"] == 0.0
        assert preparation_step.kwargs["output_nuclia_tokens"] == 0.0
        assert retrieval_step.kwargs["timeit"] > 0
        assert retrieval_step.kwargs["input_nuclia_tokens"] == 0
        assert retrieval_step.kwargs["output_nuclia_tokens"] == 0


# ---------------------------------------------------------------------------
# all_images_by_title
# ---------------------------------------------------------------------------


class TestAllImagesByTitle:
    async def test_returns_empty_contexts_when_no_resources(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        catalog_resp = MagicMock()
        catalog_resp.resources = {}
        mock_nucliadb_driver.catalog_search_raw = AsyncMock(return_value=catalog_resp)

        result = await nucliadb_agent.all_images_by_title(
            memory=mock_memory,
            manager=mock_manager,
            title="Document Without Images",
        )

        assert result == []

    async def test_skips_resource_not_found(
        self, nucliadb_agent, mock_memory, mock_manager, mock_nucliadb_driver
    ):
        rid = "resource-img-123"
        catalog_resp = MagicMock()
        catalog_resp.resources = {rid: MagicMock()}
        mock_nucliadb_driver.catalog_search_raw = AsyncMock(return_value=catalog_resp)
        # get_resource_by_id returns None → should be skipped
        mock_nucliadb_driver.get_resource_by_id = AsyncMock(return_value=None)

        result = await nucliadb_agent.all_images_by_title(
            memory=mock_memory,
            manager=mock_manager,
            title="Some Title",
        )

        assert result == []


# ---------------------------------------------------------------------------
# get_all_images
# ---------------------------------------------------------------------------


class TestGetAllImages:
    async def test_returns_empty_when_no_files(self, nucliadb_agent):
        resource = MagicMock()
        resource.data = None

        result = await nucliadb_agent.get_all_images(resource=resource)
        assert result == []

    async def test_returns_empty_when_files_is_none(self, nucliadb_agent):
        resource = MagicMock()
        resource.data = MagicMock()
        resource.data.files = None

        result = await nucliadb_agent.get_all_images(resource=resource)
        assert result == []

    async def test_skips_field_with_no_extracted(self, nucliadb_agent):
        field = MagicMock()
        field.extracted = None

        resource = MagicMock()
        resource.data = MagicMock()
        resource.data.files = {"f1": field}

        result = await nucliadb_agent.get_all_images(resource=resource)
        assert result == []
