"""
test_config.py — unit tests for NucliaDBAgentConfig.
"""

import pytest
from pydantic import ValidationError

from hyperforge_nucliadb_agentic.config import NucliaDBAgentConfig


class TestNucliaDBAgentConfigDefaults:
    def test_module_is_always_nucliadb_agent(self):
        cfg = NucliaDBAgentConfig()
        assert cfg.module == "nucliadb_agent"

    def test_sources_default_is_empty(self):
        cfg = NucliaDBAgentConfig()
        assert cfg.sources == []

    def test_default_generative_model(self):
        cfg = NucliaDBAgentConfig()
        assert cfg.generative_model == "chatgpt-azure-4o-mini"

    def test_generate_inner_answer_defaults_to_true(self):
        cfg = NucliaDBAgentConfig()
        assert cfg.generate_inner_answer is True

    def test_published_functions_are_not_empty_by_default(self):
        cfg = NucliaDBAgentConfig()
        assert cfg.published_functions is not None
        assert len(cfg.published_functions) > 0


class TestNucliaDBAgentConfigCustomisation:
    def test_set_sources(self):
        cfg = NucliaDBAgentConfig(sources=["kb-1", "kb-2"])
        assert cfg.sources == ["kb-1", "kb-2"]

    def test_set_generative_model(self):
        cfg = NucliaDBAgentConfig(generative_model="gpt-4o")
        assert cfg.generative_model == "gpt-4o"

    def test_disable_inner_answer(self):
        cfg = NucliaDBAgentConfig(generate_inner_answer=False)
        assert cfg.generate_inner_answer is False

    def test_override_published_functions(self):
        cfg = NucliaDBAgentConfig(published_functions=("ask_agent",))
        assert cfg.published_functions == ("ask_agent",)

    def test_published_functions_can_be_none(self):
        cfg = NucliaDBAgentConfig(published_functions=None)
        assert cfg.published_functions is None


class TestNucliaDBAgentConfigKnownFunctions:
    """Verify the default published_functions tuple includes all expected names."""

    def test_ask_agent_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "ask_agent" in cfg.published_functions

    def test_ask_labels_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "ask_labels" in cfg.published_functions

    def test_ask_labels_list_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "ask_labels_list" in cfg.published_functions

    def test_search_by_title_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "search_by_title" in cfg.published_functions

    def test_facets_count_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "facets_count" in cfg.published_functions

    def test_facets_search_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "facets_search" in cfg.published_functions

    def test_catalog_search_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "catalog_search" in cfg.published_functions

    def test_all_images_by_title_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "all_images_by_title" in cfg.published_functions

    def test_search_images_is_published(self):
        cfg = NucliaDBAgentConfig()
        assert "search_images" in cfg.published_functions


class TestNucliaDBAgentConfigSerialization:
    def test_round_trip_json(self):
        cfg = NucliaDBAgentConfig(
            sources=["kb-a"],
            generative_model="gpt-4o",
            generate_inner_answer=False,
        )
        data = cfg.model_dump()
        restored = NucliaDBAgentConfig(**data)
        assert restored.sources == cfg.sources
        assert restored.generative_model == cfg.generative_model
        assert restored.generate_inner_answer == cfg.generate_inner_answer

    def test_model_config_title(self):
        assert NucliaDBAgentConfig.model_config["title"] == "Knowledge Box Agent"
