"""
conftest.py — shared fixtures for the hyperforge_nucliadb_agentic test suite.

All fixtures are lightweight and use unittest.mock to avoid any real network
connections or external services.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperforge_nucliadb_agentic.config import NucliaDBAgentConfig


# ---------------------------------------------------------------------------
# Helpers / small data builders
# ---------------------------------------------------------------------------


def make_config(
    sources: List[str] | None = None,
    generative_model: str = "chatgpt-azure-4o-mini",
    generate_inner_answer: bool = True,
    **kwargs: Any,
) -> NucliaDBAgentConfig:
    """Return a minimal NucliaDBAgentConfig for testing."""
    return NucliaDBAgentConfig(
        sources=sources or ["kb-source-1"],
        generative_model=generative_model,
        generate_inner_answer=generate_inner_answer,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config() -> NucliaDBAgentConfig:
    """Default agent configuration used across tests."""
    return make_config()


@pytest.fixture
def multi_source_config() -> NucliaDBAgentConfig:
    """Agent configuration with multiple sources."""
    return make_config(sources=["kb-source-1", "kb-source-2"])


# ---------------------------------------------------------------------------
# NucliaDBAgent fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def nucliadb_agent(agent_config: NucliaDBAgentConfig):
    """A NucliaDBAgent instance constructed with the default test config."""
    from hyperforge_nucliadb_agentic.agent import NucliaDBAgent

    return NucliaDBAgent(config=agent_config, agent_id="test-agent-id")


# ---------------------------------------------------------------------------
# Mock QuestionMemory
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory():
    """A mock QuestionMemory that records add_step calls."""
    memory = MagicMock()
    memory.original_question_uuid = "test-question-uuid"
    memory.arguments = {}
    memory.add_step = AsyncMock()
    memory.get_tracking_info = MagicMock(return_value={})
    return memory


# ---------------------------------------------------------------------------
# Mock Manager + NucliaDBDriver
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_nucliadb_driver():
    """A mock NucliaDBDriver with sensible async defaults."""
    driver = MagicMock()

    # labels() returns an empty dict by default
    driver.labels = AsyncMock(return_value={})
    driver.synonyms_raw = AsyncMock(return_value={})
    driver.field_labels = AsyncMock(return_value=({}, 0))

    # catalog / find stubs
    driver.catalog_search_raw = AsyncMock(return_value=MagicMock(resources={}, fulltext=None))
    driver.find_raw = AsyncMock(return_value=MagicMock(resources={}))
    driver.get_resource_by_id = AsyncMock(return_value=None)
    driver.get_ephemeral_token = AsyncMock(return_value="fake-token")

    # config sub-object
    driver.config = MagicMock()
    driver.config.kbid = "test-kbid"
    driver.config.url = "http://nucliadb.test"
    driver.config.filters = None
    driver.config.filter_expression = None
    driver.config.catalog_filter_expression = None

    # The underlying SDK driver (used as search_sdk / reader_sdk in ask())
    driver.driver = MagicMock()

    return driver


@pytest.fixture
def mock_manager(mock_nucliadb_driver):
    """A mock Manager whose drivers dict maps the default source key."""
    manager = MagicMock()
    manager.drivers = {"kb-source-1": mock_nucliadb_driver}
    manager.execute = AsyncMock(return_value=("generated answer", 10, 5, "success"))
    manager.execute_json = AsyncMock(
        return_value=({"label_sets": [], "labels": [], "filters": "[]"}, 10, 5)
    )
    return manager


# ---------------------------------------------------------------------------
# Source stub
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_source():
    """A mock hyperforge Source."""
    source = MagicMock()
    source.id = "kb-source-1"
    return source
