import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection

from nucliadb_agentic_api.db.agentic_configs import agentic_config_table
from nucliadb_agentic_api.db.llm_migration_utils import (
    migrate_llm_models,
    walk_and_replace,
)
from nucliadb_agentic_api.db.sources import sources_table


@pytest.fixture(params=["chatgpt-o3-mini", "chatgpt-o1"])
def old_model(request):
    return request.param


@pytest.fixture
def replacements(old_model):
    return {old_model: "chatgpt-5.6-sol"}


@pytest.fixture
def migration(old_model):
    root = Path(__file__).resolve().parents[2]
    scripts = ScriptDirectory.from_config(Config(str(root / "alembic.ini")))
    revision = scripts.get_revision(
        {
            "chatgpt-o3-mini": "50540732f342",
            "chatgpt-o1": "eff180d61535",
        }[old_model]
    )
    assert revision is not None
    return revision


def test_revision_follows_previous_head(migration, old_model):
    assert (
        migration.down_revision
        == {
            "chatgpt-o3-mini": "b3781ec6a8ad",
            "chatgpt-o1": "50540732f342",
        }[old_model]
    )
    assert migration.branch_labels == set()
    assert migration.dependencies is None


def test_upgrade_calls_helper_with_only_retired_model_mapping(
    migration, monkeypatch, replacements
):
    connection = Mock(spec=Connection)
    get_bind = Mock(return_value=connection)
    migrate = Mock()
    monkeypatch.setattr(migration.module.op, "get_bind", get_bind)
    monkeypatch.setattr(migration.module, "migrate_llm_models", migrate)

    migration.module.upgrade()

    get_bind.assert_called_once_with()
    migrate.assert_called_once_with(connection, replacements)


def test_downgrade_does_not_reverse_model_replacements(migration, monkeypatch):
    migrate = Mock()
    get_bind = Mock()
    monkeypatch.setattr(migration.module, "migrate_llm_models", migrate)
    monkeypatch.setattr(migration.module.op, "get_bind", get_bind)

    migration.module.downgrade()

    migrate.assert_not_called()
    get_bind.assert_not_called()


def test_replacement_recurses_through_objects_and_nested_lists(old_model, replacements):
    model = {"_type": "llm_config", "model_id": old_model, "temperature": 0.2}
    config = {
        "smart_agent": {
            "models": {"planner": copy.deepcopy(model)},
            "sources": [[{"tool_choice_model": copy.deepcopy(model)}]],
        },
        "summarize": {"model": copy.deepcopy(model)},
        "metadata": [None, 42, False, old_model],
    }
    expected = copy.deepcopy(config)
    expected["smart_agent"]["models"]["planner"]["model_id"] = "chatgpt-5.6-sol"
    expected["smart_agent"]["sources"][0][0]["tool_choice_model"]["model_id"] = (
        "chatgpt-5.6-sol"
    )
    expected["summarize"]["model"]["model_id"] = "chatgpt-5.6-sol"

    assert walk_and_replace(config, replacements)
    assert config == expected
    assert not walk_and_replace(config, replacements)


@pytest.mark.parametrize(
    "config",
    [
        {"model_id": "chatgpt-o3-mini"},
        {"_type": "custom", "model_id": "chatgpt-o3-mini"},
        {"_type": "LLMConfig", "model_id": "chatgpt-o3-mini"},
        {"_type": "llm_config", "model_id": "chatgpt-o3-mini/custom-id"},
        {"_type": "llm_config", "model_id": "custom-id/chatgpt-o3-mini"},
        {"_type": "llm_config", "model_id": "chatgpt-azure-o3-mini"},
        {"_type": "llm_config", "model_id": "chatgpt-o3-mini-extra"},
        {"model_id": "chatgpt-o1"},
        {"_type": "custom", "model_id": "chatgpt-o1"},
        {"_type": "LLMConfig", "model_id": "chatgpt-o1"},
        {"_type": "llm_config", "model_id": "chatgpt-o1/custom-id"},
        {"_type": "llm_config", "model_id": "custom-id/chatgpt-o1"},
        {"_type": "llm_config", "model_id": "chatgpt-azure-o1"},
        {"_type": "llm_config", "model_id": "chatgpt-o1-extra"},
        {"_type": "llm_config", "model_id": "chatgpt4o"},
        {"_type": "llm_config", "model_id": "chatgpt4o-mini"},
        {"_type": "llm_config", "model_id": "chatgpt-5.6-sol"},
        {"_type": "llm_config", "model_id": None},
        {"_type": "llm_config", "model_id": 42},
        {"_type": "llm_config"},
    ],
)
def test_replacement_preserves_other_models_and_discriminators(config, replacements):
    original = copy.deepcopy(config)

    assert not walk_and_replace(config, replacements)
    assert config == original


@pytest.mark.parametrize("serialize", [False, True])
def test_migration_updates_only_matching_agent_and_source_rows(
    serialize, old_model, replacements
):
    config = {
        "models": [{"_type": "llm_config", "model_id": old_model}],
        "custom": {"_type": "llm_config", "model_id": f"{old_model}/custom-id"},
    }
    expected = copy.deepcopy(config)
    expected["models"][0]["model_id"] = "chatgpt-5.6-sol"
    connection = Mock(spec=Connection)
    executions = []

    def execute(statement, params=None):
        sql = str(statement)
        executions.append((sql, params))
        if sql.startswith("SELECT"):
            stored = json.dumps(config) if serialize else copy.deepcopy(config)
            return Mock(
                fetchall=Mock(
                    return_value=[
                        ("account", "kbid", "changed-id", stored),
                        ("other-account", "kbid", "changed-id", expected),
                        ("account", "kbid", "null-id", None),
                    ]
                )
            )
        return Mock()

    connection.execute.side_effect = execute

    assert migrate_llm_models(connection, replacements) == 2
    assert len(executions) == 4
    for (select_sql, _), (update_sql, params), table, id_column in (
        (*executions[:2], "agentic_config_table", "agentic_id"),
        (*executions[2:], "sources_table", "source_id"),
    ):
        assert select_sql == (
            f"SELECT account, kbid, {id_column}, config FROM {table} "
            'WHERE config::text LIKE \'%"_type": "llm_config"%\''
        )
        assert update_sql == (
            f"UPDATE {table} SET config = CAST(:config AS jsonb) "
            f"WHERE account = :account AND kbid = :kbid AND {id_column} = :{id_column}"
        )
        assert params == {
            "account": "account",
            "kbid": "kbid",
            id_column: "changed-id",
            "config": json.dumps(expected),
        }
    connection.commit.assert_not_called()


@pytest.mark.parametrize(
    ("table", "id_column", "extra_values"),
    [
        (agentic_config_table, "agentic_id", {}),
        (sources_table, "source_id", {"type": "mcp"}),
    ],
)
def test_upgrade_persists_agent_and_source_configs(
    agentic_pg_dsn, migration, table, id_column, extra_values, old_model
):
    preserved = {
        "custom": {"_type": "llm_config", "model_id": f"{old_model}/custom-id"},
        "untyped": {"model_id": old_model},
        "other_type": {"_type": "custom", "model_id": old_model},
        "azure": {
            "_type": "llm_config",
            "model_id": old_model.replace("chatgpt-", "chatgpt-azure-"),
        },
    }
    config = {
        **preserved,
        "nested": [[{"model": {"_type": "llm_config", "model_id": old_model}}]],
    }
    expected = copy.deepcopy(config)
    expected["nested"][0][0]["model"]["model_id"] = "chatgpt-5.6-sol"
    row_id = f"{old_model}-migration"
    rows = [
        ("o3-account", "o3-kbid", config),
        ("other-account", "o3-kbid", preserved),
        ("o3-account", "other-kbid", preserved),
    ]
    engine = sa.create_engine(agentic_pg_dsn)
    try:
        with engine.begin() as conn:
            conn.execute(
                table.insert(),
                [
                    {
                        "account": account,
                        "kbid": kbid,
                        id_column: row_id,
                        "config": stored,
                        **extra_values,
                    }
                    for account, kbid, stored in rows
                ],
            )

        with (
            engine.begin() as conn,
            Operations.context(MigrationContext.configure(conn)),
        ):
            migration.module.upgrade()
            migration.module.upgrade()

        with engine.connect() as conn:
            stored_rows = conn.execute(
                sa.select(table.c.account, table.c.kbid, table.c.config).where(
                    table.c[id_column] == row_id
                )
            ).all()
        assert {(account, kbid): stored for account, kbid, stored in stored_rows} == {
            ("o3-account", "o3-kbid"): expected,
            ("other-account", "o3-kbid"): preserved,
            ("o3-account", "other-kbid"): preserved,
        }
    finally:
        with engine.begin() as conn:
            conn.execute(table.delete().where(table.c[id_column] == row_id))
        engine.dispose()
