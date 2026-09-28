import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

AGENTIC_LLM_TABLES = [
    ("agentic_config_table", "config", ("account", "kbid", "agentic_id")),
    ("sources_table", "config", ("account", "kbid", "source_id")),
]


def walk_and_replace(obj: Any, replacements: dict[str, str]) -> bool:
    """Replace exact model IDs in recursively nested LLMConfig objects."""
    modified = False
    if isinstance(obj, dict):
        model_id = obj.get("model_id")
        if (
            obj.get("_type") == "llm_config"
            and isinstance(model_id, str)
            and "/" not in model_id
            and model_id in replacements
        ):
            obj["model_id"] = replacements[model_id]
            modified = True

        for value in obj.values():
            modified = walk_and_replace(value, replacements) or modified
    elif isinstance(obj, list):
        for value in obj:
            modified = walk_and_replace(value, replacements) or modified
    return modified


def migrate_llm_models(conn: Connection, replacements: dict[str, str]) -> int:
    """Migrate LLMConfig model IDs in agentic and source JSONB configs."""
    modified_rows = 0
    for table, column, key_columns in AGENTIC_LLM_TABLES:
        selected_columns = ", ".join((*key_columns, column))
        rows = conn.execute(
            text(
                f"SELECT {selected_columns} FROM {table} "
                f'WHERE {column}::text LIKE \'%"_type": "llm_config"%\''
            )
        ).fetchall()
        for row in rows:
            config = row[-1]
            if isinstance(config, str):
                config = json.loads(config)
            if config is None or not walk_and_replace(config, replacements):
                continue

            where = " AND ".join(f"{key} = :{key}" for key in key_columns)
            params = dict(zip(key_columns, row[:-1], strict=True))
            params["config"] = json.dumps(config)
            conn.execute(
                text(
                    f"UPDATE {table} SET {column} = CAST(:config AS jsonb) "
                    f"WHERE {where}"
                ),
                params,
            )
            modified_rows += 1
    return modified_rows
