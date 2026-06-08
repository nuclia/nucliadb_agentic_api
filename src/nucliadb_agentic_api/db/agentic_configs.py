import datetime
import json
from time import time

import databases
import sqlalchemy as sa
from hyperforge.database import metadata
from hyperforge.retrieval.config import RetrievalAgentConfig
from hyperforge_google.config import GoogleDriverConfig, GoogleInnerConfig
from hyperforge_perplexity.config import PerplexityDriverConfig, PerplexityInnerConfig
from lru import LRU
from nucliadb_telemetry.utils import get_telemetry, init_telemetry
from sqlalchemy.dialects.postgresql import JSONB

from nucliadb_agentic_api import exceptions
from nucliadb_agentic_api.ask.model import AskRequest
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.db.transform import transform_agentic_config
from nucliadb_agentic_api.models import AgenticConfigSchema, AgenticConfiguration


# Imported lazily in methods to avoid any load-order sensitivity between the two
# table modules (both share the same `hyperforge.database.metadata`).
def _get_sources_table():  # pragma: no cover
    from nucliadb_agentic_api.db.sources import sources_table  # noqa: PLC0415

    return sources_table


SERVICE_NAME = "AGENTIC_CONFIGS_DB"


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


agentic_config_table = sa.Table(
    "agentic_config_table",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("kbid", sa.String, primary_key=True, nullable=False, index=True),  # KBID
    sa.Column("agentic_id", sa.String, primary_key=True, nullable=False),  # Agentic ID
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("config", JSONB, nullable=False),
)


CACHE = LRU(size=1024)


def _cache_key(account: str, kbid: str, agentic_id: str) -> str:
    return f"{account}:{kbid}:{agentic_id}:{int(time()) // 5}"


def _collect_source_ids(config: AgenticConfigSchema) -> list[str]:
    """Return all non-None source_id values declared in smart_agent sources."""
    if not config.config.smart_agent:
        return []
    return [
        source.source_id
        for source in config.config.smart_agent.sources
        if source.source_id is not None
    ]


def _serialize_config(config: AgenticConfigSchema) -> dict:
    return config.model_dump(mode="json")


def _config_from_row(row) -> AgenticConfigSchema:
    return AgenticConfigSchema(
        title=row["title"],
        config=AgenticConfiguration.model_validate(row["config"]),
    )


class AgenticConfigs:
    settings: DataManagerSettings

    def __init__(
        self,
        database: databases.Database,
        settings: DataManagerSettings,
    ):
        self.database = database
        self.settings = settings

    @classmethod
    async def from_settings(
        cls,
        settings: DataManagerSettings,
    ):
        tracer_provider = get_telemetry(SERVICE_NAME)
        if tracer_provider:
            await init_telemetry(tracer_provider)

        database = databases.Database(settings.postgresql_dsn)

        return cls(database=database, settings=settings)

    async def initialize(self):
        await self.database.connect()

    async def finalize(self):
        await self.database.disconnect()

    async def patch_agentic_config(
        self, account: str, kbid: str, agentic_id: str, config: AgenticConfigSchema
    ):
        source_ids = _collect_source_ids(config)
        if source_ids:
            await self._validate_sources_exist(account, kbid, source_ids)

        query = (
            sa.update(agentic_config_table)
            .values(
                title=config.title,
                config=_serialize_config(config)["config"],
            )
            .where(
                agentic_config_table.c.account == account,
                agentic_config_table.c.kbid == kbid,
                agentic_config_table.c.agentic_id == agentic_id,
            )
            .returning(agentic_config_table.c.agentic_id)
        )
        updated = await self.database.fetch_one(query)
        if updated is None:
            raise exceptions.NotFound("Agentic configuration not found")
        CACHE[_cache_key(account, kbid, agentic_id)] = config

    async def create_agentic_config(
        self, account: str, kbid: str, agentic_id: str, config: AgenticConfigSchema
    ):
        source_ids = _collect_source_ids(config)
        if source_ids:
            await self._validate_sources_exist(account, kbid, source_ids)

        query = sa.select(agentic_config_table.c.agentic_id).where(
            agentic_config_table.c.account == account,
            agentic_config_table.c.kbid == kbid,
            agentic_config_table.c.agentic_id == agentic_id,
        )
        existing = await self.database.fetch_one(query)
        if existing is not None:
            raise exceptions.Conflict("Agentic configuration already exists")

        query = sa.insert(agentic_config_table).values(
            account=account,
            kbid=kbid,
            agentic_id=agentic_id,
            title=config.title,
            config=_serialize_config(config)["config"],
        )
        await self.database.execute(query)
        CACHE[_cache_key(account, kbid, agentic_id)] = config

    async def _validate_sources_exist(
        self, account: str, kbid: str, source_ids: list[str]
    ) -> None:
        """Raise InvalidReference if any source_id is not present in sources_table."""
        sources_table = _get_sources_table()
        query = sa.select(sources_table.c.source_id).where(
            sources_table.c.account == account,
            sources_table.c.kbid == kbid,
            sources_table.c.source_id.in_(source_ids),
        )
        rows = await self.database.fetch_all(query)
        found = {row["source_id"] for row in rows}
        missing = sorted(set(source_ids) - found)
        if missing:
            raise exceptions.InvalidReference(
                f"Source(s) not found: {', '.join(missing)}"
            )

    async def delete_configs_referencing_source(
        self, account: str, kbid: str, source_id: str
    ) -> int:
        """Delete every agentic config that references source_id in its smart_agent
        sources list.  Returns the number of deleted configs.

        Uses the JSONB containment operator (@>) to find matching rows:
            config->'smart_agent'->'sources' @> '[{"source_id": "<id>"}]'
        """
        query = (
            sa.delete(agentic_config_table)
            .where(
                agentic_config_table.c.account == account,
                agentic_config_table.c.kbid == kbid,
                agentic_config_table.c.config["smart_agent"]["sources"].op("@>")(
                    sa.cast(
                        json.dumps([{"source_id": source_id}]),
                        JSONB,
                    )
                ),
            )
            .returning(agentic_config_table.c.agentic_id)
        )
        deleted_rows = await self.database.fetch_all(query)
        for row in deleted_rows:
            key = _cache_key(account, kbid, row["agentic_id"])
            if key in CACHE:
                del CACHE[key]
        return len(deleted_rows)

    async def get_agentic_config(
        self, account: str, kbid: str, agentic_id: str
    ) -> AgenticConfigSchema:
        key = _cache_key(account, kbid, agentic_id)
        if key not in CACHE:
            query = sa.select(agentic_config_table).where(
                agentic_config_table.c.account == account,
                agentic_config_table.c.kbid == kbid,
                agentic_config_table.c.agentic_id == agentic_id,
            )
            row = await self.database.fetch_one(query)
            if not row:
                raise exceptions.NotFound("Agentic configuration not found")

            config = _config_from_row(row)

            CACHE[key] = config
        else:
            config = CACHE[key]
        return config

    async def list_agentic_configs(
        self, account: str, kbid: str
    ) -> dict[str, AgenticConfigSchema]:
        query = sa.select(agentic_config_table).where(
            agentic_config_table.c.account == account,
            agentic_config_table.c.kbid == kbid,
        )
        rows = await self.database.fetch_all(query)
        return {row["agentic_id"]: _config_from_row(row) for row in rows}

    async def get_agent_config(
        self,
        account: str,
        agent_id: str,
        internal_nucliadb_url: str | None = None,
        default_memory: bool = False,
        workflow_id: str = "default",
        ask_request: AskRequest | None = None,
    ) -> RetrievalAgentConfig:
        # For now, we only support one config per KB, so we ignore agent_id and workflow_id, but in the future we can extend this method to support multiple configs per KB and select
        # the right one based on these parameters
        retrieval_config: RetrievalAgentConfig
        agentic_config = await self.get_agentic_config(account, agent_id, workflow_id)
        global_drivers = {}
        retrieval_config, drivers = await transform_agentic_config(
            agentic_config, global_drivers, ask_request, agent_id
        )

        retrieval_config.drivers.append(
            GoogleDriverConfig(
                identifier="google",
                name="google",
                config=GoogleInnerConfig(
                    api_key=self.settings.hyperforge_google_key, vertexai=False
                ),
            )
        )

        retrieval_config.drivers.append(
            PerplexityDriverConfig(
                identifier="perplexity",
                name="perplexity",
                provider="perplexity",
                config=PerplexityInnerConfig(
                    key=self.settings.hyperforge_perplexity_key
                ),
            )
        )
        return retrieval_config

    async def ensure_workflow_active(
        self, account: str, agent_id: str, workflow_id: str
    ):
        # For now, we only support one config per KB, so we ignore agent_id and workflow_id, but in the future we can extend this method to support multiple configs per KB and select
        # the right one based on these parameters
        await self.get_agentic_config(account, agent_id, workflow_id)
