import datetime
from time import time

import databases
import sqlalchemy as sa
from hyperforge.database import metadata
from lru import LRU
from nucliadb_telemetry.utils import get_telemetry, init_telemetry
from pydantic import TypeAdapter
from sqlalchemy.dialects.postgresql import JSONB

from nucliadb_agentic_api import exceptions
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.models import SourceSchema

SERVICE_NAME = "SOURCES_DB"

_source_adapter: TypeAdapter = TypeAdapter(SourceSchema)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


sources_table = sa.Table(
    "sources_table",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("kbid", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("source_id", sa.String, primary_key=True, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("type", sa.String, nullable=False),
    sa.Column("config", JSONB, nullable=False),
)


CACHE: LRU = LRU(size=1024)


def _cache_key(account: str, kbid: str, source_id: str) -> str:
    return f"src:{account}:{kbid}:{source_id}:{int(time()) // 5}"


def _serialize_source(source: SourceSchema) -> dict:  # type: ignore[valid-type]
    return source.model_dump(mode="json")  # type: ignore[union-attr]


def _source_from_row(row) -> SourceSchema:  # type: ignore[valid-type]
    data = {
        "type": row["type"],
        "title": row["title"],
        "config": row["config"],
    }
    return _source_adapter.validate_python(data)


class Sources:
    settings: DataManagerSettings

    def __init__(
        self,
        database: databases.Database,
        settings: DataManagerSettings,
    ):
        self.database = database
        self.settings = settings

    @classmethod
    async def from_settings(cls, settings: DataManagerSettings) -> "Sources":
        tracer_provider = get_telemetry(SERVICE_NAME)
        if tracer_provider:
            await init_telemetry(tracer_provider)

        database = databases.Database(settings.postgresql_dsn)
        return cls(database=database, settings=settings)

    async def initialize(self) -> None:
        await self.database.connect()

    async def finalize(self) -> None:
        await self.database.disconnect()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_source(
        self,
        account: str,
        kbid: str,
        source_id: str,
        source: SourceSchema,  # type: ignore[valid-type]
    ) -> None:
        query = sa.select(sources_table.c.source_id).where(
            sources_table.c.account == account,
            sources_table.c.kbid == kbid,
            sources_table.c.source_id == source_id,
        )
        existing = await self.database.fetch_one(query)
        if existing is not None:
            raise exceptions.Conflict("Source already exists")

        serialized = _serialize_source(source)
        query = sa.insert(sources_table).values(
            account=account,
            kbid=kbid,
            source_id=source_id,
            title=serialized.get("title"),
            type=serialized["type"],
            config=serialized["config"],
        )
        await self.database.execute(query)
        CACHE[_cache_key(account, kbid, source_id)] = source

    async def get_source(
        self,
        account: str,
        kbid: str,
        source_id: str,
    ) -> SourceSchema:  # type: ignore[valid-type]
        key = _cache_key(account, kbid, source_id)
        if key not in CACHE:
            query = sa.select(sources_table).where(
                sources_table.c.account == account,
                sources_table.c.kbid == kbid,
                sources_table.c.source_id == source_id,
            )
            row = await self.database.fetch_one(query)
            if not row:
                raise exceptions.NotFound("Source not found")

            source = _source_from_row(row)
            CACHE[key] = source
        else:
            source = CACHE[key]
        return source

    async def patch_source(
        self,
        account: str,
        kbid: str,
        source_id: str,
        source: SourceSchema,  # type: ignore[valid-type]
    ) -> None:
        serialized = _serialize_source(source)
        query = (
            sa.update(sources_table)
            .values(
                title=serialized.get("title"),
                type=serialized["type"],
                config=serialized["config"],
            )
            .where(
                sources_table.c.account == account,
                sources_table.c.kbid == kbid,
                sources_table.c.source_id == source_id,
            )
            .returning(sources_table.c.source_id)
        )
        updated = await self.database.fetch_one(query)
        if updated is None:
            raise exceptions.NotFound("Source not found")
        CACHE[_cache_key(account, kbid, source_id)] = source

    async def delete_source(
        self,
        account: str,
        kbid: str,
        source_id: str,
    ) -> None:
        query = (
            sa.delete(sources_table)
            .where(
                sources_table.c.account == account,
                sources_table.c.kbid == kbid,
                sources_table.c.source_id == source_id,
            )
            .returning(sources_table.c.source_id)
        )
        deleted = await self.database.fetch_one(query)
        if deleted is None:
            raise exceptions.NotFound("Source not found")
        CACHE.pop(_cache_key(account, kbid, source_id), None)

    async def list_sources(
        self,
        account: str,
        kbid: str,
    ) -> dict[str, SourceSchema]:  # type: ignore[valid-type]
        query = sa.select(sources_table).where(
            sources_table.c.account == account,
            sources_table.c.kbid == kbid,
        )
        rows = await self.database.fetch_all(query)
        return {row["source_id"]: _source_from_row(row) for row in rows}
