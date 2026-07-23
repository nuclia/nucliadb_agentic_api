from contextlib import asynccontextmanager
from typing import Tuple

import prometheus_client
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hyperforge.api.authentication import RaoAuthenticationBackend
from hyperforge.broker import Broker
from hyperforge.broker.redis import RedisBroker
from hyperforge.driver import Driver
from hyperforge_nucliadb_agentic.ask.audit import (
    AuditMiddleware,
    start_audit_utility,
    stop_audit_utility,
)
from hyperforge_nucliadb_agentic.ask.predict import (
    start_predict_engine,
    stop_predict_engine,
)
from lru import LRU
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import (
    StreamableHTTPServerTransport,
)
from nucliadb_sdk import NucliaDBAsync
from nucliadb_telemetry.utils import clean_telemetry, setup_telemetry
from nucliadb_utils.settings import AuditSettings
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse

from nucliadb_agentic_api import SERVICE_NAME, v1
from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.db.sources import Sources
from nucliadb_agentic_api.settings import Settings

router = APIRouter()


@router.get("/metrics")
async def serve_metrics():  # pragma: no cover
    output = prometheus_client.exposition.generate_latest()
    return PlainTextResponse(
        output.decode("utf8"), headers={"Content-Type": CONTENT_TYPE_LATEST}
    )


@router.get("/health/ready")
async def health_ready():
    return {"status": "ok"}


@router.get("/health/alive")
async def health_alive():
    return {"status": "ok"}


class HTTPApplication(FastAPI):
    agent_manager: AgenticConfigs
    source_manager: Sources
    broker: Broker
    hyperforge_drivers: dict[str, "Driver"]

    def __init__(
        self,
        settings: Settings,
        data_manager_settings: DataManagerSettings,
        audit_settings: AuditSettings,
        *args,
        **kwargs,
    ):
        @asynccontextmanager
        async def lifespan(app: "HTTPApplication"):
            await app.startup()
            yield
            await app.shutdown()

        super().__init__(
            *args,
            lifespan=lifespan,
            # REVIEW: this is a patch to return to the previous behavior of
            # FastAPI that doesn't check content types. If all our internal
            # clients set headers properly, we wouldn't need that
            strict_content_type=False,
            **kwargs,
        )
        self.settings = settings
        self.data_manager_settings = data_manager_settings
        self.audit_settings = audit_settings
        self.include_router(v1.router)
        self.include_router(router)
        self.add_middleware(
            AuthenticationMiddleware,
            backend=RaoAuthenticationBackend(),
        )
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.add_middleware(AuditMiddleware)

    async def startup(self) -> None:
        await setup_telemetry(SERVICE_NAME)

        await start_predict_engine()
        await start_audit_utility(SERVICE_NAME, self.audit_settings)

        self.broker = RedisBroker.from_url(
            url=self.settings.valkey_url,
            activate_subject=self.settings.activate_subject,
            keepalive_ms=int(self.settings.pubsub_keepalive_seconds * 1000),
            cluster_mode=self.settings.valkey_cluster_mode,
        )
        if self.settings.internal_nucliadb:
            headers = {"X-NUCLIADB-ROLES": "READER"}
            api_key = None
            nucliadb_url = self.settings.internal_nucliadb_url
        else:
            nucliadb_url = self.settings.external_nucliadb_url
            api_key = self.settings.external_nucliadb_key
            headers = {}

        self.arag_reader = NucliaDBAsync(
            url=nucliadb_url,
            api_key=api_key,
            headers=headers,
        )
        self.arag_search = NucliaDBAsync(
            url=nucliadb_url,
            api_key=api_key,
            headers=headers,
        )

        self.sses: LRU[Tuple[str, str], StreamableHTTPServerTransport] = LRU(size=100)
        self.mcp_servers: LRU[str, MCPServer] = LRU(size=100)

        self.agent_manager = await AgenticConfigs.from_settings(
            settings=self.data_manager_settings
        )
        await self.agent_manager.initialize()

        self.source_manager = await Sources.from_settings(
            settings=self.data_manager_settings
        )
        await self.source_manager.initialize()

    async def shutdown(self) -> None:
        await self.agent_manager.finalize()
        await self.source_manager.finalize()
        await self.broker.finalize()
        await stop_audit_utility()
        await stop_predict_engine()
        await clean_telemetry(SERVICE_NAME)
