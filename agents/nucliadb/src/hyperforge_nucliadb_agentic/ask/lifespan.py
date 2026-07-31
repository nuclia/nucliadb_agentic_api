from contextlib import asynccontextmanager

from fastapi import FastAPI
from nucliadb_telemetry.utils import clean_telemetry, setup_telemetry

from hyperforge_nucliadb_agentic.ask import SERVICE_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_telemetry(SERVICE_NAME)

    yield

    await clean_telemetry(SERVICE_NAME)
