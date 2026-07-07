from contextlib import asynccontextmanager

from fastapi import FastAPI
from nucliadb_telemetry.utils import clean_telemetry, setup_telemetry

from hyperforge_nucliadb_agentic.ask import SERVICE_NAME
from hyperforge_nucliadb_agentic.ask.predict import (
    start_predict_engine,
    stop_predict_engine,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_telemetry(SERVICE_NAME)
    await start_predict_engine()

    yield

    await stop_predict_engine()
    await clean_telemetry(SERVICE_NAME)
