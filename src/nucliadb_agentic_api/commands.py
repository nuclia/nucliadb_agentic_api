import uvicorn
from hyperforge import openapi
from hyperforge.feature_flag import get_flag_service
from nucliadb_telemetry.fastapi import instrument_app
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogFormatType, LogLevel, LogSettings
from nucliadb_telemetry.utils import get_telemetry
from nucliadb_utils.settings import AuditSettings

from nucliadb_agentic_api import SERVICE_NAME
from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.logging import set_sentry
from nucliadb_agentic_api.settings import Settings
from nucliadb_agentic_api.v1.router import router


def run():  # pragma: no cover
    settings = Settings()
    setup_logging(
        settings=LogSettings(
            log_format_type=LogFormatType.STRUCTURED
            if not settings.debug
            else LogFormatType.PLAIN,
            debug=settings.debug,
            log_level=LogLevel(settings.log_level),
            logger_levels={
                "uvicorn.error": LogLevel.ERROR,
                "nucliadb_telemetry": LogLevel.ERROR,
                "mcp.client.streamable_http": LogLevel.WARNING,
                "mcp.server.lowlevel.server": LogLevel.WARNING,
                "hyperforge.configure": LogLevel.WARNING,
            },
        )
    )
    if settings.sentry_url is not None:
        set_sentry(
            settings.zone,
            settings.running_environment,
            settings.sentry_url,
        )

    get_flag_service()  # precache the flag service

    data_manager_settings = DataManagerSettings()  # type: ignore
    audit_settings = AuditSettings()
    app = HTTPApplication(
        settings,
        data_manager_settings=data_manager_settings,
        audit_settings=audit_settings,
    )
    instrument_app(
        app,
        tracer_provider=get_telemetry(SERVICE_NAME),
        excluded_urls=["/", "/metrics", "/health/ready", "/health/alive"],
        metrics=True,
        trace_id_on_responses=True,
    )
    uvicorn.run(app, host=settings.http_host, port=settings.http_port)


def extract_openapi():
    openapi.extract_openapi_command(
        "nucliadb_agentic_api", "NucliaDB Agentic API", router
    )
