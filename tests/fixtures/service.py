import os
from collections.abc import AsyncGenerator
from typing import Tuple

import pytest
from hyperforge.broker.redis import RedisBroker
from hyperforge.server.cache import ValkeyCache
from nucliadb_sdk.tests.fixtures import NucliaFixture
from nucliadb_utils.settings import AuditSettings
from redis.asyncio import Redis

from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager
from nucliadb_agentic_api.server.settings import Settings as ServerSettings

NUA = os.environ.get("NUA_KEY", "DUMMY")


@pytest.fixture(scope="function")
async def nucliadb_agentic_api_server(
    valkey: Tuple[str, str],
    agentic_configs_db_server: AgenticConfigs,
    nucliadb: NucliaFixture,
    nucliadb_agentic_audit_settings: AuditSettings,
    disable_safe_transport,
) -> AsyncGenerator[NucliaDBAgenticSessionManager, None]:

    valkey_host, valkey_port = valkey
    valkey_url = f"redis://{valkey_host}:{valkey_port}"
    settings = ServerSettings(
        valkey_url=valkey_url,
        valkey_cluster_mode=False,
        internal_nucliadb=True,
        internal_nucliadb_url=nucliadb.url.replace("127.0.0.1", "localhost"),
        internal_nua=False,
        local_openai=None,
        pubsub_keepalive_seconds=40,
        external_nua_api_key=NUA,
        activate_subject="test_activate",
        answers_subject="test_agentic.{account}.{agent_id}.{workflow_id}.{session}.{question}",
        oauth_subject="test_oauth_agentic.{account}.{agent_id}.{workflow_id}.{session}.{question}",
        health_check_enabled=False,
    )
    broker = RedisBroker.from_url(
        url=valkey_url,
        activate_subject=settings.activate_subject,
        keepalive_ms=int(settings.pubsub_keepalive_seconds * 1000),
        cluster_mode=settings.valkey_cluster_mode,
    )
    session = NucliaDBAgenticSessionManager(
        settings=settings,
        broker=broker,
        agent_manager=agentic_configs_db_server,
        cache=ValkeyCache(Redis(host=valkey_host, port=int(valkey_port))),
        audit_settings=nucliadb_agentic_audit_settings,
    )
    await session.initialize()
    yield session
    await session.finalize()
