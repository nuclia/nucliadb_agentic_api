import json

pytest_plugins = [
    "pytest_docker_fixtures",
    "pytest_mock",
    # fixtures from dependencies
    "nuclia.tests.fixtures",
    "nucliadb_sdk.tests.fixtures",
    "nucliadb_utils.tests.fixtures",
    "nucliadb_utils.tests.nats",
    "nucliadb_utils.tests.gcs",
    "nucliadb_utils.tests.s3",
    "nucliadb_utils.tests.azure",
    "nucliadb_utils.tests.local",
    "nucliadb_telemetry.tests.telemetry",
    # our own fixtures
    "hyperforge.fixtures",
    "tests.fixtures.standalone",
    "tests.fixtures.api",
    "tests.fixtures.arag_ask",
    "tests.fixtures.audit",
    "tests.fixtures.predict",
    "tests.fixtures.db",
    "tests.fixtures.service",
    "tests.fixtures.knowledge_graph",
    "tests.fixtures.knowledgebox",
    "tests.fixtures.ask_predict",
    "tests.fixtures.kbs",
]


def nua_chat_match(r1, r2):
    if r1.uri == "https://europe-1.dp.progress.cloud/api/v1/predict/chat":
        if r2.uri == "https://europe-1.dp.progress.cloud/api/v1/predict/chat":
            r1_payload = json.loads(r1.body)
            r2_payload = json.loads(r2.body)
            assert r1_payload["question"] == r2_payload["question"], (
                "Questions do not match"
            )
        else:
            return False
    return True


def pytest_recording_configure(config, vcr):
    vcr.register_matcher("nua_chat", nua_chat_match)
