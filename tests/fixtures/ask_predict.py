import os

import pytest
from hyperforge.minimal_fixtures import cassette_nua_key
from nucliadb_utils.settings import nuclia_settings

NUA_KEY = os.environ.get("NUA_KEY") or cassette_nua_key(
    "https://europe-1.dp.progress.cloud/"
)


@pytest.fixture(scope="session")
async def ask_predict_configure():
    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = "https://europe-1.dp.progress.cloud"
    nuclia_settings.nuclia_service_account = NUA_KEY
    nuclia_settings.nuclia_zone = "europe-1.dp.progress.cloud"
