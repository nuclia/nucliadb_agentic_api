import os

import pytest
from hyperforge.minimal_fixtures import cassette_nua_key
from nuclia import REGIONAL
from nuclia.lib.nua import AsyncNuaClient
from nucliadb_utils.settings import nuclia_settings

NUA_KEY = os.environ.get("NUA_KEY") or cassette_nua_key(
    "https://europe-1.dp.progress.cloud/"
)


@pytest.fixture(scope="session")
async def ask_predict_configure():

    nua_driver = AsyncNuaClient(token=NUA_KEY, account="nuclia", region="europe-1")
    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = NUA_KEY
    nuclia_settings.nuclia_zone = nua_driver.region
