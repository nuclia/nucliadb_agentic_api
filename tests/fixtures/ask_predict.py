import os

import pytest
from hyperforge.llm import NUAConnection
from nuclia import REGIONAL
from nucliadb_utils.settings import nuclia_settings

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")


@pytest.fixture(scope="session")
async def ask_predict_configure():

    nua_driver = await NUAConnection.model_validate(
        {
            "key": NUA_KEY,
        }
    ).connect()
    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = NUA_KEY
    nuclia_settings.nuclia_zone = nua_driver.region
