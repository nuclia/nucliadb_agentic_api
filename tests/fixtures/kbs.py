import pytest
import asyncio
from hyperforge.fixtures import init_fixture
from nucliadb_sdk.tests.fixtures import NucliaFixture


Eric_Dataset = (
    "https://storage.googleapis.com/ncl-testbed-gcp-stage-1/test_nucliadb/eric.kb"
)


@pytest.fixture(scope="session")
def eric_dataset(nucliadb: NucliaFixture):
    kbid = asyncio.run(
        init_fixture(
            nucliadb,
            "conv",
            Eric_Dataset,
            "multilingual-2024-05-06",
            "gemini-2.5-flash-lite",
            kbid="00000000-0000-0000-0000-000000000002",
        )
    )
    yield kbid
