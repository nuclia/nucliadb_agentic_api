import asyncio

from hyperforge.broker import Broker
from hyperforge.server.session import SessionManager
from hyperforge.server.cache import Cache
from lru import LRU

from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from hyperforge.server.settings import Settings as ServerSettings


class NucliaDBAgenticSessionManager(SessionManager):
    def __init__(
        self,
        settings: ServerSettings,
        broker: Broker,
        agent_manager: AgenticConfigs,
        cache: Cache,
    ):
        self.settings = settings
        self.agent_manager = agent_manager
        self.broker = broker
        self.memory: LRU = LRU(800)
        self.activation_task: asyncio.Task | None = None
        self.tasks = []
        self.cache = cache

    self.hyperforge_drivers = {}
    if self.settings.hyperforge_google_key:
        self.hyperforge_drivers["google"] = GoogleDriver(
            api_key=settings.hyperforge_google_key
        )
    if self.settings.hyperforge_perplexity_key:
        self.hyperforge_drivers["perplexity"] = PerplexityDriver(
            api_key=settings.hyperforge_perplexity_key
        )
