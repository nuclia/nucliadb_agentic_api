import asyncio

from hyperforge import api
from hyperforge.broker import Broker
from hyperforge.engine import get_state
from hyperforge.interaction import ARAGException, AnswerOperation, AragAnswer
from hyperforge.pubsub import AgentDone, StartInteraction
from hyperforge.server.session import SessionManager
from hyperforge.server.cache import Cache

from lru import LRU

from nucliadb_agentic_api.ask.model import AskRequest
from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from hyperforge.server.settings import Settings as ServerSettings
from nucliadb_agentic_api import logger


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

    async def initialize(self, health_check: bool = True):
        await super().initialize(health_check)

    async def activate(self, message: StartInteraction):
        topic = None

        ask_request_json = message.arguments.get("ask_request")
        ask_request = None
        if ask_request_json:
            ask_request = AskRequest.model_validate(ask_request_json)

        logger.info("Activation message received: %s", message)
        observation = activation_observer()
        observation.start()
        try:
            nucliadb_telemetry.context.add_context(
                {
                    "agent_id": message.agent_id,
                    "session_id": message.session,
                    "question_id": message.question_id,
                }
            )

            topic = self.question_topic(
                message.account,
                message.agent_id,
                message.session,
                message.question_id,
                message.workflow_id,
            )

            # Get or load session
            config = await self.agent_manager.get_agent_config(
                account=message.account,
                agent_id=message.agent_id,
                internal_nucliadb_url=self.settings.internal_nucliadb_url,
                workflow_id=message.workflow_id,
                ask_request=ask_request,
            )

            state = await get_state(
                agent_id=message.agent_id,
                config=config,
                internal_nua_api=self.settings.internal_nua_api,
                internal_nua=self.settings.internal_nua,
                local_openai=self.settings.local_openai,
                external_nua_api_key=self.settings.external_nua_api_key,
                account=message.account,
                kbid=None if self.settings.standalone else message.agent_id,
            )

            if message.session not in self.memory:
                memory = await get_memory(
                    settings=self.settings,
                    session=message.session,
                    cache=self.cache,
                    config=config.memory,
                    agent=message.agent_id,
                    workflow_id=message.workflow_id,
                )
                self.memory[message.session] = memory
            else:
                memory = self.memory[message.session]

            memory.rules = config.rules.rules

            question = memory.start_question(
                message.question,
                question_id=message.question_id,
                headers=message.headers,
                arguments=message.arguments,
                streaming=message.streaming,
            )

            task = asyncio.create_task(
                self.answer(
                    message.account,
                    message.agent_id,
                    message.workflow_id,
                    topic,
                    state,
                    question,
                )
            )
            task.add_done_callback(self._remove_task)
            self.tasks.append(task)

        except Exception as e:
            logger.exception("Activation exception")
            errors.capture_exception(e)
            observation.set_status("error")
            if topic:
                await self.callback(
                    topic,
                    AragAnswer(
                        exception=ARAGException(detail="Unable to start agent"),
                        operation=AnswerOperation.ERROR,
                    ),
                )
                await self.send_message(topic, AgentDone())

        observation.end()


ask_request
