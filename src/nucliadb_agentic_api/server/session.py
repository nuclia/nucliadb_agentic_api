import asyncio
import os
from functools import partial

import nucliadb_telemetry.context
import nucliadb_telemetry.metrics
import prometheus_client
from hyperforge.broker import Broker
from hyperforge.configure import GLOBAL_REGISTRY, load_all_configurations, scan
from hyperforge.engine import State, get_state
from hyperforge.interaction import AnswerOperation, AragAnswer, ARAGException
from hyperforge.memory import QuestionMemory
from hyperforge.pubsub import AgentDone, StartInteraction
from hyperforge.server.cache import Cache
from hyperforge.server.session import SessionManager
from hyperforge.server.utils import get_memory
from hyperforge_nucliadb_agentic.ask.audit import (
    start_audit_utility,
    stop_audit_utility,
)
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from lru import LRU
from nucliadb_telemetry import errors
from nucliadb_telemetry.utils import get_telemetry
from nucliadb_utils.settings import AuditSettings
from opentelemetry import trace

from nucliadb_agentic_api import logger
from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.server import SERVICE_NAME
from nucliadb_agentic_api.server.settings import Settings as ServerSettings

HOSTNAME = os.environ.get("HOSTNAME", "nucliadb-agentic-api-server").encode()

answer_observer = nucliadb_telemetry.metrics.Observer("nucliadb_agentic_api_answer")
activation_observer = nucliadb_telemetry.metrics.Observer(
    "nucliadb_agentic_api_activation"
)
answer_running = prometheus_client.Gauge(
    "nucliadb_agentic_api_running_answers_count",
    "Number of answering processess currently running",
)


def tracer():
    provider = get_telemetry(SERVICE_NAME)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()


class NucliaDBAgenticSessionManager(SessionManager):
    agent_manager: AgenticConfigs  # type: ignore

    def __init__(
        self,
        settings: ServerSettings,
        audit_settings: AuditSettings,
        broker: Broker,
        agent_manager: AgenticConfigs,
        cache: Cache,
    ):
        self.settings = settings
        self.audit_settings = audit_settings
        self.agent_manager = agent_manager
        self.broker = broker
        self.memory: LRU = LRU(800)
        self.activation_task: asyncio.Task | None = None
        self.tasks = []
        self.cache = cache

    async def initialize(self, health_check: bool = True):
        await super().initialize(health_check)

        await start_audit_utility(SERVICE_NAME, self.audit_settings)

        for load_module in self.settings.load_modules:
            try:
                scan(load_module)
                load_all_configurations(load_module)
            except ImportError:
                logger.error(f"Module {load_module} could not be loaded")

    async def finalize(self):
        await super().finalize()
        await stop_audit_utility()
        GLOBAL_REGISTRY.clear()

    async def activate(self, message: StartInteraction):
        topic = None

        ask_request_json = message.arguments.get("ask_request")
        ask_request = None
        if ask_request_json:
            ask_request = AskRequest.model_validate_json(ask_request_json)

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
                kbid=message.agent_id,
                internal_nucliadb_url=self.settings.internal_nucliadb_url,
                internal_nucliadb=self.settings.internal_nucliadb,
                external_nucliadb_url=self.settings.external_nucliadb_url,
                external_nucliadb_key=self.settings.external_nucliadb_key,
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

    async def answer(
        self,
        account_id: str,
        agent_id: str,
        workflow_id: str,
        topic: str,
        state: State,
        question_memory: QuestionMemory,
    ):
        error = None

        keepalive = asyncio.create_task(self.keep_alive(topic))
        observation = answer_observer()
        observation.start()
        answer_running.inc()

        try:
            callback = partial(self.callback, topic)
            question_memory.set_callback_fn(callback)

            feedback = partial(self.feedback, topic)
            question_memory.set_feedback_fn(feedback)

            oauth = partial(self.oauth, topic)
            question_memory.set_oauth_fn(oauth)

            oauth_callback = partial(
                self.get_oauth_callback,
                account_id,
                agent_id,
                question_memory.session.id,
                workflow_id,
            )
            question_memory.set_oauth_callback_fn(oauth_callback)

            await self.callback(
                topic,
                AragAnswer(operation=AnswerOperation.START),
            )

            async with asyncio.timeout(self.settings.question_timeout_seconds):
                await state.agent(question_memory, state.manager)

        except Exception as e:
            logger.exception("Answering exception")
            errors.capture_exception(e)
            error = ARAGException(detail=str(e))
            observation.set_status("error")

        observation.end()
        answer_running.dec()
        keepalive.cancel()

        await self.callback(
            topic,
            AragAnswer(
                exception=error,
                answer=question_memory.final_answer,
                answer_citations=question_memory.final_answer_citations,
                answer_urls=question_memory.final_answer_urls,
                operation=AnswerOperation.ERROR
                if error is not None
                else AnswerOperation.ANSWER,
                data_visualizations=question_memory.data_visualizations
                if question_memory.data_visualizations
                else None,
            ),
        )
        await self.send_message(
            topic,
            AgentDone(),
        )

        try:
            await question_memory.save()
            self.process_event(
                "memory_saved",
                {"account_id": account_id, "question_memory": question_memory},
            )
        except Exception as e:
            # Log memory errors but don't report them to the user
            logger.exception("Error saving memory")
            errors.capture_exception(e)
