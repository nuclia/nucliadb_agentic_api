# Copyright (C) 2021 Bosutech XXI S.L.
#
# nucliadb is offered under the AGPL v3.0 and as commercial software.
# For commercial licensing, contact us at info@nuclia.com.
#
# AGPL:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import asyncio
from unittest.mock import patch

import nats
import nats.errors
import nats.js.errors
import pytest
from httpx import AsyncClient
from hyperforge.feature_flag import get_flag_service
from hyperforge.models import ExternalUsage, ExternalUsageOperation, Step
from hyperforge_nucliadb_agentic.ask.audit import (
    StreamAuditStorage,
    external_usage_to_predict,
)
from hyperforge_nucliadb_agentic.ask.model import AskRequest
from nats.aio.client import Client
from nats.js import JetStreamContext
from nucliadb_models.search import NucliaDBClientType
from nucliadb_protos.audit_pb2 import AuditRequest
from nucliadb_protos.kb_usage_pb2 import (
    ActivityLogMatchType,
    KBSource,
    PredictType,
    Service,
)
from nucliadb_utils.settings import audit_settings


async def get_audit_messages(sub):
    msg = await sub.fetch(1)
    auditreq = AuditRequest()
    auditreq.ParseFromString(msg[0].data)
    return auditreq


def test_external_usage_to_predict() -> None:
    predicts = external_usage_to_predict(
        ExternalUsage(
            operation=ExternalUsageOperation.INTERNET_SEARCH,
            provider="google",
            model="gemini-2.5-flash",
            input_tokens=10,
            output_tokens=20,
            requests=2,
        ),
        NucliaDBClientType.API,
    )

    assert len(predicts) == 1
    search = predicts[0]
    assert search.type == PredictType.INTERNET_SEARCH
    assert search.external_requests == 2
    assert search.model == "google"
    assert search.input == 10
    assert search.output == 20
    assert search.image == 0


def test_search_only_external_usage_to_predict() -> None:
    predicts = external_usage_to_predict(
        ExternalUsage(
            operation=ExternalUsageOperation.INTERNET_SEARCH,
            provider="perplexity",
            model="search",
        ),
        NucliaDBClientType.API,
    )

    assert len(predicts) == 1
    assert predicts[0].type == PredictType.INTERNET_SEARCH
    assert predicts[0].model == "perplexity"
    assert predicts[0].external_requests == 1
    assert predicts[0].input == 0
    assert predicts[0].output == 0
    assert predicts[0].image == 0


async def test_external_usage_is_reported(
    audit: StreamAuditStorage,
    knowledgebox: str,
) -> None:
    step = Step(
        original_question_uuid="question",
        actual_question_uuid="question",
        module="google",
        title="Search results",
        timeit=0.1,
        input_nuclia_tokens=None,
        output_nuclia_tokens=None,
        agent_path="/context/google",
        external_usage=[
            ExternalUsage(
                operation=ExternalUsageOperation.INTERNET_SEARCH,
                provider="google",
                model="gemini-2.5-flash",
                input_tokens=10,
                output_tokens=20,
            )
        ],
    )

    assert audit.kb_usage_utility is not None
    audit.report_step_usage(
        account_id="account",
        kbid=knowledgebox,
        client_type=NucliaDBClientType.API,
        step=step,
        trace_id="trace-id",
    )

    usage = audit.kb_usage_utility.queue.get_nowait()
    assert usage.service == Service.RAO
    assert usage.account_id == "account"
    assert usage.kb_id == knowledgebox
    assert usage.kb_source == KBSource.HOSTED
    assert usage.activity_log_match.id == "trace-id"
    assert usage.activity_log_match.type == ActivityLogMatchType.TRACE_ID
    assert len(usage.predicts) == 1
    search = usage.predicts[0]
    assert search.client == 0
    assert search.type == PredictType.INTERNET_SEARCH
    assert search.model == "google"
    assert search.external_requests == 1
    assert search.input == 10
    assert search.output == 20
    assert search.image == 0
    assert search.customer_key is False


@pytest.mark.deploy_modes("standalone")
async def test_ask_sends_only_one_audit(
    audit: StreamAuditStorage,
    nucliadb_agentic_ask_api: AsyncClient,
    knowledgebox: str,
    resource: str,
) -> None:
    kbid = knowledgebox

    with patch.object(get_flag_service(), "enabled", return_value=True):
        # Prepare a test audit stream to receive our messages
        partition = audit.get_partition(kbid)
        nats_client: Client = await nats.connect(audit.nats_servers)
        jetstream: JetStreamContext = nats_client.jetstream()
        if audit_settings.audit_jetstream_target is None:
            assert False, "Missing jetstream target in audit settings"
        subject = audit_settings.audit_jetstream_target.format(
            partition=partition, type="*"
        )

        try:
            await jetstream.delete_stream(name=audit_settings.audit_stream)
            await jetstream.delete_stream(name="test_usage")
        except nats.js.errors.NotFoundError:
            pass

        await jetstream.add_stream(name=audit_settings.audit_stream, subjects=[subject])

        psub = await jetstream.pull_subscribe(subject, "psub")

        resp = await nucliadb_agentic_ask_api.post(
            f"/kb/{kbid}/ask",
            json={"query": "title"},
        )
        assert resp.status_code == 200

        # Wait until audit and kb usage finish sending messages. This is
        # required as some times asyncio is funny and we run the asserts before
        # waiting for the message to be sent.
        #
        # Calling .join() on the queues doesn't work as we may call it when the
        # message has been taken from the queue but not yet processed.
        await asyncio.sleep(1)

        # Testing the middleware integration where it collects audit calls and sends a single message
        # at requests ends. In this case we expect one retrieve and one ask calls and sent once
        audit.retrieve.assert_called_once()  # type: ignore
        audit.ask.assert_called_once()  # type: ignore
        assert audit.js.publish.call_count == 1  # type: ignore
        audit.send.assert_called_once()  # type: ignore

        auditreq = await get_audit_messages(psub)
        assert auditreq.type == AuditRequest.AuditType.ASK
        assert auditreq.kbid == kbid
        assert AskRequest.model_validate_json(auditreq.user_request) == AskRequest(
            query="title"
        )
        assert auditreq.HasField("chat")
        assert auditreq.HasField("search")
        assert auditreq.request_time > 0
        assert auditreq.generative_answer_time > 0
        assert auditreq.retrieval_time > 0
        assert (
            auditreq.generative_answer_time + auditreq.retrieval_time
        ) < auditreq.request_time
        try:
            auditreq = await get_audit_messages(psub)
        except nats.errors.TimeoutError:
            pass
        else:
            assert "There was an unexpected extra audit message in nats"
        await psub.unsubscribe()
        await nats_client.flush()
        await nats_client.close()
