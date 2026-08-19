import json
from typing import Any, Dict

import pytest
from httpx import AsyncClient
from hyperforge.api.models import InteractionOperation
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
)
from hyperforge_nucliadb_agentic.ask.audit import StreamAuditStorage
from pytest_mock import MockerFixture
from websockets.asyncio.client import connect

from nucliadb_agentic_api.server.session import NucliaDBAgenticSessionManager

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,  # Ignore localhost requests (e.g., to the test server)
        match_on=["scheme", "host", "port", "path", "nua_chat"],
    ),
    pytest.mark.asyncio,
]


async def test_agentic_websocket_nucliadb(
    nucliadb_agentic_api_http: str,
    nucliadb_agentic_api_http_client: AsyncClient,
    nucliadb_agentic_api_server: NucliaDBAgenticSessionManager,
    eric_dataset: str,
):
    # In this basic test we just want to verify that the ask endpoint is working end-to-end with a simple question, without any agentic config.
    # We use a dataset with a single article to have a predictable answer.

    payload: Dict[str, Any] = {
        "type": "nucliadb",
        "description": "Information about movies and actors.",
    }

    # Create
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/full-kb",
        json=payload,
        headers={"X-NUCLIADB-ACCOUNT": "nuclia"},
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "title": "Agentic config with NucliaDB source",
        "rephrase": {},
        "smart_agent": {"mode": "reactive", "sources": ["full-kb"]},
        "summarize": {},
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/agentic_configs/key-nucliadb",
        json=payload,
        headers={"X-NUCLIADB-ACCOUNT": "nuclia"},
    )
    assert resp.status_code == 201, resp.text

    headers = {
        "X-NUCLIADB-ROLES": "READER",
        "X-NUCLIADB-USER": "user1",
        "X-STF-USER": "user1",
        "X-NUCLIADB-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
        "X-NUCLIADB-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{nucliadb_agentic_api_http}/api/v1/kb/{eric_dataset}/ask?agentic_config_id=key-nucliadb&keep_open=true",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "Who is Carrie Fisher mother and what is she known for?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        answers = []
        async for message in websocket:
            print(message)
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.ANSWER:
                if response.answer:
                    answers.append(response.answer)
            elif response.operation == AnswerOperation.DONE:
                break
            elif response.operation == AnswerOperation.ERROR:
                assert False, (
                    f"Interaction error: {response.exception.detail if response.exception else ''}"
                )
            else:
                print(
                    "No feedback, step, possible_answer, context or generated_text in response"
                )

        assert "Debbie" in answers[0]


async def test_agentic_websocket_perplexity(
    nucliadb_agentic_api_http: str,
    nucliadb_agentic_api_http_client: AsyncClient,
    nucliadb_agentic_api_server: NucliaDBAgenticSessionManager,
    eric_dataset: str,
    mocker: MockerFixture,
):
    # In this basic test we just want to verify that the ask endpoint is working end-to-end with a simple question, without any agentic config.
    # We use a dataset with a single article to have a predictable answer.

    payload = {
        "type": "nucliadb",
        "description": "Our favorite pastries recipes",
        "filter_expression": {
            "field": {
                "prop": "resource_mimetype",
                "type": "application",
                "subtype": "pdf",
            }
        },
    }

    # Create
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/recipes-kb",
        json=payload,
        headers={
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "nuclia",
        },
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "type": "perplexity",
        "description": "Other recipes on internet",
        "domains": ["https://www.allrecipes.com"],
    }

    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/sources/perplexity",
        json=payload,
        headers={
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "nuclia",
        },
    )
    assert resp.status_code == 201, resp.text

    payload = {
        "smart_agent": {
            "mode": "reactive",
            "sources": ["recipes-kb", "perplexity"],
            "extra_prompt": "When asked about a recipe, always recommend the most relevant one from our favorite recipes, but if asked about a very specific recipe that is not in our favorites, check the recipes from internet.",
        },
        "summarize": {},
    }
    resp = await nucliadb_agentic_api_http_client.post(
        f"/api/v1/kb/{eric_dataset}/agentic_configs/default",
        json=payload,
        headers={
            "X-NUCLIADB-ROLES": "OWNER;READER;WRITER",
            "X-NUCLIADB-ACCOUNT": "nuclia",
        },
    )
    assert resp.status_code == 201, resp.text

    headers = {
        "X-NUCLIADB-ROLES": "READER",
        "X-NUCLIADB-USER": "user1",
        "X-STF-USER": "user1",
        "X-NUCLIADB-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
        "X-NUCLIADB-ACCOUNT-TYPE": "basic",
    }

    report_step_usage = mocker.spy(StreamAuditStorage, "report_step_usage")
    async with connect(
        f"ws://{nucliadb_agentic_api_http}/api/v1/kb/{eric_dataset}/ask?agentic_config_id=default&keep_open=true",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "Give me the recipe for 'Grilled Romaine Caesar Salad'",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        answers = []
        async for message in websocket:
            print(message)
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.ANSWER:
                if response.answer:
                    answers.append(response.answer)
            elif response.operation == AnswerOperation.DONE:
                break
            elif response.operation == AnswerOperation.ERROR:
                assert False, (
                    f"Interaction error: {response.exception.detail if response.exception else ''}"
                )
            else:
                print(
                    "No feedback, step, possible_answer, context or generated_text in response"
                )

        assert "parmesan" in answers[0].lower()
    report_step_usage.assert_called_once()
    _, kwargs = report_step_usage.call_args
    assert kwargs["account_id"] == "nuclia"
    assert kwargs["kbid"] == eric_dataset
    assert kwargs["step"].external_usage is not None
    assert kwargs["step"].external_usage[0].provider == "perplexity"
