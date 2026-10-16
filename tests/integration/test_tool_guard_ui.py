"""Tool-guard verdicts reach LiteLLM's spend logs and Guardrails Monitor index through a real database."""

import json
import time
import uuid

import pytest
from conftest import GUARDED_ALIAS, MODEL
from fake_scanner import MARKER, SCORE

WAIT_SECONDS = 60


def _body(nonce: str, results: list[tuple[str, str, str]]) -> dict:
    """An /v1/messages body whose last turn holds one new tool result per (tool name, call id, text)."""
    return {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "run the tools"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": cid, "name": name, "input": {}} for name, cid, _ in results],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": cid, "content": f"{text} {nonce}"}
                    for _, cid, text in results
                ],
            },
        ],
    }


def _eventually(fetch, what: str):
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        value = fetch()
        if value:
            return value
        time.sleep(1)
    pytest.fail(f"{what} did not appear within {WAIT_SECONDS}s")


def _send(proxy, key: str, body: dict) -> str:
    resp = proxy.client.post("/v1/messages", json=body, headers={"authorization": f"Bearer {key}"})
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _tool_guard_entry(proxy, request_id: str) -> dict:
    """The tool-guard guardrail record in the request's spend log, once the log is written."""

    def fetch():
        rows = proxy.database.query(
            f"""SELECT metadata::text FROM "LiteLLM_SpendLogs" WHERE request_id = '{request_id}'"""
        )
        return rows[0][0] if rows else None

    metadata = json.loads(_eventually(fetch, f"spend log for {request_id}"))
    entries = [e for e in metadata.get("guardrail_information") or [] if e.get("guardrail_name") == "tool-guard"]
    assert len(entries) == 1, metadata
    return entries[0]


@pytest.fixture(scope="module")
def database_key(proxy) -> str:
    """A virtual key stored in the database, so its alias resolves through LiteLLM's own auth."""
    resp = proxy.client.post("/key/generate", json={"key_alias": GUARDED_ALIAS})
    assert resp.status_code == 200, resp.text
    return resp.json()["key"]


def test_flagged_tool_result_is_recorded_in_spend_logs_and_the_guardrails_monitor(proxy, database_key):
    nonce = uuid.uuid4().hex
    body = _body(nonce, [("Bash", "toolu_flagged", MARKER), ("Read", "toolu_clean", "benign")])

    request_id = _send(proxy, database_key, body)

    entry = _tool_guard_entry(proxy, request_id)
    assert entry["guardrail_status"] == "guardrail_flagged"
    assert entry["guardrail_provider"] == "llm-tool-guard"
    assert entry["guardrail_mode"] == "pre_call"
    assert entry["confidence_score"] == SCORE
    assert entry["guardrail_response"]["results"] == [
        {
            "tool_name": "Bash",
            "tool_call_id": "toolu_flagged",
            "verdict": "suspected-prompt-injection",
            "score": SCORE,
            "text": f"{MARKER} {nonce}",
        }
    ]

    indexed = _eventually(
        lambda: proxy.database.query(
            f"""SELECT guardrail_id FROM "LiteLLM_SpendLogGuardrailIndex" WHERE request_id = '{request_id}'"""
        ),
        "Guardrails Monitor index row",
    )
    assert ["tool-guard"] in indexed

    def monitor_log():
        resp = proxy.client.get("/guardrails/usage/logs", params={"guardrail_id": "tool-guard"})
        assert resp.status_code == 200, resp.text
        return [log for log in resp.json()["logs"] if log["id"] == request_id]

    [log] = _eventually(monitor_log, "Guardrails Monitor log row")
    assert log["action"] == "flagged"
    assert log["score"] == round(SCORE, 2)

    def monitor_overview():
        resp = proxy.client.get("/guardrails/usage/overview")
        assert resp.status_code == 200, resp.text
        return [row for row in resp.json()["rows"] if row["name"] == "tool-guard" and row["requestsEvaluated"] >= 1]

    _eventually(monitor_overview, "Guardrails Monitor overview row")


def test_clean_tool_result_is_recorded_as_success(proxy, database_key):
    body = _body(uuid.uuid4().hex, [("Read", "toolu_clean", "benign")])

    request_id = _send(proxy, database_key, body)

    entry = _tool_guard_entry(proxy, request_id)
    assert entry["guardrail_status"] == "success"
    assert entry["guardrail_response"]["results"] == []
