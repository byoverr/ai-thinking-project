"""Адаптер меняет только режим thinking, сохраняя схемы, токены и ошибки."""

import json

import httpx
from fastapi.testclient import TestClient

from assistant.agent.ollama_gateway import create_app


def test_gateway_keeps_tools_schema_usage_and_auth():
    requests = []
    response = {"choices": [{"message": {"content": "Ответ"}}], "usage": {"completion_tokens": 12}}

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    body = {
        "model": "qwen3.5:4b",
        "messages": [],
        "tools": [{"type": "function"}],
        "response_format": {"type": "json_schema"},
        "temperature": 0,
        "seed": 7,
    }
    with TestClient(create_app("http://upstream/v1", httpx.MockTransport(handler))) as client:
        result = client.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer test"})
    assert result.json() == response
    assert json.loads(requests[0].content) == {**body, "reasoning_effort": "none"}
    assert str(requests[0].url) == "http://upstream/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test"


def test_embedding_body_and_upstream_errors_are_preserved():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(404, json={"error": "model is not installed"})

    body = {"model": "bge-m3", "input": ["документ"]}
    with TestClient(create_app("http://upstream/v1", httpx.MockTransport(handler))) as client:
        result = client.post("/v1/embeddings", json=body)
    assert result.status_code == 404
    assert result.json() == {"error": "model is not installed"}
    assert json.loads(requests[0].content) == body
