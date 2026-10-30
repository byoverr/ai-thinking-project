"""Адаптер Ollama для thinking-моделей без изменения исходного LLMClient.

Запуск: OLLAMA_UPSTREAM=http://host:11434/v1 uv run python -m assistant.agent.ollama_gateway
В .env: LLM_BASE_URL=http://127.0.0.1:11435/v1
Все запросы и учёт токенов по-прежнему идут через инфраструктурный LLMClient.
Адаптер добавляет только reasoning_effort=none; ответы передаются без изменений.
"""

import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def create_app(upstream: str, transport=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url=upstream.rstrip("/") + "/", timeout=600, transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    async def forward(request: Request, endpoint: str):
        body = await request.json()
        if endpoint == "chat/completions":
            body["reasoning_effort"] = "none"
        headers = {}
        if request.headers.get("authorization"):
            headers["authorization"] = request.headers["authorization"]
        try:
            response = await app.state.client.post(endpoint, json=body, headers=headers)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"Ollama недоступна: {exc}"}, status_code=502)
        return Response(
            response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await forward(request, "chat/completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await forward(request, "embeddings")

    return app


if __name__ == "__main__":
    upstream = os.environ.get("OLLAMA_UPSTREAM")
    if not upstream:
        raise SystemExit("Укажите OLLAMA_UPSTREAM, например http://192.168.64.1:11434/v1")
    uvicorn.run(create_app(upstream), host="127.0.0.1", port=11435)
