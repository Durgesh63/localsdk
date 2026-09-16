"""FastAPI app factory: lifespan, shared httpx client, router wiring.

``uvicorn app.main:app --reload`` serves the module-level ``app``; tests build
their own via :func:`create_app` with an ``httpx.MockTransport`` so no live
Ollama (or network) is ever required.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, Depends, FastAPI

from app import __version__
from app.auth.keystore import KeyStore, build_keystore
from app.config import Settings, get_settings
from app.deps import require_api_key
from app.errors import register_exception_handlers
from app.ollama.client import OllamaClient
from app.routes import chat, embeddings, health, models

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    keystore: KeyStore | None = None,
) -> FastAPI:
    """Build the application.

    ``transport`` and ``keystore`` exist purely as test seams; production calls
    this with no arguments.
    """
    resolved = settings or get_settings()
    logging.basicConfig(level=getattr(logging, resolved.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(
            base_url=resolved.ollama_base_url,
            timeout=httpx.Timeout(resolved.request_timeout_s, connect=10.0),
            transport=transport,
        )
        application.state.settings = resolved
        application.state.keystore = keystore or build_keystore(resolved)
        application.state.http = client
        application.state.ollama = OllamaClient(client)
        logger.info(
            "server ready: ollama=%s model=%s keystore=%s timeout=%ss",
            resolved.ollama_base_url,
            resolved.default_model,
            resolved.keystore_backend,
            resolved.request_timeout_s,
        )
        try:
            yield
        finally:
            await client.aclose()

    application = FastAPI(
        title="LocalSDK Local LLM Server",
        version=__version__,
        description="OpenAI-compatible facade in front of a local Ollama instance.",
        lifespan=lifespan,
    )

    register_exception_handlers(application)

    application.include_router(health.router)

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
    v1.include_router(models.router)
    v1.include_router(chat.router)
    v1.include_router(embeddings.router)
    application.include_router(v1)

    return application


app = create_app()
