"""``POST /v1/embeddings``."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import ApiKey, Config, Ollama
from app.errors import InvalidRequestError
from app.ollama import translate
from app.schemas.openai import EmbeddingsRequest, EmbeddingsResponse

router = APIRouter(tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(
    body: EmbeddingsRequest,
    ollama: Ollama,
    settings: Config,
    _key: ApiKey,
) -> EmbeddingsResponse:
    text_input = body.input
    if isinstance(text_input, list) and not text_input:
        raise InvalidRequestError("'input' must not be an empty list.", param="input")
    if isinstance(text_input, str) and not text_input:
        raise InvalidRequestError("'input' must not be empty.", param="input")

    model = body.model or settings.embed_model
    payload = translate.to_ollama_embeddings_request(model, text_input)
    data = await ollama.embeddings(payload)
    return translate.from_ollama_embeddings(data, model=model)
