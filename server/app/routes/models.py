"""``GET /v1/models`` -- the models Ollama has pulled, in OpenAI shape."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import Ollama
from app.ollama import translate
from app.schemas.openai import ModelList

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelList)
async def list_models(ollama: Ollama) -> ModelList:
    tags = await ollama.list_models()
    return translate.from_ollama_tags(tags)
