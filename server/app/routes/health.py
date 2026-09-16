"""``GET /healthz`` -- unauthenticated liveness probe (CONTRACT section 3)."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.deps import Config, Ollama
from app.schemas.openai import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz(settings: Config, ollama: Ollama) -> HealthResponse:
    """Always 200: this is a probe, not a gate."""
    up = await ollama.ping()
    return HealthResponse(
        status="ok",
        ollama="up" if up else "down",
        model=settings.default_model,
        version=__version__,
    )
