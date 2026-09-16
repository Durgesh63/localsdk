"""Runtime configuration, loaded from the environment (CONTRACT section 7)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server settings.

    Every field maps 1:1 to a variable in ``server/.env.example``.  ``API_KEYS`` is
    kept as a raw comma separated string (rather than a ``list[str]``) because
    pydantic-settings would otherwise try to JSON-decode the env value.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5:14b"
    embed_model: str = "nomic-embed-text"

    keystore_backend: str = "static"
    api_keys: str = ""
    database_url: str = ""

    request_timeout_s: float = 300.0

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def api_key_list(self) -> list[str]:
        """``API_KEYS`` split into individual keys, blanks removed."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
