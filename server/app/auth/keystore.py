"""Key storage seam.

One tiny interface, two implementations:

* :class:`StaticKeyStore` -- reads ``API_KEYS`` from the environment, no database,
  works out of the box.
* :class:`PostgresKeyStore` -- a stub that satisfies the Protocol and raises
  ``NotImplementedError``.  It exists to prove the seam (CONTRACT section 9 puts the
  real Postgres implementation out of scope for v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.config import Settings


@dataclass(frozen=True)
class KeyRecord:
    """What a key store knows about one API key."""

    key: str
    label: str = ""
    revoked: bool = False
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class KeyStore(Protocol):
    """Lookup interface for API keys."""

    def validate(self, key: str) -> KeyRecord | None:
        """Return the record for ``key``, or ``None`` when the key is unknown.

        A *known but revoked* key still returns a record (with ``revoked=True``)
        so the caller can answer 403 instead of 401.
        """

    def revoked(self, key: str) -> bool:
        """``True`` when the key is known and revoked."""


class StaticKeyStore:
    """In-memory key store backed by the comma separated ``API_KEYS`` env var.

    Input:  StaticKeyStore(["sk-a"]).validate("sk-a")
    Output: KeyRecord(key="sk-a", label="static", revoked=False, metadata={})
    """

    backend = "static"

    def __init__(self, keys: list[str] | None = None, revoked_keys: list[str] | None = None) -> None:
        self._keys = {key.strip() for key in (keys or []) if key.strip()}
        self._revoked = {key.strip() for key in (revoked_keys or []) if key.strip()}

    @classmethod
    def from_settings(cls, settings: Settings) -> "StaticKeyStore":
        return cls(settings.api_key_list)

    def validate(self, key: str) -> KeyRecord | None:
        key = (key or "").strip()
        if key in self._revoked:
            return KeyRecord(key=key, label="static", revoked=True)
        if key in self._keys:
            return KeyRecord(key=key, label="static", revoked=False)
        return None

    def revoked(self, key: str) -> bool:
        return (key or "").strip() in self._revoked


class PostgresKeyStore:
    """Stub Postgres-backed key store.

    Satisfies :class:`KeyStore` structurally; every method raises so that wiring
    ``KEYSTORE_BACKEND=postgres`` fails loudly instead of silently allowing access.
    """

    backend = "postgres"

    def __init__(self, database_url: str = "") -> None:
        self.database_url = database_url

    @classmethod
    def from_settings(cls, settings: Settings) -> "PostgresKeyStore":
        return cls(settings.database_url)

    def _unimplemented(self) -> NotImplementedError:
        return NotImplementedError(
            "PostgresKeyStore is a v1 stub: the Postgres backend is interface-only "
            "(CONTRACT section 9). Set KEYSTORE_BACKEND=static and populate API_KEYS."
        )

    def validate(self, key: str) -> KeyRecord | None:
        raise self._unimplemented()

    def revoked(self, key: str) -> bool:
        raise self._unimplemented()


def build_keystore(settings: Settings) -> KeyStore:
    """Pick a key store from ``KEYSTORE_BACKEND``."""
    backend = (settings.keystore_backend or "static").strip().lower()
    if backend == "static":
        return StaticKeyStore.from_settings(settings)
    if backend == "postgres":
        return PostgresKeyStore.from_settings(settings)
    raise ValueError(
        f"Unknown KEYSTORE_BACKEND {backend!r}; expected 'static' or 'postgres'."
    )
