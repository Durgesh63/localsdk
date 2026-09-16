"""Configuration resolution.

Precedence, highest first (CONTRACT section 7)::

    explicit argument  >  environment variable  >  ~/.localsdk/config.toml  >  default

Environment variables (derived from :data:`localsdk._version.ENV_PREFIX`)::

    LOCALSDK_API_KEY
    LOCALSDK_BASE_URL
    LOCALSDK_MODEL
    LOCALSDK_TIMEOUT
    LOCALSDK_MAX_RETRIES   (extension: retries are off unless you ask)

Config file ``~/.localsdk/config.toml``::

    [default]
    api_key = "sk-your-own-key"
    base_url = "https://xxxx.ngrok-free.app"
    model = "qwen2.5:14b"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._version import (
    CONFIG_DIR_NAME,
    CONFIG_FILE_NAME,
    CONFIG_SECTION,
    ENV_PREFIX,
)
from .errors import ConfigurationError

#: Model used when nothing else says otherwise (CONTRACT section 7).
DEFAULT_MODEL = "qwen2.5:14b"

#: A 14B model on one box is slow; 300s is the server's own REQUEST_TIMEOUT_S.
DEFAULT_TIMEOUT = 300.0

#: Retries are OFF by default. Free ngrok meters requests; a silent retry
#: doubles spend on a box that can only run one generation at a time.
DEFAULT_MAX_RETRIES = 0


def env_var(name: str) -> str:
    """Return the fully qualified env var for a setting.

    Input:  "api_key"      Output: "LOCALSDK_API_KEY"
    """
    return "{0}_{1}".format(ENV_PREFIX, name.upper())


def default_config_path() -> Path:
    """Path of the per-user config file, ``~/.localsdk/config.toml``."""
    return Path.home() / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _load_toml(path: Path) -> dict[str, Any]:
    """Read ``[default]`` out of a TOML file.

    Returns an empty dict when the file is absent or no TOML parser is
    available (Python 3.10 without ``tomli`` installed) -- the file layer is a
    convenience, never a hard dependency.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - only on 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return {}

    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    except Exception as exc:  # malformed TOML is worth complaining about
        raise ConfigurationError(
            "could not parse config file {0}: {1}".format(path, exc)
        ) from exc

    section = data.get(CONFIG_SECTION, {})
    if not isinstance(section, dict):
        raise ConfigurationError(
            "config file {0} has a [{1}] entry that is not a table".format(path, CONFIG_SECTION)
        )
    return dict(section)


def _coerce_float(value: Any, source: str, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "{0} from {1} must be a number, got {2!r}".format(name, source, value)
        ) from exc


def _coerce_int(value: Any, source: str, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "{0} from {1} must be an integer, got {2!r}".format(name, source, value)
        ) from exc


def normalize_base_url(base_url: str) -> str:
    """Strip trailing slashes and a trailing ``/v1`` from a base URL.

    The SDK appends ``/v1`` itself (CONTRACT section 2), so both spellings
    of the tunnel URL work.

    Input:  "https://x.ngrok-free.app/v1/"   Output: "https://x.ngrok-free.app"
    """
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url.rstrip("/")


@dataclass(frozen=True)
class Config:
    """Fully resolved client configuration."""

    api_key: str
    base_url: str
    model: str
    timeout: float
    max_retries: int

    @property
    def api_base(self) -> str:
        """The versioned API root, ``{base_url}/v1``."""
        return "{0}/v1".format(self.base_url)


def resolve_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    config_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Resolve settings by precedence and validate them.

    Input:  resolve_config(api_key="sk-1", base_url="https://x.ngrok-free.app")
    Output: Config(api_key="sk-1", base_url="https://x.ngrok-free.app",
                   model="qwen2.5:14b", timeout=300.0, max_retries=0)

    Raises:
        ConfigurationError: when ``api_key`` or ``base_url`` cannot be found,
            naming the environment variable to set.
    """
    env = os.environ if env is None else env
    path = Path(config_file) if config_file is not None else default_config_path()
    file_cfg = _load_toml(path)

    def pick(name: str, explicit: Any) -> tuple[Any, str]:
        """Return (value, source) for one setting."""
        if explicit is not None:
            return explicit, "argument"
        var = env_var(name)
        if env.get(var):
            return env[var], "${0}".format(var)
        if file_cfg.get(name) is not None:
            return file_cfg[name], str(path)
        return None, "default"

    api_key_value, _ = pick("api_key", api_key)
    base_url_value, _ = pick("base_url", base_url)
    model_value, _ = pick("model", model)
    timeout_value, timeout_source = pick("timeout", timeout)
    retries_value, retries_source = pick("max_retries", max_retries)

    if not api_key_value:
        raise ConfigurationError(
            "no API key found. Pass api_key=..., set {0}, or add "
            'api_key = "sk-..." under [{1}] in {2}.'.format(
                env_var("api_key"), CONFIG_SECTION, path
            )
        )
    if not base_url_value:
        raise ConfigurationError(
            "no base_url found. Pass base_url=..., set {0}, or add "
            'base_url = "https://<your>.ngrok-free.app" under [{1}] in {2}. '
            "Note that a free ngrok tunnel gets a new URL every restart, so this "
            "value is configuration -- never hardcode it.".format(
                env_var("base_url"), CONFIG_SECTION, path
            )
        )

    resolved_base = normalize_base_url(str(base_url_value))
    if not resolved_base.startswith(("http://", "https://")):
        raise ConfigurationError(
            "base_url must start with http:// or https://, got {0!r}".format(base_url_value)
        )

    resolved_timeout = (
        DEFAULT_TIMEOUT
        if timeout_value is None
        else _coerce_float(timeout_value, timeout_source, "timeout")
    )
    if resolved_timeout <= 0:
        raise ConfigurationError("timeout must be > 0, got {0!r}".format(resolved_timeout))

    resolved_retries = (
        DEFAULT_MAX_RETRIES
        if retries_value is None
        else _coerce_int(retries_value, retries_source, "max_retries")
    )
    if resolved_retries < 0:
        raise ConfigurationError("max_retries must be >= 0, got {0!r}".format(resolved_retries))

    return Config(
        api_key=str(api_key_value),
        base_url=resolved_base,
        model=str(model_value) if model_value else DEFAULT_MODEL,
        timeout=resolved_timeout,
        max_retries=resolved_retries,
    )


__all__ = [
    "Config",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "default_config_path",
    "env_var",
    "normalize_base_url",
    "resolve_config",
]
