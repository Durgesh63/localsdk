"""Single source of truth for naming and versioning.

Every user-visible name (distribution name, env var prefix, config directory,
User-Agent) is derived from the constants here. Renaming the product is a
one-file change: edit ``PACKAGE_NAME``/``BRAND`` and nothing else moves.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Distribution (pip) name. Placeholder -- rename here and nowhere else.
PACKAGE_NAME = "localsdk"

#: Import name of this package.
IMPORT_NAME = "localsdk"

#: Brand token used to derive env vars and the config directory.
BRAND = "localsdk"

#: Environment variable prefix, e.g. ``LOCALSDK_API_KEY``.
ENV_PREFIX = BRAND.upper()

#: Per-user config directory under ``~``.
CONFIG_DIR_NAME = f".{BRAND}"

#: Config file name inside :data:`CONFIG_DIR_NAME`.
CONFIG_FILE_NAME = "config.toml"

#: Section read from the config file.
CONFIG_SECTION = "default"

#: Value of the ``User-Agent`` header on every request (CONTRACT section 2).
USER_AGENT = f"{PACKAGE_NAME}-python/{__version__}"

__all__ = [
    "__version__",
    "PACKAGE_NAME",
    "IMPORT_NAME",
    "BRAND",
    "ENV_PREFIX",
    "CONFIG_DIR_NAME",
    "CONFIG_FILE_NAME",
    "CONFIG_SECTION",
    "USER_AGENT",
]
