"""LocalSDK AI -- a small, typed client for a self-hosted OpenAI-compatible LLM.

Quickstart::

    from localsdk import Client

    client = Client(api_key="sk-...", base_url="https://xxxx.ngrok-free.app")
    print(client.chat([{"role": "user", "content": "hello"}]).text)

Everything is available in both flavours: :class:`Client` (sync) and
:class:`AsyncClient` (async) share one implementation and one set of
signatures.
"""

from ._sync import Client
from ._version import PACKAGE_NAME, __version__
from .client import AsyncClient
from .config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    Config,
    default_config_path,
    resolve_config,
)
from .errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConfigurationError,
    MaxRoundsExceeded,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    StructuredOutputError,
    ToolExecutionError,
    UpstreamError,
    LocalSDKError,
)
from .tools import append_tool_results, execute_tool_call, function_schema
from .types import (
    ChatResponse,
    Choice,
    Chunk,
    Delta,
    FunctionCall,
    Health,
    Message,
    Model,
    ToolCall,
    Usage,
)

__all__ = [
    # clients
    "Client",
    "AsyncClient",
    # config
    "Config",
    "resolve_config",
    "default_config_path",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    # types
    "Message",
    "ChatResponse",
    "Choice",
    "Chunk",
    "Delta",
    "ToolCall",
    "FunctionCall",
    "Model",
    "Usage",
    "Health",
    # tools
    "function_schema",
    "execute_tool_call",
    "append_tool_results",
    # errors
    "LocalSDKError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "UpstreamError",
    "APIConnectionError",
    "APITimeoutError",
    "ConfigurationError",
    "ToolExecutionError",
    "MaxRoundsExceeded",
    "StructuredOutputError",
    # meta
    "__version__",
    "PACKAGE_NAME",
]
