"""Client and AsyncClient must stay interchangeable: same names, same signatures.

If this file fails, the sync mirror has drifted from the async original.
"""

from __future__ import annotations

import inspect

import pytest

from localsdk import AsyncClient, Client

PUBLIC_METHODS = ["chat", "stream", "structured", "embed", "models", "health", "run_tools", "close"]


def public_names(cls: type) -> set[str]:
    return {
        name
        for name, value in vars(cls).items()
        if not name.startswith("_") and callable(value)
    }


def test_both_clients_expose_the_same_public_methods():
    assert public_names(Client) == public_names(AsyncClient)
    assert set(PUBLIC_METHODS) <= public_names(Client)


@pytest.mark.parametrize("name", PUBLIC_METHODS)
def test_signatures_match(name):
    sync_sig = inspect.signature(getattr(Client, name))
    async_sig = inspect.signature(getattr(AsyncClient, name))

    assert list(sync_sig.parameters) == list(async_sig.parameters), name
    for param in sync_sig.parameters:
        sync_param = sync_sig.parameters[param]
        async_param = async_sig.parameters[param]
        assert sync_param.kind == async_param.kind, "{0}.{1} kind".format(name, param)
        assert sync_param.default == async_param.default, "{0}.{1} default".format(name, param)
        assert sync_param.annotation == async_param.annotation, "{0}.{1} annotation".format(
            name, param
        )


def test_constructor_signatures_match():
    sync_sig = inspect.signature(Client.__init__)
    async_sig = inspect.signature(AsyncClient.__init__)
    assert sync_sig.parameters == async_sig.parameters


@pytest.mark.parametrize("name", PUBLIC_METHODS)
def test_every_public_method_is_documented(name):
    assert (getattr(Client, name).__doc__ or "").strip(), name
    assert (getattr(AsyncClient, name).__doc__ or "").strip(), name


def test_async_methods_are_coroutines_and_sync_ones_are_not():
    for name in ["chat", "structured", "embed", "models", "health", "run_tools", "close"]:
        assert inspect.iscoroutinefunction(getattr(AsyncClient, name)), name
        assert not inspect.iscoroutinefunction(getattr(Client, name)), name


def test_stream_is_a_plain_function_on_both_returning_an_iterator():
    """stream() must not be a coroutine: you iterate it, you do not await it."""
    assert not inspect.iscoroutinefunction(AsyncClient.stream)
    assert not inspect.isasyncgenfunction(AsyncClient.stream)
    assert not inspect.isgeneratorfunction(Client.stream)


def test_context_manager_protocols_exist():
    assert hasattr(Client, "__enter__") and hasattr(Client, "__exit__")
    assert hasattr(AsyncClient, "__aenter__") and hasattr(AsyncClient, "__aexit__")


def test_sync_client_never_touches_an_event_loop():
    """A sync client that spins up a loop breaks inside async frameworks."""
    import localsdk._sync as sync_module

    # skip the module docstring, which legitimately mentions these names
    source = inspect.getsource(sync_module).split('"""', 2)[-1]
    for forbidden in ("asyncio.run", "get_event_loop", "new_event_loop", "run_until_complete"):
        assert forbidden not in source, forbidden
