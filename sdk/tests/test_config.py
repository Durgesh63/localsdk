"""Config precedence: explicit arg > env > ~/.localsdk/config.toml > default."""

from __future__ import annotations

import pytest

from localsdk import Client, ConfigurationError, resolve_config
from localsdk.config import DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_TIMEOUT, env_var

from conftest import API_KEY, BASE_URL, json_handler, make_client


def write_config(tmp_path, **values) -> str:
    body = "[default]\n" + "".join(
        '{0} = "{1}"\n'.format(key, value) for key, value in values.items()
    )
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_explicit_argument_wins_over_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALSDK_API_KEY", "sk-from-env")
    monkeypatch.setenv("LOCALSDK_MODEL", "model-from-env")
    path = write_config(tmp_path, api_key="sk-from-file", model="model-from-file")

    cfg = resolve_config(
        api_key="sk-explicit", base_url=BASE_URL, model="model-explicit", config_file=path
    )

    assert cfg.api_key == "sk-explicit"
    assert cfg.model == "model-explicit"


def test_env_wins_over_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALSDK_API_KEY", "sk-from-env")
    monkeypatch.setenv("LOCALSDK_BASE_URL", "https://env.ngrok-free.app")
    path = write_config(
        tmp_path, api_key="sk-from-file", base_url="https://file.ngrok-free.app"
    )

    cfg = resolve_config(config_file=path)

    assert cfg.api_key == "sk-from-env"
    assert cfg.base_url == "https://env.ngrok-free.app"


def test_config_file_wins_over_default(tmp_path):
    path = write_config(tmp_path, api_key="sk-from-file", base_url=BASE_URL, model="mini")

    cfg = resolve_config(config_file=path)

    assert cfg.api_key == "sk-from-file"
    assert cfg.model == "mini"


def test_defaults_apply_when_nothing_else_is_set(tmp_path):
    cfg = resolve_config(
        api_key=API_KEY, base_url=BASE_URL, config_file=str(tmp_path / "missing.toml")
    )

    assert cfg.model == DEFAULT_MODEL
    assert cfg.timeout == DEFAULT_TIMEOUT
    assert cfg.max_retries == DEFAULT_MAX_RETRIES == 0


def test_missing_api_key_names_the_env_var(tmp_path):
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_config(base_url=BASE_URL, config_file=str(tmp_path / "missing.toml"))

    assert env_var("api_key") in str(excinfo.value)
    assert "LOCALSDK_API_KEY" == env_var("api_key")


def test_missing_base_url_names_the_env_var_and_warns_about_ngrok(tmp_path):
    with pytest.raises(ConfigurationError) as excinfo:
        resolve_config(api_key=API_KEY, config_file=str(tmp_path / "missing.toml"))

    message = str(excinfo.value)
    assert "LOCALSDK_BASE_URL" in message
    assert "ngrok" in message


def test_client_construction_raises_configuration_error_immediately():
    with pytest.raises(ConfigurationError):
        Client(base_url=BASE_URL)


def test_base_url_is_normalised():
    cfg = resolve_config(api_key=API_KEY, base_url="https://x.ngrok-free.app/v1/")
    assert cfg.base_url == "https://x.ngrok-free.app"
    assert cfg.api_base == "https://x.ngrok-free.app/v1"


def test_base_url_must_be_absolute():
    with pytest.raises(ConfigurationError):
        resolve_config(api_key=API_KEY, base_url="x.ngrok-free.app")


def test_timeout_from_env_is_coerced(monkeypatch):
    monkeypatch.setenv("LOCALSDK_TIMEOUT", "42")
    cfg = resolve_config(api_key=API_KEY, base_url=BASE_URL)
    assert cfg.timeout == 42.0


def test_non_numeric_timeout_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("LOCALSDK_TIMEOUT", "soon")
    with pytest.raises(ConfigurationError):
        resolve_config(api_key=API_KEY, base_url=BASE_URL)


def test_malformed_config_file_is_reported(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[default\napi_key = ", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        resolve_config(config_file=str(path))


def test_missing_config_file_is_not_an_error(tmp_path):
    cfg = resolve_config(
        api_key=API_KEY, base_url=BASE_URL, config_file=str(tmp_path / "nope.toml")
    )
    assert cfg.api_key == API_KEY


def test_client_exposes_resolved_config():
    client = make_client(json_handler({}), model="custom-model", timeout=12, max_retries=2)
    try:
        assert client.base_url == BASE_URL
        assert client.model == "custom-model"
        assert client.timeout == 12
        assert client.max_retries == 2
        assert client.config.api_base == BASE_URL + "/v1"
    finally:
        client.close()
