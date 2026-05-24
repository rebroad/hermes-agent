from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = SimpleNamespace(write=lambda _data: None, close=lambda: None)
        self._poll = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):  # noqa: ARG002
        self._poll = 0
        return 0


def test_codex_proxy_launches_and_returns_local_base_url(monkeypatch, tmp_path):
    from hermes_cli import codex_proxy

    server_info = tmp_path / "server-info.json"
    server_info.write_text(json.dumps({"port": 43128}), encoding="utf-8")
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        server_info.write_text(json.dumps({"port": 43128}), encoding="utf-8")
        return _FakeProc()

    monkeypatch.setenv("HERMES_CODEX_USE_PROXY", "1")
    monkeypatch.setattr(codex_proxy, "shutil", SimpleNamespace(which=lambda _name: "/usr/bin/codex-responses-api-proxy"))
    monkeypatch.setattr(codex_proxy, "_server_info_path", lambda _fp: server_info)
    monkeypatch.setattr(codex_proxy.subprocess, "Popen", fake_popen)

    base_url = codex_proxy.resolve_codex_proxy_base_url("token-123")

    assert base_url == "http://127.0.0.1:43128"
    assert captured["cmd"][0].endswith("codex-responses-api-proxy")
    assert "--upstream-url" in captured["cmd"]
    assert "https://chatgpt.com/backend-api/codex" in captured["cmd"]
    assert "--provider-id" in captured["cmd"]
    assert "openai-codex" in captured["cmd"]


def test_codex_runtime_credentials_fail_closed_without_proxy(monkeypatch):
    from hermes_cli.auth import AuthError, resolve_codex_runtime_credentials

    monkeypatch.setattr(
        "hermes_cli.auth._read_codex_tokens",
        lambda: {"tokens": {"access_token": "token-123"}, "last_refresh": None},
    )
    monkeypatch.setattr("hermes_cli.auth.resolve_codex_proxy_base_url", lambda _token: None)

    with pytest.raises(AuthError, match="Codex proxy is required"):
        resolve_codex_runtime_credentials(refresh_if_expiring=False)


def test_codex_runtime_credentials_use_proxy_base_url(monkeypatch):
    from hermes_cli.auth import resolve_codex_runtime_credentials

    monkeypatch.setattr(
        "hermes_cli.auth._read_codex_tokens",
        lambda: {"tokens": {"access_token": "token-123"}, "last_refresh": "now"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_proxy_base_url",
        lambda _token: "http://127.0.0.1:43128",
    )

    creds = resolve_codex_runtime_credentials(refresh_if_expiring=False)

    assert creds["provider"] == "openai-codex"
    assert creds["api_key"] == "token-123"
    assert creds["base_url"] == "http://127.0.0.1:43128"


def test_auxiliary_codex_client_uses_proxy_base_url(monkeypatch):
    from agent import auxiliary_client

    monkeypatch.setattr(auxiliary_client, "_select_pool_entry", lambda _provider: (False, None))
    monkeypatch.setattr(auxiliary_client, "_read_codex_access_token", lambda: "token-123")
    monkeypatch.setattr(
        "hermes_cli.codex_proxy.resolve_codex_proxy_base_url",
        lambda _token: "http://127.0.0.1:43128",
    )
    mock_openai = MagicMock()
    mock_openai.return_value = MagicMock()
    monkeypatch.setattr(auxiliary_client, "OpenAI", mock_openai)

    client, model = auxiliary_client._build_codex_client("gpt-5.4")

    assert model == "gpt-5.4"
    assert client is not None
    assert mock_openai.call_args.kwargs["base_url"] == "http://127.0.0.1:43128"


def test_raw_codex_client_uses_proxy_base_url(monkeypatch):
    from agent import auxiliary_client

    monkeypatch.setattr(auxiliary_client, "_read_codex_access_token", lambda: "token-123")
    monkeypatch.setattr(
        "hermes_cli.codex_proxy.resolve_codex_proxy_base_url",
        lambda _token: "http://127.0.0.1:43128",
    )
    mock_openai = MagicMock()
    mock_openai.return_value = MagicMock()
    monkeypatch.setattr(auxiliary_client, "OpenAI", mock_openai)

    client, model = auxiliary_client.resolve_provider_client(
        "openai-codex",
        model="gpt-5.4",
        raw_codex=True,
    )

    assert model == "gpt-5.4"
    assert client is not None
    assert mock_openai.call_args.kwargs["base_url"] == "http://127.0.0.1:43128"
