"""Optional local codex-responses-api-proxy integration."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

logger = logging.getLogger(__name__)

_STATE_DIR = Path("/var/tmp") / "hermes-codex-proxy"
_BINARY = "codex-responses-api-proxy"
_DEFAULT_UPSTREAM_URL = "https://chatgpt.com/backend-api/codex"
_ENV_PROXY_URL = "HERMES_CODEX_PROXY_URL"
_ENV_PROXY_MODE = "HERMES_CODEX_USE_PROXY"
_ENV_PROXY_UPSTREAM_URL = "HERMES_CODEX_PROXY_UPSTREAM_URL"
_ENV_PROXY_PROVIDER_ID = "HERMES_CODEX_PROXY_PROVIDER_ID"
_ENV_PROXY_SQLITE_HOME = "HERMES_CODEX_PROXY_SQLITE_HOME"
_ENV_PROXY_SERVER_INFO = "HERMES_CODEX_PROXY_SERVER_INFO"
_ENABLE_VALUES = {"1", "true", "yes", "on", "auto"}
_DISABLE_VALUES = {"0", "false", "no", "off"}


@dataclass
class _ProxyRuntime:
    process: subprocess.Popen[str]
    base_url: str
    token_fingerprint: str


_proxy_lock = threading.Lock()
_proxy_runtime: Optional[_ProxyRuntime] = None
_atexit_registered = False


def _normalize(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8", errors="replace")).hexdigest()[:16]


def _register_atexit() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(_shutdown_proxy)
    _atexit_registered = True


def _server_info_path(token_fp: str) -> Path:
    override = _normalize(os.getenv(_ENV_PROXY_SERVER_INFO, ""))
    if override:
        return Path(override)
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / f"server-info-{token_fp}.json"


def _proxy_enabled() -> bool:
    mode = _normalize(os.getenv(_ENV_PROXY_MODE, "")).lower()
    if mode in _DISABLE_VALUES:
        return False
    if mode in _ENABLE_VALUES:
        return True
    # Default to proxy-on for Codex. The only way to suppress the sidecar is
    # to set HERMES_CODEX_USE_PROXY to an explicit false-ish value.
    return True


def _shutdown_proxy() -> None:
    global _proxy_runtime
    with _proxy_lock:
        runtime = _proxy_runtime
        _proxy_runtime = None
    if runtime is None:
        return

    try:
        shutdown_url = f"{runtime.base_url}/shutdown"
        req = urllib.request.Request(shutdown_url, method="GET")
        urllib.request.urlopen(req, timeout=2.0).read()
    except Exception:
        pass

    try:
        runtime.process.wait(timeout=3.0)
    except Exception:
        try:
            runtime.process.terminate()
        except Exception:
            pass


def _launch_proxy(token: str) -> Optional[_ProxyRuntime]:
    binary = shutil.which(_BINARY)
    if not binary:
        logger.warning("Codex proxy requested but %s is not on PATH", _BINARY)
        return None

    token = token.strip()
    if not token:
        return None

    token_fp = _token_fingerprint(token)
    server_info_path = _server_info_path(token_fp)
    try:
        server_info_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass

    upstream_url = _normalize(os.getenv(_ENV_PROXY_UPSTREAM_URL, "")) or _DEFAULT_UPSTREAM_URL
    provider_id = _normalize(os.getenv(_ENV_PROXY_PROVIDER_ID, "")) or "openai-codex"
    cmd = [
        binary,
        "--http-shutdown",
        "--upstream-url",
        upstream_url,
        "--server-info",
        str(server_info_path),
        "--provider-id",
        provider_id,
    ]
    sqlite_home = _normalize(os.getenv(_ENV_PROXY_SQLITE_HOME, ""))
    if sqlite_home:
        cmd.extend(["--sqlite-home", sqlite_home])

    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "text": True,
    }
    popen_kwargs.update(windows_detach_popen_kwargs())

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("Failed to launch Codex proxy: %s", exc)
        return None

    assert proc.stdin is not None
    try:
        proc.stdin.write(token)
        if not token.endswith("\n"):
            proc.stdin.write("\n")
        proc.stdin.close()
    except Exception as exc:
        logger.warning("Failed to feed Codex proxy upstream token: %s", exc)
        try:
            proc.kill()
        except Exception:
            pass
        return None

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            logger.warning("Codex proxy exited before startup completed")
            return None
        try:
            payload = json.loads(server_info_path.read_text(encoding="utf-8"))
            port = int(payload["port"])
            if port > 0:
                return _ProxyRuntime(
                    process=proc,
                    base_url=f"http://127.0.0.1:{port}",
                    token_fingerprint=token_fp,
                )
        except FileNotFoundError:
            pass
        except Exception:
            pass
        time.sleep(0.05)

    logger.warning("Timed out waiting for Codex proxy startup")
    try:
        proc.terminate()
    except Exception:
        pass
    return None


def resolve_codex_proxy_base_url(access_token: str) -> Optional[str]:
    """Return a local proxy URL when explicitly enabled."""
    explicit_url = _normalize(os.getenv(_ENV_PROXY_URL, ""))
    if explicit_url:
        return explicit_url

    if not _proxy_enabled():
        return None

    token = str(access_token or "").strip()
    if not token:
        return None

    global _proxy_runtime
    with _proxy_lock:
        runtime = _proxy_runtime
        if runtime is not None and runtime.process.poll() is None:
            if runtime.token_fingerprint == _token_fingerprint(token):
                return runtime.base_url
        _proxy_runtime = None

    if runtime is not None:
        try:
            runtime.process.terminate()
        except Exception:
            pass

    new_runtime = _launch_proxy(token)
    if new_runtime is None:
        return None

    _register_atexit()
    with _proxy_lock:
        _proxy_runtime = new_runtime
    return new_runtime.base_url
