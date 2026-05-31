"""Injectable runtime context for the Console Agent.

All side-effecting I/O (subprocess, HTTP to api_hcl) is funnelled through
callables on `Context` so every handler is unit-testable with fakes — the
agent never reaches the network or the shell in tests.
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# run(argv) -> (returncode, stdout, stderr)
Runner = Callable[[list], tuple]
# http(method, url, body, headers, timeout) -> (status, bytes)
HttpFn = Callable[..., tuple]


def default_runner(argv: list, *, timeout: float = 30.0) -> tuple:
    """Run a command, capturing output. Never raises — returns (124/127, …)."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s: {' '.join(argv)}"
    except FileNotFoundError:
        return 127, "", f"command not found: {argv[0] if argv else '?'}"
    except OSError as e:
        return 126, "", str(e)


def default_http(method: str, url: str, *, body: Optional[bytes] = None,
                 headers: Optional[dict] = None, timeout: float = 5.0) -> tuple:
    """Minimal urllib request → (status, bytes). Never raises on HTTP errors."""
    req = urllib.request.Request(url, data=body, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if hasattr(e, "read") else b""
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, json.dumps({"error": str(e)}).encode()


@dataclass
class Context:
    """Everything the handlers need, with seams for testing."""
    install_root: Path
    titan_id: str = "T1"
    api_base: str = "http://127.0.0.1:7777"
    internal_key: Optional[str] = None     # for the chat proxy
    token: Optional[str] = None            # console mutation auth (None = open on localhost)
    dist_dir: Optional[Path] = None        # built SPA bundle
    secrets_path: Optional[Path] = None    # ~/.titan/secrets.toml override (tests)
    run: Runner = default_runner
    http: HttpFn = default_http

    @property
    def manage_script(self) -> Path:
        return self.install_root / "scripts" / f"{self.titan_id.lower()}_manage.sh"

    @property
    def service_unit(self) -> str:
        return f"titan-{self.titan_id.lower()}.service"


def resolve_titan_id(install_root: Path, fallback: str = "T1") -> str:
    """Read titan_id from data/titan_identity.json (matches genesis_runner)."""
    ident = install_root / "data" / "titan_identity.json"
    try:
        if ident.exists():
            tid = json.loads(ident.read_text()).get("titan_id")
            if tid:
                return str(tid)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return fallback
