"""The Console Agent HTTP surface.

`dispatch()` is the pure, side-effect-injected router (fully unit-testable
without a socket). `ConsoleHandler` + `make_server` are the thin stdlib
adapter that wires it to http.server.

Routes (all under /console; live cognition is proxied under /console/api):
  GET  /console/health            agent self-health
  GET  /console/host              host resources (stdlib)
  GET  /console/titan-status      liveness + why-down + journal tail
  GET  /console/journal?lines=N   journal tail
  GET  /console/backups           local backup records + manifest summary
  GET  /console/backup/options    mode-aware backup info (mainnet auto / s3 / local)
  GET  /console/backup/config     off-site convenience-copy config (secrets redacted)
  GET  /console/config[?section=] settings list (value+help+editable)
  GET  /console/config/get?key=   one setting
  GET  /console/api/<v6 path>     proxied live readout (allow-listed)
  POST /console/restart           {force}            (token-gated)
  POST /console/clean-hdd         {confirm}          (token-gated)
  POST /console/config/set        {key,value}        (token-gated)
  POST /console/chat              {message,session}  (token-gated)
  POST /console/backup/config     {enabled,backend,…} off-site copy config (token-gated)
  GET  /*                         static SPA (dist_dir), index.html fallback
"""
from __future__ import annotations

import json
import posixpath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__, backup_config, config_api, dev_endpoints, ops, proxy
from .context import Context
from .host import read_host_resources
from .titan_status import titan_status

_MUTATIONS = {"/console/restart", "/console/clean-hdd", "/console/config/set",
              "/console/chat", "/console/backup/config"}


def _backup_options(ctx: Context) -> dict:
    """Mode-aware backup info (decision #15). Config WRITE lands with the
    System tab; this is the read/info side."""
    # Infer mode from genesis_on_chain-ish signals: presence of soul_keypair.enc
    # / genesis_record implies a born (mainnet/devnet) Titan.
    data = ctx.install_root / "data"
    mainnet = (data / "soul_keypair.enc").exists() or (data / "genesis_record.json").exists()
    return {
        "mainnet_sovereign": {
            "automatic": True,
            "note": "Sovereign Arweave + ZK Vault backups run AUTOMATICALLY and "
                    "must never be triggered by hand. Keep ~0.05 SOL in your "
                    "wallet for upload fees.",
            "active": mainnet,
        },
        "offsite_convenience": {
            "targets": ["s3", "local"],
            "note": "Optional off-site COPIES (not the sovereign restore path). "
                    "Choose S3 (aws CLI / boto3 + your key) or a local VPS path, "
                    "on a cron schedule. Configure via POST /console/backup/config.",
            **{k: backup_config.get_backup_config(ctx)[k]
               for k in ("configured", "enabled", "cron_schedule")},
        },
    }


def dispatch(ctx: Context, method: str, path: str, query: dict,
             body: bytes, headers: dict) -> tuple:
    """Pure router. Returns (status_int, payload) where payload is dict|bytes."""
    # ── mutation auth gate ───────────────────────────────────────────────
    if method == "POST" and path in _MUTATIONS and ctx.token:
        supplied = headers.get("x-console-token") or headers.get("X-Console-Token")
        if supplied != ctx.token:
            return 401, {"error": "missing or invalid X-Console-Token"}

    def _json_body() -> dict:
        if not body:
            return {}
        try:
            return json.loads(body.decode())
        except (ValueError, UnicodeDecodeError):
            return {}

    if method == "GET":
        if path == "/console/health":
            return 200, {"ok": True, "agent": "titan-console", "version": __version__,
                         "titan_id": ctx.titan_id}
        if path == "/console/host":
            return 200, read_host_resources()
        if path == "/console/titan-status":
            return 200, titan_status(ctx)
        if path == "/console/journal":
            from .titan_status import _journal_tail
            n = int(query.get("lines", ["50"])[0] or 50)
            return 200, {"service": ctx.service_unit, "lines": _journal_tail(ctx, n)}
        if path == "/console/backups":
            return 200, ops.list_backups(ctx)
        if path == "/console/backup/options":
            return 200, _backup_options(ctx)
        if path == "/console/backup/config":
            return 200, backup_config.get_backup_config(ctx)
        if path == "/console/config":
            section = query.get("section", [None])[0]
            return 200, config_api.list_config(ctx.install_root, section=section)
        if path == "/console/config/get":
            key = query.get("key", [""])[0]
            if not key:
                return 400, {"error": "missing ?key="}
            return 200, config_api.get_config(ctx.install_root, key)
        if path.startswith("/console/api/"):
            v6 = path[len("/console/api"):]  # → "/v6/..."
            if query:
                from urllib.parse import urlencode
                v6 = f"{v6}?{urlencode({k: v[0] for k, v in query.items()})}"
            return proxy.proxy_readout(ctx, v6)
        if path == "/console" or path.startswith("/console/"):
            return 404, {"error": f"no such console route: {path}"}
        if ctx.dev_enabled and path == "/dev/latest.apk":
            return dev_endpoints.serve_apk(ctx)
        # static SPA
        return _serve_static(ctx, path)

    if method == "POST":
        data = _json_body()
        if path == "/console/restart":
            return 200, ops.restart_titan(ctx, force=bool(data.get("force")))
        if path == "/console/clean-hdd":
            return 200, ops.clean_hdd(ctx, confirm=bool(data.get("confirm")))
        if path == "/console/config/set":
            key, val = data.get("key"), data.get("value")
            if not key:
                return 400, {"error": "missing 'key'"}
            return 200, config_api.set_config(ctx.install_root, key, str(val))
        if path == "/console/chat":
            status, payload = proxy.proxy_chat(ctx, data.get("message", ""),
                                               session=data.get("session"))
            return status, payload
        if path == "/console/backup/config":
            res = backup_config.set_backup_config(ctx, data)
            return (200 if res.get("ok") else 400), res
        if ctx.dev_enabled and path == "/console/dev/log":
            return dev_endpoints.ingest_log(ctx, body)
        return 404, {"error": f"no such console route: {path}"}

    return 405, {"error": f"method not allowed: {method}"}


def _serve_static(ctx: Context, path: str) -> tuple:
    """Serve the built SPA from dist_dir; SPA-route fallback to index.html."""
    if not ctx.dist_dir or not Path(ctx.dist_dir).is_dir():
        return 200, (b"<!doctype html><meta charset=utf-8>"
                     b"<title>TC2</title><h1>Titan Command Center agent</h1>"
                     b"<p>Agent is up. SPA bundle not built yet "
                     b"(titan-console/ ships next).</p>")
    root = Path(ctx.dist_dir).resolve()
    rel = posixpath.normpath(path.lstrip("/"))
    candidate = (root / rel).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    # path-traversal guard + SPA fallback
    if root not in candidate.parents and candidate != root and \
            not str(candidate).startswith(str(root)):
        candidate = root / "index.html"
    if not candidate.is_file():
        candidate = root / "index.html"
    try:
        return 200, candidate.read_bytes()
    except OSError:
        return 404, {"error": "not found"}


_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript",
                  ".css": "text/css", ".json": "application/json",
                  ".svg": "image/svg+xml", ".png": "image/png",
                  ".woff2": "font/woff2", ".ico": "image/x-icon"}


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = f"titan-console/{__version__}"
    ctx: Context = None  # set on the server instance

    def _handle(self, method: str) -> None:
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        headers = {k.lower(): v for k, v in self.headers.items()}
        try:
            status, payload = dispatch(self.server.ctx, method, parts.path,
                                       query, body, headers)
        except Exception as e:  # the agent must never 500-crash silently
            status, payload = 500, {"error": f"agent exception: {e}"}

        if isinstance(payload, (bytes, bytearray)):
            ext = posixpath.splitext(parts.path)[1]
            ctype = _CONTENT_TYPES.get(ext, "text/html; charset=utf-8")
            data = bytes(payload)
        else:
            ctype = "application/json"
            data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, fmt, *args):
        pass  # quiet by default; journald captures stderr if needed


def make_server(ctx: Context, host: str = "127.0.0.1", port: int = 7799) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), ConsoleHandler)
    httpd.ctx = ctx
    return httpd
