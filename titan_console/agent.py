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
  GET  /console/pair              operator pairing GUI (stdlib HTML)
  GET  /console/device/me         signed device self-check (paired? → 200/401)
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

from . import __version__, backup_config, config_api, dev_endpoints, ops, pairing, proxy
from .context import Context
from .host import read_host_resources
from .titan_status import titan_status

_MUTATIONS = {"/console/restart", "/console/clean-hdd", "/console/config/set",
              "/console/chat", "/console/backup/config"}
# Pairing CONTROL routes are operator-only (mint/confirm/inspect device pairings).
# /console/pair/submit is deliberately NOT here — it's the unauthenticated bootstrap,
# gated by the single-use pairing token inside pairing.submit_device.
_PAIR_OPERATOR = {"/console/pair/start", "/console/pair/confirm", "/console/pair/status"}

# Self-contained (stdlib, no deps) operator pairing page served at GET /console/pair.
# The richer graphical-QR experience lives in the Observatory panel (bundled npm qrcode);
# this page uses copy-paste so the crash-decoupled agent keeps zero third-party deps.
_PAIR_PAGE_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Pair phone · Titan Command Center</title>
<style>
 :root{--ink:#0B0E14;--sf:#141A24;--sfh:#1C2433;--cy:#4DD0E1;--vi:#9C7BFF;--tx:#E6EAF2;--mu:#8A93A6;--ok:#5BD9A6;--wn:#E7B450}
 *{box-sizing:border-box}body{margin:0;background:var(--ink);color:var(--tx);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:560px;margin:0 auto;padding:32px 20px}
 .orb{width:64px;height:64px;border-radius:50%;background:radial-gradient(circle at 35% 30%,var(--cy),var(--vi) 55%,var(--ink));margin:0 auto 18px}
 h1{font-size:22px;text-align:center;margin:0 0 4px}p.sub{color:var(--mu);text-align:center;margin:0 0 26px}
 .card{background:var(--sf);border:1px solid var(--sfh);border-radius:16px;padding:20px;margin:14px 0}
 label{display:block;color:var(--mu);font-size:13px;margin:0 0 6px}
 input{width:100%;background:var(--ink);border:1px solid var(--sfh);color:var(--tx);border-radius:10px;padding:11px 12px;font-size:15px}
 button{width:100%;border:0;border-radius:12px;padding:13px;font-size:15px;font-weight:600;cursor:pointer;margin-top:12px}
 .primary{background:var(--cy);color:var(--ink)}.violet{background:var(--vi);color:var(--ink)}.ghost{background:var(--sfh);color:var(--tx)}
 pre{background:var(--ink);border:1px dashed var(--sfh);border-radius:10px;padding:12px;white-space:pre-wrap;word-break:break-all;color:var(--cy);font-size:12px;max-height:170px;overflow:auto}
 .code{font-size:40px;font-weight:800;letter-spacing:6px;color:var(--cy);text-align:center;margin:8px 0}
 .muted{color:var(--mu);font-size:13px}.hide{display:none}.ok{color:var(--ok)}.warn{color:var(--wn)}
 .step{display:flex;gap:9px;align-items:flex-start;margin:8px 0;color:var(--mu);font-size:14px}.step b{color:var(--tx)}
</style></head><body><div class=wrap>
 <div class=orb></div><h1>Pair your phone</h1>
 <p class=sub>Carry Titan in your pocket — a signed, private remote.</p>

 <div class=card id=authCard>
   <label>Operator token <span class=muted>(from <code>~/.titan/console_token</code>; blank if local)</span></label>
   <input id=tok type=password placeholder="X-Console-Token" autocomplete=off>
   <button class=primary id=startBtn>Start pairing</button>
   <p class=muted id=err></p>
 </div>

 <div class="card hide" id=qrCard>
   <div class=step><span>1️⃣</span><div>On your phone open <b>Titan → Paste pairing code</b> and paste this:</div></div>
   <pre id=payload></pre>
   <button class=ghost id=copyBtn>Copy pairing code</button>
   <p class=muted id=waiting>Waiting for your phone to submit its key…</p>
 </div>

 <div class="card hide" id=confirmCard>
   <div class=step><span>2️⃣</span><div>Your phone shows a 6-digit code. The Console computed:</div></div>
   <div class=code id=agentCode>— — —</div>
   <label>Type the code shown on <b>your phone</b> to confirm it matches</label>
   <input id=codeIn inputmode=numeric maxlength=6 placeholder="000000">
   <button class=violet id=confirmBtn>Confirm pairing</button>
   <p class=muted id=cErr></p>
 </div>

 <div class="card hide" id=doneCard>
   <h1 class=ok>✓ Paired</h1>
   <p class=sub id=doneMsg></p>
 </div>
</div>
<script>
 const $=id=>document.getElementById(id); let token="",poll=null;
 const hdr=()=>{const h={'Content-Type':'application/json'};if(token)h['X-Console-Token']=token;return h};
 async function jpost(p,b){const r=await fetch(p,{method:'POST',headers:hdr(),body:JSON.stringify(b||{})});return[r.status,await r.json().catch(()=>({}))]}
 async function jget(p){const r=await fetch(p,{headers:hdr()});return[r.status,await r.json().catch(()=>({}))]}
 $('startBtn').onclick=async()=>{
   token=$('tok').value.trim();$('err').textContent='';
   const[s,d]=await jpost('/console/pair/start',{});
   if(s!==200){$('err').className='warn';$('err').textContent=d.error||('error '+s);return}
   $('payload').textContent=JSON.stringify(d);
   $('authCard').classList.add('hide');$('qrCard').classList.remove('hide');
   const tok=d.pairing_token;
   poll=setInterval(async()=>{
     const[ps,pd]=await jget('/console/pair/status?pairing_token='+encodeURIComponent(tok));
     if(ps===200&&pd.state==='submitted'){
       clearInterval(poll);$('agentCode').textContent=(pd.code6||'').replace(/(\\d{3})(\\d{3})/,'$1 $2');
       $('confirmCard').classList.remove('hide');$('waiting').textContent='Phone submitted ✓ — confirm below.';
       $('confirmBtn').onclick=async()=>{
         const[cs,cd]=await jpost('/console/pair/confirm',{pairing_token:tok,code:$('codeIn').value.trim()});
         if(cs===200){$('confirmCard').classList.add('hide');$('qrCard').classList.add('hide');
           $('doneCard').classList.remove('hide');$('doneMsg').textContent='“'+(cd.label||'phone')+'” can now talk to your Titan.';}
         else{$('cErr').className='warn';$('cErr').textContent=cd.error||('error '+cs)}
       };
     } else if(ps===200&&pd.state==='expired'){clearInterval(poll);$('waiting').className='warn';$('waiting').textContent='Pairing expired — reload to start again.'}
   },2000);
 };
 $('copyBtn').onclick=()=>navigator.clipboard.writeText($('payload').textContent).then(()=>$('copyBtn').textContent='Copied ✓');
</script></body></html>"""


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
    # ── auth gate (AG4): operator token OR a paired device's signature ────
    # A signed request from a registered device is authorized like the operator.
    device_authed = bool(headers.get("x-device-id")) and pairing.verify_request_signature(
        ctx, device_id=headers.get("x-device-id", ""),
        timestamp=headers.get("x-timestamp", ""),
        signature_b64=headers.get("x-signature", ""),
        method=method, path=path, body=body)

    def _operator_ok() -> bool:
        if not ctx.token:
            return True  # no token configured = open (localhost dev)
        return (headers.get("x-console-token") or headers.get("X-Console-Token")) == ctx.token

    if path in _PAIR_OPERATOR:
        if not _operator_ok():
            return 401, {"error": "operator token required for pairing control"}
    elif method == "POST" and path in _MUTATIONS and not device_authed:
        if not _operator_ok():
            return 401, {"error": "missing or invalid X-Console-Token (or device signature)"}

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
        if path == "/console/pair/status":
            return pairing.pair_status(ctx, query.get("pairing_token", [""])[0])
        if path == "/console/pair":
            # Operator pairing GUI (stdlib, no deps). The page itself is open; the
            # pair/start|status|confirm calls it makes stay operator-token-gated.
            return 200, _PAIR_PAGE_HTML.encode()
        if path == "/console/device/me":
            # Signed self-check: the app polls this to learn the operator confirmed
            # the code-match (device now in devices.json). 401 until registered.
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            rec = pairing.device_record(ctx, headers.get("x-device-id", ""))
            return (200, rec) if rec else (404, {"error": "device not registered"})
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
        if path == "/console/pair/start":
            return pairing.mint_pairing(ctx, public_url=data.get("public_url"))
        if path == "/console/pair/submit":
            return pairing.submit_device(ctx, data)
        if path == "/console/pair/confirm":
            return pairing.confirm_device(ctx, data.get("pairing_token", ""), data.get("code", ""))
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
