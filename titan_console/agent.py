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
  GET  /console/ops/processes     dry-run process scan (orphan-helper reapables)  (§7.2b)
  GET  /console/agent-status      console self-status (uptime/version/reachable)   (§7.2b)
  POST /console/ops/module/<action>/<name>  L2 reload|restart|enable → kernel admin (§7.2b)
  POST /console/ops/reload-api    L3 zero-downtime api reload → kernel /v4/reload-api(§7.2b)
  POST /console/ops/reboot        {confirm_phrase}  VPS reboot (primary device only)(§7.2b)
  POST /console/ops/processes/reap{pids:[…]}   kill allow-listed orphan helpers     (§7.2b)
  POST /console/ops/prune-arweave-devnet {keep,confirm}  prune devnet Arweave cache  (§7.2b)
  GET  /*                         static SPA (dist_dir), index.html fallback
"""
from __future__ import annotations

import hmac
import json
import posixpath
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import (__version__, alerts, backup_config, config_api, dev_endpoints, events,
               ops, pairing, presence, proxy)
from .context import Context
from .host import read_host_resources
from .titan_status import titan_status

_MUTATIONS = {"/console/restart", "/console/clean-hdd", "/console/config/set",
              "/console/chat", "/console/backup/config", "/console/app/heartbeat",
              "/console/events/respond", "/console/context", "/console/presence/settings"}
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


# RFP_titan_mobile_app §7.2b — a module name must be a safe identifier AND live in the
# kernel roster. The kernel is the source of truth for which modules exist.
_MODULE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# action → (kernel path template, method, optional query). reload/restart/enable mirror
# the verified kernel admin endpoints (RFP §7.2b matrix, 2026-06-20).
_MODULE_OPS = {
    "reload":  ("/v6/admin/reload-module/{name}", "POST", None),
    "restart": ("/v6/admin/restart-module/{name}", "POST", "spawn=true"),
    "enable":  ("/v6/system/guardian/enable/{name}", "POST", None),
}


def _module_in_roster(ctx: Context, name: str) -> bool:
    """True iff `name` is a live kernel WORKER module (per /v6/readiness `modules[].name`).

    NOTE: the worker roster is /v6/readiness — NOT /v6/nervous-system (that returns the 11
    cognitive axes REFLEX/FOCUS/… , no workers). Verified live 2026-06-20."""
    status, payload = proxy.proxy_readout(ctx, "/v6/readiness")
    if status != 200 or not isinstance(payload, dict):
        return False
    return any(isinstance(m, dict) and m.get("name") == name
               for m in payload.get("modules", []))


def _handle_module_op(ctx: Context, action: str, name: str) -> tuple:
    """Validate + proxy an L2 module op (reload/restart/enable) to the kernel (§7.2b)."""
    if action not in _MODULE_OPS:
        return 404, {"error": f"unknown module action: {action}"}
    if not _MODULE_NAME_RE.match(name or ""):
        return 400, {"error": f"invalid module name: {name!r}"}
    if not _module_in_roster(ctx, name):
        return 404, {"error": f"unknown module (not in kernel roster): {name}"}
    template, kmethod, kquery = _MODULE_OPS[action]
    kernel_path = template.format(name=name)
    if kquery:
        kernel_path = f"{kernel_path}?{kquery}"
    return proxy.proxy_admin(ctx, kernel_path, method=kmethod)


def dispatch(ctx: Context, method: str, path: str, query: dict,
             body: bytes, headers: dict, is_local: bool = True) -> tuple:
    """Pure router. Returns (status_int, payload) where payload is dict|bytes.

    `is_local` is True when the TCP peer is loopback (computed in `_handle` from
    `client_address`). It drives AD-5: beyond localhost EVERY route is auth-gated.
    Default True so localhost behavior + the 6-arg test convention are unchanged.
    """
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

    def _operator_token_valid() -> bool:
        # STRICT: a token must be configured AND match. The "open when no token"
        # escape (above) is a localhost-only convenience — it MUST NOT apply remotely.
        return bool(ctx.token) and (
            (headers.get("x-console-token") or headers.get("X-Console-Token")) == ctx.token)

    # ── AD-5 (AG4/AG-TLS): beyond localhost, EVERY route requires a verified device
    # signature OR a strictly-valid operator token (reads too). The only exemption is
    # the unauthenticated bootstrap /console/pair/submit (single-use-token-gated in-body).
    if not is_local and path != "/console/pair/submit":
        if not (device_authed or _operator_token_valid()):
            return 401, {"error": "device signature or operator token required (AD-5)"}

    if path in _PAIR_OPERATOR:
        if not _operator_ok():
            return 401, {"error": "operator token required for pairing control"}
    elif method == "POST" and (path in _MUTATIONS or path.startswith("/console/ops/")) \
            and not device_authed:
        # Every /console/ops/* POST is a privileged §7.2b op — gate it like a mutation
        # even on localhost (the prefix covers the variable /module/<action>/<name> routes).
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
        if path == "/console/presence":
            # The Maker's latest uploaded context (RFP_titan_mobile_app Phase 3 / AG6). Device-
            # authed (mirrors /console/device/me) — never anonymous. Flat readout, no cognition.
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            return 200, presence.read_latest(ctx)
        if path == "/console/presence/settings":
            # Per-sensor opt-in flags + cadence (console-local store, NOT config.toml).
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            return 200, presence.get_settings(ctx)
        if path == "/console/events":
            # Device drains its outbound queue (RFP event-channel AG-EVT-6). The non-local
            # gate already enforces a device signature remotely; require it on localhost too
            # (mirrors /console/device/me) so the route is never anonymous.
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            try:
                wait_s = int((query.get("wait", ["0"])[0] or "0"))
                since = int((query.get("since", ["0"])[0] or "0"))
            except (TypeError, ValueError):
                return 400, {"error": "wait/since must be integers"}
            try:
                evs, cursor = events.drain(ctx, headers.get("x-device-id", ""),
                                           since, max(0, wait_s))
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, {"events": evs, "cursor": cursor}
        if path == "/console/events/inbox":
            # The co-located kernel consumer drains phone→kernel responses/feedback for a
            # device (RFP §7.3). Local-only + internal-key, mirroring /console/events/enqueue.
            if not is_local:
                return 403, {"error": "inbox drain is local-only"}
            expected = alerts.resolve_internal_key(ctx)
            provided = headers.get("x-titan-internal-key", "")
            if not expected or not hmac.compare_digest(provided, expected):
                return 401, {"error": "internal key required"}
            device_id = (query.get("device_id", [""])[0] or "")
            try:
                since = int((query.get("since", ["0"])[0] or "0"))
            except (TypeError, ValueError):
                return 400, {"error": "since must be an integer"}
            try:
                items, cursor = events.drain_inbox(ctx, device_id, since)
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, {"items": items, "cursor": cursor}
        if path == "/console/ops/processes":
            # Advanced ops: ALWAYS-dry-run process scan (§7.2b decision-b). Device-authed.
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            return 200, ops.scan_processes(ctx)
        if path == "/console/agent-status":
            # Console self-status — the app polls this to detect the console coming back
            # after a VPS reboot (§7.2b worked example). Device-authed.
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            return 200, ops.agent_status(ctx)
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
            # RFP_affective_grounding_loop §7.D (D.2): a `device_authed` chat is a
            # cryptographically-verified Maker (the phone's pairing-Ed25519 sig was
            # checked against the maker_pubkey-bound binding) → relay it so the
            # Titan fires the app-channel maker_bond. A local web-UI owner chat is
            # NOT device_authed → no bond (honest).
            status, payload = proxy.proxy_chat(ctx, data.get("message", ""),
                                               session=data.get("session"),
                                               maker_verified=device_authed)
            return status, payload
        if path == "/console/backup/config":
            res = backup_config.set_backup_config(ctx, data)
            return (200 if res.get("ok") else 400), res
        if path == "/console/pair/start":
            return pairing.mint_pairing(ctx, public_url=data.get("public_url"),
                                        mode=data.get("mode"))
        if path == "/console/pair/submit":
            return pairing.submit_device(ctx, data)
        if path == "/console/pair/confirm":
            return pairing.confirm_device(ctx, data.get("pairing_token", ""), data.get("code", ""),
                                          maker_pubkey=data.get("maker_pubkey"),
                                          maker_sig=data.get("maker_sig"))
        if path == "/console/events/enqueue":
            # Titan's cognition / HealthMonitor push an event to a paired phone. This is an
            # INTERNAL, co-located call (kernel + console share the box): local-only + the
            # internal key (RFP decision #1). No remote party — not even the app — enqueues.
            if not is_local:
                return 403, {"error": "enqueue is local-only"}
            expected = alerts.resolve_internal_key(ctx)
            provided = headers.get("x-titan-internal-key", "")
            if not expected or not hmac.compare_digest(provided, expected):
                return 401, {"error": "internal key required"}
            device_id = data.get("device_id", "")
            if not pairing._find_device(ctx, device_id):
                return 404, {"error": "unknown device"}
            try:
                evt = events.enqueue(ctx, device_id,
                                     type=str(data.get("type", "message")),
                                     payload=data.get("payload"),
                                     urgency=str(data.get("urgency", "normal")),
                                     dedupe_key=data.get("dedupe_key"))
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, {"ok": True, "seq": evt["seq"]}
        if path == "/console/events/respond":
            # Maker taps a Channel-2 action button or a feedback chip (RFP §7.3). Device-authed
            # (in _MUTATIONS). Lands durably in the per-device inbox; the kernel consumes it when
            # up (AG-EVT-3 — survives a kernel-down because the Console Agent is the crash domain).
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            kind = str(data.get("kind", ""))
            if kind not in ("action", "feedback"):
                return 400, {"error": "kind must be 'action' or 'feedback'"}
            item = {"kind": kind, "in_reply_to": data.get("in_reply_to"),
                    "action_id": data.get("action_id"), "reaction": data.get("reaction"),
                    "stars": data.get("stars")}
            try:
                stored = events.append_inbox(ctx, headers.get("x-device-id", ""), item)
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, {"ok": True, "seq": stored["seq"]}
        if path == "/console/events/inbox/ack":
            # The kernel consumer prunes inbox items it has read. Local-only + internal-key.
            if not is_local:
                return 403, {"error": "inbox ack is local-only"}
            expected = alerts.resolve_internal_key(ctx)
            provided = headers.get("x-titan-internal-key", "")
            if not expected or not hmac.compare_digest(provided, expected):
                return 401, {"error": "internal key required"}
            cursor = data.get("cursor")
            if not isinstance(cursor, int) or isinstance(cursor, bool):
                return 400, {"error": "cursor must be an integer"}
            try:
                pruned = events.ack_inbox(ctx, data.get("device_id", ""), cursor)
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, {"ok": True, "pruned": pruned}
        if path == "/console/app/heartbeat":
            # Phone reports presence (+ acks delivered events). Device-authed (in _MUTATIONS).
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            device_id = headers.get("x-device-id", "")
            try:
                presence.put(ctx, device_id, data)
                ack_cursor = data.get("ack_cursor")
                if isinstance(ack_cursor, int) and not isinstance(ack_cursor, bool):
                    events.ack(ctx, device_id, ack_cursor)
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, {"ok": True}
        if path == "/console/context":
            # Phone uploads opt-in-gated context samples (RFP_titan_mobile_app Phase 3 / AG6).
            # Device-authed (in _MUTATIONS). Each field is gated by its per-sensor flag; nothing
            # un-opted-in is stored. STOPS at persistence — no cognition (AG8).
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            device_id = headers.get("x-device-id", "")
            try:
                res = presence.ingest(ctx, device_id, data.get("samples"))
            except ValueError as e:
                return 400, {"error": str(e)}
            return 200, res
        if path == "/console/presence/settings":
            # Phone sets its per-sensor opt-in flags / cadence (console-local store). Device-authed.
            if not device_authed:
                return 401, {"error": "valid device signature required"}
            return 200, presence.set_settings(ctx, data)
        if path.startswith("/console/ops/module/"):
            # L2 worker op: /console/ops/module/<action>/<name> (§7.2b matrix). Auth handled
            # by the /console/ops/ mutation gate above; name is roster-checked in the helper.
            rest = path[len("/console/ops/module/"):].split("/", 1)
            if len(rest) != 2 or not rest[1]:
                return 400, {"error": "expected /console/ops/module/<action>/<name>"}
            return _handle_module_op(ctx, rest[0], rest[1])
        if path == "/console/ops/reload-api":
            # L3 api reload (§7.2b). Canonical kernel path is /v6/admin/reload-api
            # (post_v4_reload_api → commands.reload_api, v6.py:389); the /v4/ alias
            # 308-redirects and a redirected POST silently no-ops the body.
            return proxy.proxy_admin(ctx, "/v6/admin/reload-api", method="POST")
        if path == "/console/ops/reboot":
            # Host VPS reboot — the most destructive op. §7.2b decision-a: device-authed
            # (mutation gate) AND a *primary* device AND a typed confirm phrase. A
            # non-primary or operator-token-only caller is refused here.
            device_id = headers.get("x-device-id", "")
            if not (device_authed and pairing.is_primary_device(ctx, device_id)):
                return 403, {"error": "reboot requires a primary paired device"}
            return 200, ops.reboot(ctx, confirm_phrase=str(data.get("confirm_phrase", "")))
        if path == "/console/ops/processes/reap":
            # Kill specific orphan-helper PIDs the app picked from a prior dry-run scan.
            # ops.reap_processes re-classifies each PID at kill time (allowlist, fail-closed).
            pids = data.get("pids")
            if not isinstance(pids, list) or not pids:
                return 400, {"error": "body must carry a non-empty 'pids' list "
                             "(confirm specific PIDs from /console/ops/processes)"}
            return 200, ops.reap_processes(ctx, pids=pids)
        if path == "/console/ops/prune-arweave-devnet":
            keep = data.get("keep", 5)
            try:
                keep = int(keep)
            except (TypeError, ValueError):
                return 400, {"error": "keep must be an integer"}
            return 200, ops.prune_arweave_devnet(ctx, keep=keep,
                                                 confirm=bool(data.get("confirm")))
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
        peer = self.client_address[0] if self.client_address else ""
        is_local = _is_loopback(peer)
        try:
            status, payload = dispatch(self.server.ctx, method, parts.path,
                                       query, body, headers, is_local)
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


def _is_loopback(host: str) -> bool:
    """True for the loopback peer (127.0.0.0/8, ::1, IPv4-mapped loopback). Drives AD-5."""
    return host in ("::1", "localhost") or host.startswith("127.") or host.startswith("::ffff:127.")


def make_server(ctx: Context, host: str = "127.0.0.1", port: int = 7799,
                tls: tuple | None = None) -> ThreadingHTTPServer:
    """Build the agent server. `tls=(cert_path, key_path)` wraps the listening socket
    in a TLS-server context (AG-TLS/AD-9) so phones reach a confidential, forward-secret,
    pinned channel on a bare IP. None ⇒ plain HTTP (localhost dev / tests)."""
    httpd = ThreadingHTTPServer((host, port), ConsoleHandler)
    httpd.ctx = ctx
    if tls is not None:
        from . import tls as _tls
        cert, key = tls
        httpd.socket = _tls.server_ssl_context(cert, key).wrap_socket(
            httpd.socket, server_side=True)
    return httpd
