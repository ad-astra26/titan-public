"""Dev-only Console Agent endpoints (gated by `ctx.dev_enabled`).

Two affordances for the headless build loop (RFP_titan_mobile_app §7.0):
  GET  /dev/latest.apk    serve the latest debug APK for one-tap sideloading
  POST /console/dev/log   device-logcat sink → ~/.titan/dev/device.log (tail on the box)

Both are OFF unless `python -m titan_console --dev` is passed, so they never exist
in production. They never touch the kernel and hold no Titan state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .context import Context

# debug APK path inside the nested titan-app repo (relative to the Console
# Agent install_root = the titan repo root). AGP names it "<module>-<buildtype>".
_APK_REL = ("titan-app", "android", "build", "outputs", "apk", "debug", "android-debug.apk")


def _dev_log_path() -> Path:
    return Path(os.path.expanduser("~/.titan/dev/device.log"))


def apk_path(ctx: Context) -> Path:
    return ctx.install_root.joinpath(*_APK_REL)


def serve_apk(ctx: Context) -> tuple:
    """Return (200, apk-bytes) or (404/500, {error})."""
    p = apk_path(ctx)
    try:
        if p.is_file():
            return 200, p.read_bytes()
    except OSError as e:
        return 500, {"error": f"apk read failed: {e}"}
    return 404, {"error": "no debug APK built yet — run ./gradlew :android:assembleDebug"}


def ingest_log(ctx: Context, body: bytes, *, log_path: Path | None = None) -> tuple:
    """Append posted device log lines to the dev log. Accepts {"lines":[...]} or raw text.

    Returns (200, {ok, written}) or (4xx/5xx, {error}).
    """
    if not body:
        return 400, {"error": "empty log body"}
    try:
        data = json.loads(body.decode())
        if isinstance(data, dict) and isinstance(data.get("lines"), list):
            text = "\n".join(str(x) for x in data["lines"])
        elif isinstance(data, str):
            text = data
        else:
            text = body.decode("utf-8", "replace")
    except (ValueError, UnicodeDecodeError):
        text = body.decode("utf-8", "replace")
    if not text.endswith("\n"):
        text += "\n"
    p = log_path or _dev_log_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        return 500, {"error": f"log write failed: {e}"}
    return 200, {"ok": True, "written": len(text)}
