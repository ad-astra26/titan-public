"""Off-site backup config R/W for the TC² System tab (W8, decision #15).

Writes the ``[backup_offsite]`` section of ``~/.titan/secrets.toml`` and
installs/removes the cron entry that runs ``scripts/offsite_backup.py``.

Sovereignty axiom (LOCKED): this configures only the *convenience* off-site
COPY of local snapshot tarballs. The sovereign restore path (Arweave + ZK
Vault) is mainnet-only, automatic, and is never configured or triggered here.
This module refuses to expose or alter anything on the sovereign path.

stdlib-only — no Titan runtime imports, no third-party deps (tomllib reads;
a small comment-preserving section writer handles writes so we don't pull in
tomli_w and don't clobber the user's other secrets/comments).
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from .context import Context

SECTION = "backup_offsite"
_CRON_MARKER = "# TITAN_OFFSITE_BACKUP"
# fields that are credentials/secrets — redacted on read
_SECRET_FIELDS = {"aws_secret_access_key"}
_ALLOWED = {
    "enabled", "backend", "schedule_cron",
    "local_dir", "s3_bucket", "s3_prefix",
    "aws_access_key_id", "aws_secret_access_key", "aws_region",
}
_VALID_BACKENDS = {"local", "s3"}


def _secrets_path(ctx: Context) -> Path:
    # honour a test override on the context, else the real per-user path
    override = getattr(ctx, "secrets_path", None)
    return Path(override) if override else Path(os.path.expanduser("~/.titan/secrets.toml"))


def _read_section(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get(SECTION, {}) or {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _render_section(cfg: dict) -> str:
    """Render [backup_offsite] as TOML text (string values quoted, bools bare)."""
    lines = [f"[{SECTION}]"]
    for k in sorted(cfg):
        v = cfg[k]
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            esc = str(v).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k} = "{esc}"')
    return "\n".join(lines) + "\n"


def _write_section(path: Path, cfg: dict) -> None:
    """Replace (or append) the [backup_offsite] block, preserving the rest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass

    existing = path.read_text() if path.exists() else ""
    out_lines: list[str] = []
    skipping = False
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped == f"[{SECTION}]":
            skipping = True            # drop the old block
            continue
        if skipping:
            if stripped.startswith("[") and stripped.endswith("]"):
                skipping = False       # next section starts — stop dropping
            else:
                continue
        out_lines.append(line)

    body = "\n".join(out_lines).rstrip()
    block = _render_section(cfg)
    new_text = (body + "\n\n" + block) if body else block

    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(new_text)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


# ── cron ────────────────────────────────────────────────────────────────────


def _python_bin(install_root: Path) -> str:
    venv = install_root / "test_env" / "bin" / "python"
    return str(venv) if venv.exists() else "python3"


def _cron_line(ctx: Context, schedule: str) -> str:
    py = _python_bin(ctx.install_root)
    root = ctx.install_root
    log = root / "data" / "offsite_backup.log"
    cmd = (f"cd {root} && {py} scripts/offsite_backup.py "
           f"--install-root {root} >> {log} 2>&1")
    return f"{schedule} {cmd} {_CRON_MARKER}"


def _current_crontab(ctx: Context) -> list[str]:
    rc, out, _err = ctx.run(["crontab", "-l"])
    if rc != 0 or not out:
        return []
    return out.splitlines()


def _install_cron(ctx: Context, schedule: str | None) -> dict:
    """Replace any marked line; add the new one when schedule is given."""
    lines = [ln for ln in _current_crontab(ctx) if _CRON_MARKER not in ln]
    if schedule:
        lines.append(_cron_line(ctx, schedule))
    new_text = "\n".join(lines).rstrip() + "\n" if lines else "\n"

    tmp = ctx.install_root / "data" / ".crontab.offsite.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(new_text)
    try:
        rc, _out, err = ctx.run(["crontab", str(tmp)])
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if rc != 0:
        return {"ok": False, "error": f"crontab install failed: {err.strip()}"}
    return {"ok": True, "scheduled": bool(schedule)}


def _cron_present(ctx: Context) -> str | None:
    for ln in _current_crontab(ctx):
        if _CRON_MARKER in ln:
            # schedule = first 5 whitespace-separated fields
            return " ".join(ln.split()[:5])
    return None


# ── public API ────────────────────────────────────────────────────────────


def get_backup_config(ctx: Context) -> dict:
    """Current off-site config (secrets redacted) + cron schedule."""
    cfg = _read_section(_secrets_path(ctx))
    redacted = {}
    for k, v in cfg.items():
        redacted[k] = "***set***" if (k in _SECRET_FIELDS and v) else v
    return {
        "offsite": redacted,
        "configured": bool(cfg.get("backend")),
        "enabled": bool(cfg.get("enabled")),
        "cron_schedule": _cron_present(ctx),
        "backends": sorted(_VALID_BACKENDS),
        "note": "Convenience off-site COPY only — NOT the sovereign Arweave "
                "restore path (mainnet/automatic).",
    }


def set_backup_config(ctx: Context, fields: dict) -> dict:
    """Validate + persist [backup_offsite] and (re)install the cron."""
    unknown = set(fields) - _ALLOWED
    if unknown:
        return {"ok": False, "error": f"unknown fields: {sorted(unknown)}"}

    # merge over existing so partial updates (e.g. just the schedule) work
    cfg = _read_section(_secrets_path(ctx))
    cfg.update({k: v for k, v in fields.items() if v is not None})

    if "enabled" in cfg:
        cfg["enabled"] = (cfg["enabled"] if isinstance(cfg["enabled"], bool)
                          else str(cfg["enabled"]).strip().lower() in {"1", "true", "yes", "on"})

    enabled = bool(cfg.get("enabled"))
    backend = str(cfg.get("backend", "")).lower()

    if enabled:
        if backend not in _VALID_BACKENDS:
            return {"ok": False, "error": f"backend must be one of {sorted(_VALID_BACKENDS)}"}
        if backend == "local" and not cfg.get("local_dir"):
            return {"ok": False, "error": "local backend requires local_dir"}
        if backend == "s3" and not cfg.get("s3_bucket"):
            return {"ok": False, "error": "s3 backend requires s3_bucket"}
        if not cfg.get("schedule_cron"):
            cfg["schedule_cron"] = "30 4 * * *"  # daily 04:30 default

    _write_section(_secrets_path(ctx), cfg)
    cron = _install_cron(ctx, cfg.get("schedule_cron") if enabled else None)
    if not cron.get("ok"):
        return cron
    return {"ok": True, "enabled": enabled, "backend": backend,
            "cron_schedule": cfg.get("schedule_cron") if enabled else None}
