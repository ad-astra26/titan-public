"""titan_hcl/health/social_x.py — MVP first plugin for the L3
health-monitor framework.

Per rFP_health_monitor_worker.md + SPEC v1.12.0 D-SPEC-67 (2026-05-17).

Closes the silent X-posting failure surfaced 2026-05-16: twitterapi.io
returned "could not extract tweet_id" → verifier false-matched old
tweets → social_x.db marked posts "verified" → X timeline showed nothing
for 8h → user noticed manually.

Two layers (Layer 2 "auth" collapsed into Layer 3 per Maker Q5 fallback
captured in SPEC §21 D-SPEC-67 — no twitterapi.io endpoint exists today
that requires session-cookies but doesn't post; refresh_session heal
still fires when posting-layer detects zero verified-posts-in-6h):

  Layer "pipeline" — twitterapi.io reachability via X-API-Key-only GET.
    Non-mutating, no session required. status=OK on HTTP 200,
    DOWN otherwise. heal_recommended=False (network failures aren't
    healable from here; escalation is operator-visible via the bus event).

  Layer "posting" — local social_x.db actions count for verified posts
    in the last 6h. status=OK on ≥1, DEGRADED on 0. heal_recommended=True
    on DEGRADED — emits HEAL_REQUEST(action="refresh_session") to
    social_worker which calls SocialXGateway._refresh_session against
    its live in-proc session state.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from titan_hcl.health import (
    HealthCheckPlugin,
    HealthResult,
)

logger = logging.getLogger("health.social_x")

_PIPELINE_PROBE_URL = "https://api.twitterapi.io/twitter/user/info"
_PIPELINE_PROBE_TIMEOUT_S = 10.0
_POSTING_WINDOW_HOURS = 6.0
_DEFAULT_BOT_USER_NAME = "iamtitanai"
_DEFAULT_SOCIAL_X_DB_PATH = "data/social_x.db"


class SocialXHealthCheck(HealthCheckPlugin):
    """X (Twitter via twitterapi.io) pipeline + posting health check.

    Only runs on the canonical poller Titan (T1 by default). Other Titans
    skip this plugin at registry-load via the applies_on filter.
    """

    name = "social_x"
    # Each Titan posts independently via its own SocialXGateway and writes to
    # its own local data/social_x.db. So each Titan should monitor its own
    # X health. The "canonical_poller" pattern is specifically for POLLING X
    # mentions/snapshots (avoiding 3x API spend) — POSTING is per-Titan.
    # Verified live 2026-05-17: T1 db has 486 own + 217 T2-posts + 158 T3-posts
    # (cross-Titan visibility from canonical poll); T2 db has 12 own; T3 has 13
    # own — confirming per-Titan posting and per-Titan DB writes.
    applies_on = "all"
    # MUST match the ModuleSpec name social_worker subscribes under
    # (verified live 2026-05-17: bus broker subscriber name is "social_worker",
    # NOT "social" — SOCIAL_CATALYST works via dst="all"+topic filter, not
    # via a "social" alias). HEAL_REQUEST is targeted dst=owning_worker, so
    # this MUST be the exact subscriber name or the message is silently dropped.
    owning_worker = "social_worker"

    # 4h cadence — X-policy-friendly and matches the rhythm at which a
    # silent failure would meaningfully accumulate (a 6h dry-spell on the
    # posting layer is the soonest we'd want to alarm).
    cadence_s = 14400.0

    # Heal caps + cooldowns. refresh_session is cheap + idempotent but
    # we still gate it to avoid hammering the upstream session endpoint.
    max_heal_attempts_per_24h = 6
    heal_cooldown_after_success_s = 3600.0    # 1h after successful refresh
    heal_cooldown_after_failure_s = 1800.0    # 30min after a failed refresh

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        # Each Titan filters posts to its own titan_id so cross-Titan rows
        # (T1's canonical-poller visibility into T2+T3 posts) don't falsely
        # flip T1's posting layer to OK when T1 itself has gone silent.
        # resolve_titan_id() reads /etc/titan/titan_id or env per
        # feedback_titan_id_canonical_resolve.md (R-PORT-1).
        try:
            from titan_hcl.core.state_registry import resolve_titan_id
            self._titan_id = resolve_titan_id() or ""
        except Exception:
            # Test path — let ctor config override; falls back to empty.
            self._titan_id = ""
        if (config or {}).get("titan_id"):
            # Test fixture override.
            self._titan_id = str(config["titan_id"])
        # api_key MUST come from the secrets-merged config (Layer 3 =
        # ~/.titan/secrets.toml per feedback_external_secrets.md). The
        # raw config dict passed to the worker (whatever `self._full_config`
        # was at boot) does NOT include the secrets layer, so direct
        # `self.config["api_key"]` is empty in production. Mirror
        # SocialXGateway._refresh_session pattern: call
        # `load_titan_config(force_reload=True)` to get the 3-layer merge.
        # Layer-1+2 fallbacks (user_name, db_path) can come from either.
        try:
            from titan_hcl.config_loader import load_titan_config
            full_cfg = load_titan_config(force_reload=True)
        except Exception:
            full_cfg = {}
        sx_secrets = (full_cfg.get("social_x") or {}) if full_cfg else {}
        # twitterapi.io key resolution mirrors SocialXGateway production
        # wiring exactly: `sx.api_key` first, then `[sage].twitterapi_io_key`
        # fallback (logic/social_x_gateway.py:473). Without this fallback
        # the plugin's pipeline probe always reports
        # `api_key_missing_in_config` because the canonical key today
        # lives under [sage], not [social_x] (verified live on T1 2026-05-17).
        # Section name is `stealth_sage` per logic/social_x_gateway.py:449
        # (NOT `sage` — common naming confusion; verified live 2026-05-17).
        sage_secrets = (full_cfg.get("stealth_sage") or {}
                         ) if full_cfg else {}
        # Allow ctor `config` to override secrets (test fixture path —
        # tests pass a synthetic config dict + don't want secrets-file IO).
        sx_passed = self.config or {}
        self._user_name = (
            sx_passed.get("user_name")
            or sx_secrets.get("user_name")
            or _DEFAULT_BOT_USER_NAME)
        self._api_key = (
            sx_passed.get("api_key")
            or sx_secrets.get("api_key")
            or sage_secrets.get("twitterapi_io_key")
            or "")
        self._db_path = (
            sx_passed.get("db_path")
            or sx_secrets.get("db_path")
            or _DEFAULT_SOCIAL_X_DB_PATH)
        # Resolve absolute path against project root for cross-cwd safety.
        if not Path(self._db_path).is_absolute():
            self._db_path = str(
                (Path(__file__).resolve().parents[2] / self._db_path))

    def check(self) -> list[HealthResult]:
        results: list[HealthResult] = []
        results.append(self._check_pipeline())
        results.append(self._check_posting())
        return results

    def heal(self, last_result: HealthResult
             ) -> tuple[str | None, dict]:
        # Only the posting layer recommends heal today. Pipeline DOWN is
        # a network/upstream issue — surfacing it on the bus is enough.
        if (last_result.layer == "posting"
                and last_result.heal_recommended):
            return "refresh_session", {
                "trigger": "verified_posts_6h_zero",
                "reason": last_result.reason,
            }
        return None, {}

    # ── Layer probes ────────────────────────────────────────────────

    def _check_pipeline(self) -> HealthResult:
        """Hit twitterapi.io with X-API-Key only — no session, no mutation."""
        if not self._api_key:
            return HealthResult(
                plugin=self.name, layer="pipeline", status="DOWN",
                reason="api_key_missing_in_config",
                details={},
                heal_recommended=False,
            )
        try:
            # Lazy import: keep health_monitor_worker cold-import cheap;
            # requests is already a project dep used everywhere in
            # SocialXGateway.
            import requests
            start = time.time()
            resp = requests.get(
                _PIPELINE_PROBE_URL,
                params={"userName": self._user_name},
                headers={"X-API-Key": self._api_key},
                timeout=_PIPELINE_PROBE_TIMEOUT_S,
            )
            latency_ms = (time.time() - start) * 1000.0
            ok = resp.status_code == 200
            return HealthResult(
                plugin=self.name, layer="pipeline",
                status="OK" if ok else "DOWN",
                reason=("ok" if ok
                        else f"twitterapi_io_http_{resp.status_code}"),
                details={
                    "latency_ms": round(latency_ms, 1),
                    "http_status": resp.status_code,
                    "user_name": self._user_name,
                },
                heal_recommended=False,
            )
        except Exception as e:
            return HealthResult(
                plugin=self.name, layer="pipeline", status="DOWN",
                reason=f"probe_exception:{type(e).__name__}",
                details={"exception": str(e)[:200]},
                heal_recommended=False,
            )

    def _check_posting(self) -> HealthResult:
        """Count verified posts in last 6h from local social_x.db."""
        cutoff = time.time() - (_POSTING_WINDOW_HOURS * 3600.0)
        try:
            if not Path(self._db_path).exists():
                # DB absent is a real DEGRADED — but not healable by
                # refresh_session; surface it for operator visibility.
                return HealthResult(
                    plugin=self.name, layer="posting", status="DEGRADED",
                    reason="social_x_db_missing",
                    details={"db_path": self._db_path},
                    heal_recommended=False,
                )
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            try:
                # Filter by titan_id so each Titan only counts its OWN
                # verified posts. T1 sees cross-Titan rows from canonical
                # polling — without this filter, T1's posting layer would
                # falsely show OK whenever T2 or T3 posted, even if T1
                # itself had gone silent for 24h.
                #
                # Phase 1.9 (2026-05-17): use COALESCE across timestamp
                # columns because `posted_at` is NULL for many verified
                # rows in production. SocialXGateway has two write paths:
                #   - logic/social_x_gateway.py:707 sets (status, tweet_id,
                #     posted_at) — the "post-then-verify" flow
                #   - logic/social_x_gateway.py:717+722 set (status,
                #     verified_at) — the "verify-only" flow that bypasses
                #     posted_at
                # The verify-only path is what's used today for most posts,
                # leaving posted_at NULL even though the post landed and was
                # verified on X. Filtering on posted_at alone returned 0
                # despite 5 verified posts/6h on T1 — false silent-failure
                # alarm against a healthy pipeline (verified live 2026-05-17).
                # COALESCE picks the first non-NULL: verified_at first
                # (most accurate "when did verification complete"), then
                # posted_at (older path), then created_at (always set).
                if self._titan_id:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM actions "
                        "WHERE status = 'verified' "
                        "AND COALESCE(verified_at, posted_at, created_at) "
                        ">= ? AND titan_id = ?",
                        (cutoff, self._titan_id))
                else:
                    # No titan_id resolved (test path or boot race) —
                    # fall back to unfiltered count.
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM actions "
                        "WHERE status = 'verified' "
                        "AND COALESCE(verified_at, posted_at, created_at) "
                        ">= ?",
                        (cutoff,))
                n_verified = int(cur.fetchone()[0] or 0)
            finally:
                conn.close()
            ok = n_verified >= 1
            return HealthResult(
                plugin=self.name, layer="posting",
                status="OK" if ok else "DEGRADED",
                reason=f"verified_posts_6h={n_verified}",
                details={
                    "verified_posts_6h": n_verified,
                    "window_hours": _POSTING_WINDOW_HOURS,
                    "titan_id": self._titan_id or "<unfiltered>",
                },
                # Heal only when zero posts AND the pipeline was OK —
                # but we can't reference the pipeline result from here;
                # the worker emits HEAL_REQUEST only when heal_recommended=
                # True, so we err on the side of attempting a refresh
                # (it's idempotent + gated by daily cap upstream).
                heal_recommended=(not ok),
            )
        except Exception as e:
            return HealthResult(
                plugin=self.name, layer="posting", status="DEGRADED",
                reason=f"probe_exception:{type(e).__name__}",
                details={"exception": str(e)[:200]},
                heal_recommended=False,
            )


__all__ = ("SocialXHealthCheck",)
