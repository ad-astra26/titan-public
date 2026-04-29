"""
titan_plugin/api/state_mapping.py — canonical map from legacy `plugin.X.Y`
patterns to TitanStateAccessor `titan_state.X.Y` targets.

Microkernel v2 Phase A §A.4 S5 amendment (2026-04-25), Phase 3.

This is the single source of truth for the codemod. Each entry is a
TUPLE of (source_path_parts, target_replacement_str).

The codemod walks every Attribute/Call chain rooted at `plugin` (or its
aliases like `request.app.state.titan_plugin`) and looks up the chain
in this map. Match found → replace with the target. No match → leave
in place (Category C, manual handling).

Source path encoding:
  Plain attribute:    ("plugin", "network", "pubkey")
  Subscript literal:  ("plugin", "_proxies", "[spirit]")
  Method call literal arg:  ("plugin", "_proxies", "get", "(spirit)")

Target encoding:
  Plain string that replaces the chain. Trailing parens are preserved
  by the codemod when the original was a call.
"""
from __future__ import annotations


# ── Direct sub-accessor mappings ──────────────────────────────────────
# `plugin.X` → `titan_state.X` for every sub-accessor on TitanStateAccessor.
# These cover the simple top-level attribute reads.

DIRECT_SUB_ACCESSORS = [
    "network", "soul", "guardian", "memory", "metabolism",
    "studio", "social", "gatekeeper", "mood_engine",
    "agency", "backup", "recorder", "language", "meta_teacher",
    "cgn", "reasoning", "dreaming", "trinity", "neuromods",
    "epoch", "spirit", "body", "mind", "identity",
    "rl", "llm", "media", "timechain", "config_loader",
    "params", "persistence", "sovereignty",
]


# ── _proxies.get("X") ↔ titan_state.X ─────────────────────────────────
# The most common pattern (~32 callsites). Sets the variable that's
# then used as `proxy.method()` further down.

PROXY_NAMES = [
    "spirit", "body", "mind", "memory", "rl", "llm", "media",
    "timechain", "metabolism", "sovereignty", "studio", "social",
    "mood_engine", "gatekeeper", "social_graph",
]


# ── Special-case patterns ─────────────────────────────────────────────
# Tuple of (source_chain, target_str). Codemod matches the longest chain
# that's a prefix of the actual usage. Order matters — longest first.

SPECIAL_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    # Soul private attrs → SoulAccessor properties
    (("plugin", "soul", "_maker_pubkey"), "titan_state.soul.maker_pubkey"),
    (("plugin", "soul", "_nft_address"), "titan_state.soul.nft_address"),
    (("plugin", "soul", "current_gen"), "titan_state.soul.current_gen"),
    (("plugin", "soul", "evolve_soul"), "titan_state.commands.evolve_soul"),
    (("plugin", "soul", "get_active_directives"),
     "titan_state.soul.get_active_directives"),

    # Network → NetworkAccessor (pre-cached, no async needed)
    (("plugin", "network", "pubkey"), "titan_state.network.pubkey"),
    (("plugin", "network", "rpc_urls"), "titan_state.network.rpc_urls"),
    (("plugin", "network", "premium_rpc"), "titan_state.network.premium_rpc"),
    (("plugin", "network", "get_balance"), "titan_state.network.balance"),
    (("plugin", "network", "get_raw_account_data"),
     "titan_state.network.get_raw_account_data"),

    # Soul/identity bus-cached paths
    (("plugin", "_full_config"), "titan_state.config.full"),
    (("plugin", "config_loader"), "titan_state.config_loader"),

    # Guardian
    (("plugin", "guardian", "get_status"), "titan_state.guardian.get_status"),
    (("plugin", "guardian", "get_modules_by_layer"),
     "titan_state.guardian.get_modules_by_layer"),
    (("plugin", "guardian", "layer_stats"), "titan_state.guardian.get_status"),
    (("plugin", "guardian", "start"), "titan_state.commands.guardian_start"),
    (("plugin", "guardian", "enable"), "titan_state.commands.guardian_start"),

    # Memory proxy methods (post-_proxies.get unwrap, the chains will be:
    # titan_state.memory.X — these were previously plugin._proxies.get("memory").X)
    # Already handled by MemoryAccessor sub-accessor; codemod handles the prefix.

    # Plugin-private attrs (cache-resident — kernel publishes these as bus events)
    (("plugin", "_limbo_mode"), 'titan_state.cache.get("plugin._limbo_mode", False)'),
    (("plugin", "_dream_inbox"), 'titan_state.cache.get("plugin._dream_inbox", [])'),
    (("plugin", "_current_user_id"), 'titan_state.cache.get("plugin._current_user_id", "")'),
    (("plugin", "_pending_self_composed"),
     'titan_state.cache.get("plugin._pending_self_composed", None)'),
    (("plugin", "_pending_self_composed_confidence"),
     'titan_state.cache.get("plugin._pending_self_composed_confidence", 0.0)'),
    (("plugin", "_last_execution_mode"),
     'titan_state.cache.get("plugin._last_execution_mode", "")'),
    (("plugin", "_last_commit_signature"),
     'titan_state.cache.get("plugin._last_commit_signature", "")'),
    (("plugin", "_last_research_sources"),
     'titan_state.cache.get("plugin._last_research_sources", [])'),
    (("plugin", "_start_time"), 'titan_state.cache.get("plugin._start_time", 0.0)'),
    (("plugin", "_is_meditating"),
     'titan_state.cache.get("plugin._is_meditating", False)'),
    (("plugin", "get_v3_status"), 'titan_state.cache.get("v3.status", lambda: {})'),

    # Recorder
    (("plugin", "recorder", "buffer"),
     'titan_state.cache.get("recorder.buffer", [])'),

    # Bus → commands
    (("plugin", "bus", "publish"), "titan_state.commands.publish"),
    (("plugin", "bus", "request"), "titan_state.commands.publish"),

    # Agency
    (("plugin", "_agency"), "titan_state.agency"),
    (("plugin", "_agency_assessment"), "titan_state.agency"),
    (("plugin", "_interface_advisor"),
     'titan_state.cache.get("interface_advisor", {})'),

    # Reload API command
    (("plugin", "reload_api"), "titan_state.commands.reload_api"),
]


def build_simple_replacements() -> dict[str, str]:
    """Generate string-level replacements for the most common patterns.

    The codemod (libcst) handles structured edits; this dict is for the
    SAFETY-NET text replacement pass that catches tail patterns the
    structured codemod might miss (e.g. patterns inside f-strings,
    string concatenations, etc.).

    Used by: scripts/s5_endpoint_codemod.py text_pass()
    """
    repl = {}

    # _proxies.get("X") → titan_state.X
    for name in PROXY_NAMES:
        for q in ("'", '"'):
            repl[f'plugin._proxies.get({q}{name}{q})'] = f'titan_state.{name}'
            repl[f'plugin._proxies[{q}{name}{q}]'] = f'titan_state.{name}'

    # plugin._full_config patterns
    repl["plugin._full_config.get("] = "titan_state.config.get("
    repl["plugin._full_config"] = "titan_state.config.full"

    # Special patterns
    for src_parts, target in SPECIAL_PATTERNS:
        src = ".".join(src_parts)
        repl[src] = target

    # Direct sub-accessors
    for sub in DIRECT_SUB_ACCESSORS:
        repl[f"plugin.{sub}"] = f"titan_state.{sub}"

    return repl
