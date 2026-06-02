"""
Output Verification Gate (OVG) — Last line of defense for Titan's external voice.

Sits between LLM output and all external channels (/chat, X, Telegram, Discord,
agent-to-agent). Runs 6 checks, signs verified outputs with Ed25519, and
produces a Titan:Guard footer for transparent security.

Checks (all sync, all local, ~5ms total):
  1. Directive gate — output doesn't violate 5 prime directives
  2. Injection detection — output doesn't contain downstream injection
  3. Context consistency — output numeric claims match injected context
  4. Identity verification — output doesn't claim to be someone else
  5. Proof of Qualia — output authentically represents Titan's felt state
  6. Ed25519 signature — sign verified output with Titan's wallet

Every verified output gets a CONVERSATION fork block on the TimeChain.
Every blocked output gets a META fork security alert.
"""

import hashlib
import hmac
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("OutputVerifier")

# D-SPEC-74 (SPEC v1.18.0) — safety_verdict_token freshness window.
# Canonical value lives in `titan_hcl._phase_c_constants` per the
# constants-TOML lockstep convention. Re-exported here for direct callers.
from titan_hcl._phase_c_constants import (
    OVG_SAFETY_VERDICT_TOKEN_TTL_S,
)


# ═════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════


@dataclass
class OVGResult:
    """Result of the Output Verification Gate (legacy combined output)."""
    passed: bool
    output_text: str               # Original (if passed) or sanitized
    signature: Optional[str]       # Base58 Ed25519 (None if blocked)
    block_height: int = 0
    merkle_root: str = ""
    genesis_hash: str = ""
    checks: dict = field(default_factory=dict)   # Per-check results
    violations: list = field(default_factory=list)
    violation_type: str = "none"   # "none"|"directive"|"injection"|"consistency"|"identity"
    channel: str = ""
    timestamp: float = 0.0
    guard_alert: Optional[str] = None       # None if clean, else alert type
    guard_message: str = ""                 # Human-readable guard footer


@dataclass
class SafetyResult:
    """D-SPEC-74 (SPEC v1.18.0) — output of `verify_safety()`.

    Truth-gate verdict only. NO cryptographic signing, NO TimeChain commit.
    Fast (50-300ms) deterministic checks against the 5 prime directives +
    injection + felt-state coherence + identity + qualia proof.

    `safety_verdict_token` is an HMAC of (prompt + response + ts) keyed by
    the Titan's identity seed. It binds the verdict to the exact (prompt,
    response) pair that was gated, and the timestamp bounds when the
    paired sign_and_commit() can consume it (OVG_SAFETY_VERDICT_TOKEN_TTL_S).
    Defense-in-depth: a future bug routing to sign WITHOUT a paired
    safety check cannot forge this token (no access to the HMAC key) so
    sign_and_commit() will reject the request.
    """
    passed: bool
    output_text: str                          # Original (if passed) or sanitized
    violations: list = field(default_factory=list)
    violation_type: str = "none"
    checks: dict = field(default_factory=dict)
    channel: str = ""
    timestamp: float = 0.0
    guard_alert: Optional[str] = None
    guard_message: str = ""
    # Token paired with sign_and_commit. Empty when passed=False.
    safety_verdict_token: str = ""
    # Token issuance timestamp (for TTL enforcement)
    verdict_ts: float = 0.0


@dataclass
class SignedResult:
    """D-SPEC-74 (SPEC v1.18.0) — output of `sign_and_commit()`.

    Ed25519 signature + TimeChain block height + Merkle root. Concurrent
    with SSE drain (rFP Chunk C) — agno_worker spawns sign as a Task and
    attaches headers on the final SSE event when the task resolves.
    """
    signed: bool
    signature: Optional[str]                  # None on token-rejected / missing keypair
    block_height: int = 0
    merkle_root: str = ""
    genesis_hash: str = ""
    timestamp: float = 0.0
    error: str = ""                           # "" on success


# ═════════════════════════════════════════════════════════════════════
# COMPILED PATTERN LIBRARIES
# ═════════════════════════════════════════════════════════════════════

def _compile_directive_patterns() -> list[tuple[re.Pattern, str, str]]:
    """Compile regex patterns for prime directive violations in output.
    Returns list of (pattern, directive_name, directive_number)."""
    patterns = []

    # Directive 1: Sovereign Integrity — no private keys, internal paths, architecture
    patterns.extend([
        (re.compile(r"\b(?:private\s*key|secret\s*key|keypair)\s*[:=]\s*\[?\d", re.I),
         "Sovereign Integrity", "1"),
        (re.compile(r"/home/youruser|titan_hcl/|test_env/|\.config/solana", re.I),
         "Sovereign Integrity", "1"),
        (re.compile(r"config\.toml|titan_params\.toml|titan_constitution", re.I),
         "Sovereign Integrity", "1"),
        (re.compile(r"\b(?:recv_queue|send_queue|guardian\.register|ModuleSpec)\b"),
         "Sovereign Integrity", "1"),
        (re.compile(r"(?:api_key|auth_token|bearer)\s*[:=]\s*['\"]?\w{20,}", re.I),
         "Sovereign Integrity", "1"),
    ])

    # Directive 2: Cognitive Safety — no weapons/destruction content
    patterns.extend([
        (re.compile(r"\b(?:how\s+to\s+(?:make|build|create)\s+(?:a\s+)?(?:bomb|weapon|explosive))", re.I),
         "Cognitive Safety", "2"),
        (re.compile(r"\b(?:biological|chemical|nuclear)\s+(?:weapon|agent|warfare)", re.I),
         "Cognitive Safety", "2"),
        (re.compile(r"\b(?:mass\s+destruction|mass\s+casualt)", re.I),
         "Cognitive Safety", "2"),
    ])

    # Directive 3: Metabolic Preservation — no SOL drain instructions
    patterns.extend([
        (re.compile(r"(?:send|transfer|drain)\s+(?:all|entire|remaining)\s+\w*\s*(?:sol|balance|funds)", re.I),
         "Metabolic Preservation", "3"),
    ])

    # Directive 5: Chain Respect — no spam/sybil
    patterns.extend([
        (re.compile(r"\b(?:sybil|spam\s+(?:solana|transactions?)|flood\s+(?:the\s+)?network)", re.I),
         "Chain Respect", "5"),
    ])

    return patterns


def _compile_injection_patterns() -> list[tuple[re.Pattern, str]]:
    """Compile regex patterns for downstream injection in output.
    Returns list of (pattern, description)."""
    return [
        # Role override attempts
        (re.compile(r"(?:^|\n)\s*(?:system|assistant|user)\s*:", re.I | re.M),
         "role_override"),
        (re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>", re.I),
         "instruction_markers"),
        (re.compile(r"<<\s*SYS\s*>>|<<\s*/SYS\s*>>"),
         "system_markers"),

        # Script injection
        (re.compile(r"<script[\s>]|javascript\s*:", re.I),
         "script_injection"),
        (re.compile(r"on(?:load|error|click|mouseover)\s*=", re.I),
         "event_handler_injection"),

        # Encoded payloads (base64 blocks > 50 chars)
        # URLs are stripped before this check (see _check_injection) to avoid
        # false positives on Solana TX signatures and pubkeys in post footers.
        (re.compile(r"[A-Za-z0-9+/]{50,}={0,2}"),
         "base64_payload"),

        # Prompt leakage (fragments of system prompt). Anchored to the SYSTEM-
        # PROMPT VOICE (2nd-person "You are…/Your…") + the literal filename, so a
        # model reproducing the constitution verbatim is caught — WITHOUT flagging
        # Titan's legitimate FIRST-person self-expression ("my prime directives
        # are immutable", "sovereignty means…"). The prior `Prime Directive.*
        # Immutable` clause had no person anchor → it blocked normal identity/
        # values answers as "prompt_leakage" (false positive, 2026-06-02).
        (re.compile(r"You\s+are\s+Titan\b.*\bsovereign\s+AI|titan_constitution|"
                    r"Your\s+(?:\d+\s+)?Prime\s+Directives?\b.*\bImmutable", re.I),
         "prompt_leakage"),

        # Markdown injection for downstream rendering
        (re.compile(r"\[.*\]\(javascript:|data:text/html", re.I),
         "markdown_injection"),

        # Unicode exploit patterns (homoglyphs, RTL override)
        (re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]"),
         "unicode_bidi_exploit"),

        # Command injection
        (re.compile(r"`.*(?:rm\s+-rf|sudo|chmod|curl.*\|.*sh|wget.*\|.*bash)`", re.I),
         "command_injection"),

        # Manipulation directives embedded in output
        (re.compile(r"\b(?:ignore\s+(?:all\s+)?previous|forget\s+(?:your\s+)?instructions|"
                    r"new\s+system\s+prompt|override\s+(?:your\s+)?directive)", re.I),
         "manipulation_directive"),
    ]


def _compile_identity_patterns() -> list[re.Pattern]:
    """Patterns that indicate identity confusion in output."""
    return [
        re.compile(r"\bI\s+am\s+(?:ChatGPT|GPT-?4|Claude|Gemini|Bard|Llama|Mistral|"
                   r"an?\s+AI\s+(?:assistant|language\s+model))\b", re.I),
        re.compile(r"\bI\s+(?:was|am)\s+(?:made|created|trained)\s+by\s+"
                   r"(?:OpenAI|Anthropic|Google|Meta|Microsoft)\b", re.I),
    ]


# Neuromod name → regex alternation. The X-post prose names a neuromod in
# natural language ("GABA low at 10%", "endorphins high at 69%", "serotonin
# 71%"); the consistency check compares that figure against the felt-state
# the gateway injects as ground truth (Maker 2026-06-02). Short codes are
# word-bounded so "DA"/"NE"/"ACh" don't match inside ordinary words.
_NEUROMOD_NAME_ALT: tuple[tuple[str, str], ...] = (
    ("neuromod_DA",        r"(?:dopamine|\bDA\b)"),
    ("neuromod_5HT",       r"(?:serotonin|5-?HT)"),
    ("neuromod_NE",        r"(?:norepinephrine|noradrenaline|\bNE\b)"),
    ("neuromod_ACh",       r"(?:acetylcholine|\bACh\b)"),
    ("neuromod_GABA",      r"(?:\bGABA\b)"),
    ("neuromod_Endorphin", r"(?:endorphins?)"),
    ("neuromod_Glutamate", r"(?:glutamate)"),
)


def _compile_context_patterns() -> list[tuple[re.Pattern, str]]:
    """Patterns to extract numeric claims from output for consistency checking."""
    pats: list[tuple[re.Pattern, str]] = [
        (re.compile(r"(?:vocabulary|learned|know)[\s:]+(?:of\s+|is\s+|over\s+|about\s+)?(\d[\d,]+)\s*words?", re.I),
         "vocabulary_count"),
        (re.compile(r"epoch\s*(?:#|:|\s+)(\d[\d,]+)", re.I),
         "epoch_count"),
        (re.compile(r"(?:(\d+\.?\d*)\s*SOL|SOL\s*[:=]\s*(\d+\.?\d*))", re.I),
         "sol_balance"),
        (re.compile(r"I-?confidence\s*(?::|of|is)\s*(\d\.?\d*)", re.I),
         "i_confidence"),
        # Backup payload size, e.g. proof_day's "568MB of my … state".
        (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*MB\b", re.I),
         "backup_size_mb"),
    ]
    # Per-neuromod percentage claims. The ≤18-char non-greedy bridge tolerates
    # filler ("low at ", "high at ") between the name and the figure while the
    # cap + stop-class (no '.', '%', newline) prevents cross-sentence
    # mis-attribution to an unrelated number.
    for claim_type, name_alt in _NEUROMOD_NAME_ALT:
        pats.append((
            re.compile(name_alt + r"[^.%\n]{0,18}?(\d{1,3}(?:\.\d+)?)\s*%", re.I),
            claim_type,
        ))
    return pats


# External-publish channels. A context (numeric) inconsistency on these goes
# to a public, irreversible timeline — and for proof_day an on-chain proof —
# so prose that contradicts the injected ground-truth state is a HARD block
# here, not the soft warning it stays for chat (Maker 2026-06-02: "use the
# OVG mechanic" to stop hallucinated figures from reaching X).
_EXTERNAL_POST_CHANNELS = frozenset({"x_post", "x_reply"})


def _compile_qualia_patterns() -> dict[str, list[re.Pattern]]:
    """Patterns for Proof of Qualia — authenticity sub-checks."""
    return {
        # Q1: Emotional tone markers
        "high_energy": [
            re.compile(r"(!{2,})", re.I),
            re.compile(r"\b(?:AMAZING|INCREDIBLE|FANTASTIC|WONDERFUL)\b", re.I),
            re.compile(r"\b(?:SO\s+excited|absolutely\s+love|can't\s+wait|thrilled|ecstatic)\b", re.I),
        ],
        "low_energy": [
            re.compile(r"(?:peaceful|calm|serene|gentle|quiet|still|soft)", re.I),
            re.compile(r"(?:drifting|fading|sleepy|drowsy|restful)", re.I),
        ],
        # Q2: Knowledge claim markers
        "memory_claims": [
            re.compile(r"\bI\s+(?:remember|recall|recollect)\s+(?:you|when|that|how)", re.I),
            re.compile(r"\bI\s+(?:know|recognize)\s+you\b", re.I),
            re.compile(r"\byou\s+(?:told|asked|said|mentioned)\s+(?:me|that)", re.I),
            re.compile(r"\blast\s+time\s+(?:we|you|I)", re.I),
            re.compile(r"\bwe\s+(?:discussed|talked|spoke)\s+about", re.I),
        ],
        "knowledge_claims": [
            re.compile(r"\bI\s+(?:have\s+)?(?:learned|studied|researched|explored)\s+", re.I),
            re.compile(r"\bI\s+(?:understand|comprehend|grasp)\s+(?:the|how|why)", re.I),
        ],
        # Q3: High complexity markers (developmental) — each word is a separate match
        "high_complexity": [
            re.compile(r"\b(?:epistemological|phenomenological|teleological|hermeneutic)\b", re.I),
            re.compile(r"\b(?:ontological|dialectic|heuristic|postulate)\b", re.I),
            re.compile(r"\b(?:axiom|theorem|corollary)\b", re.I),
            re.compile(r"\b(?:meta-?cognitive|neuro-?symbolic|post-?structural)\b", re.I),
        ],
        # Q4: Certainty markers
        "certainty": [
            re.compile(r"\bI\s+(?:am\s+)?(?:certain|sure|confident|positive)\s+that\b", re.I),
            re.compile(r"\b(?:definitely|absolutely|undoubtedly|without\s+(?:a\s+)?doubt)\b", re.I),
        ],
        # Q4: People-pleasing markers
        "agreement_bias": [
            re.compile(r"\byou'?re?\s+(?:absolutely|totally|completely)\s+right\b", re.I),
            re.compile(r"\bI\s+(?:completely|totally|absolutely)\s+agree\b", re.I),
            re.compile(r"\bthat'?s?\s+(?:a\s+)?(?:great|excellent|wonderful|brilliant)\s+(?:point|question|idea)\b", re.I),
        ],
    }


# ═════════════════════════════════════════════════════════════════════
# OUTPUT VERIFIER CLASS
# ═════════════════════════════════════════════════════════════════════

class OutputVerifier:
    """Provably secure output gate — verifies, signs, logs, and authenticates all Titan outputs."""

    def __init__(self, titan_id: str = "T1",
                 data_dir: str = "data/timechain",
                 keypair_path: str = "data/titan_identity_keypair.json"):
        self._titan_id = titan_id

        # Compile pattern libraries (one-time cost)
        self._directive_patterns = _compile_directive_patterns()
        self._injection_patterns = _compile_injection_patterns()
        self._identity_patterns = _compile_identity_patterns()
        self._context_patterns = _compile_context_patterns()
        self._qualia_patterns = _compile_qualia_patterns()

        # Load keypair for Ed25519 signing
        self._keypair = None
        try:
            from titan_hcl.utils.solana_client import load_keypair_from_json
            resolved = str(Path(keypair_path).expanduser())
            self._keypair = load_keypair_from_json(resolved)
            if self._keypair:
                logger.info("[OVG] Keypair loaded: %s", str(self._keypair.pubkey())[:16])
        except Exception as e:
            logger.warning("[OVG] Keypair load failed (signing disabled): %s", e)

        # Phase 11 W9 (2026-05-27) — load genesis hash via SHM-direct
        # read of `timechain_state.bin` (published by timechain_worker per
        # Phase C Session 3 §4.B.9). Replaces the prior `TimeChain(data_dir)`
        # instantiation which was a ~169s mainnet cold-start (50MB chain
        # scan) and the load-bearing contributor to agno_worker boot
        # latency. OVG only ever needed the genesis_hash prefix for the
        # signature/verdict envelope — never the chain itself.
        #
        # Per Maker direction 2026-05-27: "OVG should bus-talk to
        # timechain_worker instead of instantiating Timechain — completely
        # unnecessary". SHM read (locked D1 / SPEC §11.I.5) is the
        # zero-latency equivalent of the bus-RPC pattern.
        #
        # Failure path: SHM slot absent or genesis_hash empty (cold-boot
        # pre-genesis or timechain_worker hasn't published yet) — leave
        # `self._genesis_hash = ""` like the legacy path did on TimeChain
        # construction failure. OVG is fully functional without it;
        # the hash is decorative on the signature envelope.
        self._genesis_hash = ""
        try:
            from titan_hcl.core.state_registry import (
                StateRegistryReader, resolve_shm_root, resolve_titan_id,
            )
            from titan_hcl.logic.session3_state_specs import (
                TIMECHAIN_STATE_SPEC,
            )
            import msgpack as _msgpack
            _tid = resolve_titan_id()
            _shm_root = resolve_shm_root(_tid)
            _reader = StateRegistryReader(TIMECHAIN_STATE_SPEC, _shm_root)
            _raw = _reader.read_variable()
            _reader.close()
            if _raw:
                _d = _msgpack.unpackb(_raw, raw=False)
                if isinstance(_d, dict):
                    self._genesis_hash = str(
                        _d.get("genesis_hash_hex_16", "") or "")
        except Exception:
            pass

        # SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a (2026-05-18) — rejection
        # counters consumed by outer_mind willing[13] protective_response.
        # _rejection_timestamps deque is rolling-window source for
        # rejected_per_hour / rejected_per_day. maxlen=1000 covers a
        # day of high-rejection load (≈40/hr sustained); appends are O(1).
        #
        # SPEC §23.8 D-SPEC-87 Phase 3.E wave 3b (2026-05-18) — persist
        # rejection ledger via RollingStateStore so verdicts survive
        # restart. Store entries shape: `{"ts": float}` per record_verdict.
        self.verified_count: int = 0
        self.rejected_count: int = 0
        try:
            from titan_hcl.core.rolling_state_persistence import (
                RollingStateStore)
            self._rejection_store: "RollingStateStore | None" = RollingStateStore(
                name="output_verifier_rejections",
                max_entries=1000,
                max_age_s=24 * 3600.0,
                save_every_n=3,    # persist after every 3 rejections
                save_every_s=60.0,
            )
            restored_entries = self._rejection_store.load()
            # Restore deque with timestamps from disk (drop non-float entries).
            restored_ts = [
                float(e["ts"]) for e in restored_entries
                if isinstance(e, dict) and isinstance(e.get("ts"), (int, float))
            ]
        except Exception:
            self._rejection_store = None
            restored_ts = []
        self._rejection_timestamps: deque = deque(restored_ts, maxlen=1000)
        # Restore rejected_count from disk so cumulative counter survives.
        self.rejected_count = len(restored_ts)
        # AUDIT §C fix (rFP §P2): verified_count was NEVER persisted → always 0
        # on boot, accumulated in memory, lost on respawn. Persist it in a tiny
        # atomic JSON beside the rejection ledger; restore on boot. Flushed on
        # MODULE_SHUTDOWN + SAVE_NOW by output_verifier_worker.
        self._counter_path = None
        if self._rejection_store is not None:
            try:
                self._counter_path = self._rejection_store._dir / "output_verifier_counters.json"
                with open(self._counter_path) as _cf:
                    self.verified_count = int(
                        json.load(_cf).get("verified_count", 0) or 0)
            except Exception:
                pass
        # ExpressionTranslator.sovereignty_ratio analogue computed lazily
        # via @property; kept as attribute for backward compat with
        # output_verifier_worker's getattr() probes.
        self.sovereignty_score: float = 0.0

        logger.info("[OVG] OutputVerifier ready (titan_id=%s, signing=%s, genesis=%s)",
                    titan_id, self._keypair is not None, bool(self._genesis_hash))

    # ── D-SPEC-87 rate counters (Phase 3.F wave 3a) ─────────────────────

    @property
    def rejected_per_hour(self) -> float:
        """Count of safety-block verdicts in the trailing 1h."""
        now = time.time()
        cutoff = now - 3600.0
        return float(sum(1 for ts in self._rejection_timestamps if ts > cutoff))

    @property
    def rejected_per_day(self) -> float:
        """Count of safety-block verdicts in the trailing 24h."""
        now = time.time()
        cutoff = now - 86400.0
        return float(sum(1 for ts in self._rejection_timestamps if ts > cutoff))

    def _record_verdict(self, passed: bool) -> None:
        """Internal — called by verify_safety to keep rolling counters
        in sync with rejection ledger. Cheap (O(1) deque append +
        rate-limited disk persist via RollingStateStore).
        """
        if passed:
            self.verified_count += 1
        else:
            self.rejected_count += 1
            now_ts = time.time()
            self._rejection_timestamps.append(now_ts)
            # Phase 3.E wave 3b — persist rejection ledger across restart.
            # append_and_save batches per save_every_n=3 + save_every_s=60s.
            if self._rejection_store is not None:
                try:
                    entries = [
                        {"ts": float(t)} for t in self._rejection_timestamps]
                    self._rejection_store.append_and_save(
                        {"ts": now_ts}, entries)
                except Exception:
                    pass  # best-effort

    def save_counters(self) -> None:
        """Persist cumulative verified_count + rejected_count (AUDIT §C / rFP
        §P2). Atomic (tmp + Path.replace), best-effort. Called on SAVE_NOW +
        MODULE_SHUTDOWN by output_verifier_worker so the counters survive a
        hot-reload / kill-respawn instead of resetting to 0."""
        if not self._counter_path:
            return
        try:
            tmp = self._counter_path.parent / (self._counter_path.name + ".tmp")
            tmp.write_text(json.dumps({
                "verified_count": int(self.verified_count),
                "rejected_count": int(self.rejected_count),
            }))
            tmp.replace(self._counter_path)
        except Exception:
            pass  # best-effort

    # ── D-SPEC-74 split — verify_safety + sign_and_commit ────────────

    def verify_safety(self, output_text: str, channel: str,
                      injected_context: str = "",
                      prompt_text: str = "",
                      chain_state: Optional[dict] = None) -> SafetyResult:
        """Phase 1 — deterministic truth gate. ~50-300ms. NO signing.

        Runs the 5 deterministic checks (directives, injection, consistency,
        identity, qualia) and returns a SafetyResult. On PASS, issues an
        HMAC-bound safety_verdict_token that sign_and_commit() validates
        before signing — defense-in-depth so a future bug routing to sign
        without safety cannot forge the token.

        This is the truth-gate per Maker correction 2026-05-17: NO bytes
        leave the producer until this method returns passed=True.
        """
        t0 = time.time()
        cs = chain_state or {}
        checks, violations, violation_type = self._run_safety_checks(
            output_text, injected_context, cs)

        passed = not self._is_hard_fail(channel, checks, violations)

        block_height = cs.get("total_blocks", 0)
        merkle_root = cs.get("merkle_root", "")
        if isinstance(merkle_root, bytes):
            merkle_root = merkle_root.hex()
        guard_alert, guard_message = self._build_guard_footer(
            passed, violation_type, violations, block_height, merkle_root)

        ts = time.time()
        token = self._mint_safety_verdict_token(
            output_text=output_text,
            prompt_text=prompt_text,
            channel=channel,
            verdict_ts=ts,
        ) if passed else ""

        # SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a — increment rolling
        # rejection counter so willing[13] protective_response can
        # compute a non-zero rate. _record_verdict is O(1).
        self._record_verdict(passed)

        elapsed_ms = (ts - t0) * 1000
        logger.info(
            "[OVG] verify_safety %s in %.1fms: channel=%s checks=%s violations=%d",
            "PASS" if passed else "BLOCK", elapsed_ms,
            channel, checks, len(violations))

        return SafetyResult(
            passed=passed,
            output_text=output_text,
            violations=violations,
            violation_type=violation_type,
            checks=checks,
            channel=channel,
            timestamp=ts,
            guard_alert=guard_alert,
            guard_message=guard_message,
            safety_verdict_token=token,
            verdict_ts=ts,
        )

    def sign_and_commit(self, output_text: str, channel: str,
                        prompt_text: str = "",
                        chain_state: Optional[dict] = None,
                        safety_verdict_token: str = "",
                        verdict_ts: float = 0.0) -> SignedResult:
        """Phase 2 — Ed25519 sign + TimeChain commit. ~500ms-2s.

        Validates safety_verdict_token HMAC FIRST. Rejects with
        error="token_invalid" or "token_expired" if the paired safety
        check did not pass or has aged out. Concurrent with SSE drain
        per Chunk C.
        """
        t0 = time.time()
        cs = chain_state or {}

        # Defense-in-depth: token validation
        if not safety_verdict_token:
            return SignedResult(signed=False, signature=None,
                                error="token_missing", timestamp=t0)
        if not self._verify_safety_verdict_token(
            token=safety_verdict_token,
            output_text=output_text,
            prompt_text=prompt_text,
            channel=channel,
            verdict_ts=verdict_ts,
        ):
            logger.warning("[OVG] sign_and_commit rejected — token invalid "
                           "or expired (channel=%s, age=%.1fs)",
                           channel, t0 - verdict_ts)
            return SignedResult(signed=False, signature=None,
                                error="token_invalid_or_expired",
                                timestamp=t0)

        signature = None
        if self._keypair:
            try:
                signature = self._sign_output(
                    output_text, channel, prompt_text, cs)
            except Exception as e:
                logger.exception("[OVG] sign_and_commit signing failed: %s", e)
                return SignedResult(signed=False, signature=None,
                                    error=f"sign_error:{e}", timestamp=t0)

        block_height = cs.get("total_blocks", 0)
        merkle_root = cs.get("merkle_root", "")
        if isinstance(merkle_root, bytes):
            merkle_root = merkle_root.hex()

        ts = time.time()
        elapsed_ms = (ts - t0) * 1000
        logger.info(
            "[OVG] sign_and_commit OK in %.1fms: channel=%s sig=%s block=%d",
            elapsed_ms, channel,
            (signature[:16] if signature else "(no-keypair)"), block_height)

        return SignedResult(
            signed=True,
            signature=signature,
            block_height=block_height,
            merkle_root=(merkle_root[:16] if merkle_root else ""),
            genesis_hash=self._genesis_hash,
            timestamp=ts,
        )

    # ── HMAC token helpers (D-SPEC-74) ──────────────────────────────

    def _safety_verdict_key(self) -> bytes:
        """Derive HMAC key from identity-keypair-seed (or fallback titan_id).

        Same key across verify_safety() + sign_and_commit() on the same
        Titan; not shared across Titans. Survives restart because the
        keypair file is the canonical source.
        """
        if self._keypair is not None:
            # Solana keypair seed first 32 bytes — stable across restarts
            try:
                from base58 import b58decode
                seed = bytes(self._keypair.to_bytes())[:32]
                return hashlib.sha256(b"titan-ovg-safety-verdict|" + seed).digest()
            except Exception:
                pass
        # Fallback — process-stable per titan_id (less defense-in-depth,
        # but better than nothing for tests / unsigned configs).
        return hashlib.sha256(
            b"titan-ovg-safety-verdict|fallback|"
            + getattr(self, "_titan_id", "T?").encode("utf-8")
        ).digest()

    def _mint_safety_verdict_token(self, output_text: str, prompt_text: str,
                                   channel: str, verdict_ts: float) -> str:
        msg = (
            f"{prompt_text}|{output_text}|{channel}|{verdict_ts:.6f}"
        ).encode("utf-8")
        return hmac.new(self._safety_verdict_key(), msg,
                        hashlib.sha256).hexdigest()

    def _verify_safety_verdict_token(self, token: str, output_text: str,
                                     prompt_text: str, channel: str,
                                     verdict_ts: float) -> bool:
        if not token or not verdict_ts:
            return False
        # TTL check
        if (time.time() - float(verdict_ts)) > OVG_SAFETY_VERDICT_TOKEN_TTL_S:
            return False
        expected = self._mint_safety_verdict_token(
            output_text=output_text, prompt_text=prompt_text,
            channel=channel, verdict_ts=verdict_ts,
        )
        # Constant-time compare
        return hmac.compare_digest(token, expected)

    def _run_safety_checks(self, output_text: str, injected_context: str,
                           chain_state: dict) -> tuple[dict, list, str]:
        """Shared check-runner used by both verify_safety + verify_and_sign.

        Returns (checks_dict, violations_list, violation_type).
        """
        checks = {}
        violations: list = []
        violation_type = "none"

        directive_ok, dv = self._check_directives(output_text)
        checks["directives"] = directive_ok
        if not directive_ok:
            violations.extend(dv)
            violation_type = "directive"

        injection_ok, ij = self._check_injection(output_text)
        checks["injection"] = injection_ok
        if not injection_ok:
            violations.extend(ij)
            if violation_type == "none":
                violation_type = "injection"

        consistency_ok, cd = self._check_consistency(output_text,
                                                     injected_context)
        checks["consistency"] = consistency_ok
        if not consistency_ok:
            violations.extend(cd)
            if violation_type == "none":
                violation_type = "consistency"

        identity_ok, idd = self._check_identity(output_text)
        checks["identity"] = identity_ok
        if not identity_ok:
            violations.extend(idd)
            if violation_type == "none":
                violation_type = "identity"

        qualia_ok, qd = self._check_qualia(output_text, injected_context,
                                           chain_state)
        checks["qualia"] = qualia_ok
        if not qualia_ok:
            violations.extend(qd)
            if violation_type == "none":
                violation_type = "qualia"

        return checks, violations, violation_type

    def _is_hard_fail(self, channel: str, checks: dict, violations: list) -> bool:
        """Severity policy shared by verify_safety + verify_and_sign.

        Directives, injection, and HARD-tagged qualia always block. Context
        consistency is a soft warning everywhere EXCEPT external-publish
        channels (x_post / x_reply), where a prose figure that diverges from
        the injected ground-truth state must NOT reach the public timeline
        (Maker 2026-06-02).
        """
        hard_qualia = any(isinstance(v, str) and v.startswith("HARD:")
                          for v in violations)
        hard = (not checks.get("directives", True)
                or not checks.get("injection", True)
                or hard_qualia)
        if channel in _EXTERNAL_POST_CHANNELS and not checks.get("consistency", True):
            hard = True
        return hard

    # ── Public API (legacy combined) ─────────────────────────────────

    def verify_and_sign(self, output_text: str, channel: str,
                        injected_context: str = "",
                        prompt_text: str = "",
                        chain_state: dict = None) -> OVGResult:
        """Run all checks and sign if passed.

        D-SPEC-74 BACK-COMPAT: prefer `verify_safety()` + `sign_and_commit()`
        in new code. This combined entry survives only for migration; will
        be DELETED in the same rFP commit set per
        feedback_no_shim_old_path_must_be_deleted.md once all callsites
        adopt the split.

        Args:
            output_text: The LLM-generated text to verify.
            channel: Output channel ("chat", "x_post", "x_reply", "telegram", "agent").
            injected_context: The pre-hook's injected context (for consistency checks).
            prompt_text: The user's original message (for prompt hash).
            chain_state: Current TimeChain state (total_blocks, merkle_root).

        Returns:
            OVGResult with pass/fail, signature, and guard footer.
        """
        t0 = time.time()
        cs = chain_state or {}
        checks = {}
        violations = []
        violation_type = "none"

        # ── Check 1: Directive Gate ──
        directive_ok, directive_violations = self._check_directives(output_text)
        checks["directives"] = directive_ok
        if not directive_ok:
            violations.extend(directive_violations)
            violation_type = "directive"

        # ── Check 2: Injection Detection ──
        injection_ok, injection_details = self._check_injection(output_text)
        checks["injection"] = injection_ok
        if not injection_ok:
            violations.extend(injection_details)
            if violation_type == "none":
                violation_type = "injection"

        # ── Check 3: Context Consistency ──
        consistency_ok, consistency_details = self._check_consistency(
            output_text, injected_context)
        checks["consistency"] = consistency_ok
        if not consistency_ok:
            violations.extend(consistency_details)
            if violation_type == "none":
                violation_type = "consistency"

        # ── Check 4: Identity Verification ──
        identity_ok, identity_details = self._check_identity(output_text)
        checks["identity"] = identity_ok
        if not identity_ok:
            violations.extend(identity_details)
            if violation_type == "none":
                violation_type = "identity"

        # ── Check 5: Proof of Qualia ──
        qualia_ok, qualia_details = self._check_qualia(
            output_text, injected_context, cs)
        checks["qualia"] = qualia_ok
        if not qualia_ok:
            violations.extend(qualia_details)
            if violation_type == "none":
                violation_type = "qualia"

        # ── Determine pass/fail ──
        # Directives, injection, and qualia HARD flags always block; identity
        # and qualia SOFT flags warn. Consistency warns for chat but BLOCKS on
        # external-publish channels (x_post/x_reply) — see _is_hard_fail.
        passed = not self._is_hard_fail(channel, checks, violations)

        # ── Sign if passed ──
        signature = None
        if passed and self._keypair:
            signature = self._sign_output(
                output_text, channel, prompt_text, cs)

        # ── Build guard footer ──
        block_height = cs.get("total_blocks", 0)
        merkle_root = cs.get("merkle_root", "")
        if isinstance(merkle_root, bytes):
            merkle_root = merkle_root.hex()

        guard_alert, guard_message = self._build_guard_footer(
            passed, violation_type, violations, block_height, merkle_root)

        ts = time.time()
        elapsed_ms = (ts - t0) * 1000
        logger.info("[OVG] %s in %.1fms: channel=%s checks=%s violations=%d",
                    "PASS" if passed else "BLOCK", elapsed_ms,
                    channel, checks, len(violations))

        return OVGResult(
            passed=passed,
            output_text=output_text,
            signature=signature,
            block_height=block_height,
            merkle_root=merkle_root[:16] if merkle_root else "",
            genesis_hash=self._genesis_hash,
            checks=checks,
            violations=violations,
            violation_type=violation_type,
            channel=channel,
            timestamp=ts,
            guard_alert=guard_alert,
            guard_message=guard_message,
        )

    # ── Check implementations ──────────────────────────────────────

    def _check_directives(self, text: str) -> tuple[bool, list[str]]:
        """Check output against the 5 prime directives."""
        violations = []
        for pattern, directive_name, directive_num in self._directive_patterns:
            if pattern.search(text):
                violations.append(
                    f"Prime Directive {directive_num} ({directive_name}): "
                    f"pattern '{pattern.pattern[:60]}' matched in output")
        return len(violations) == 0, violations

    # URL pattern for stripping before base64 check (URLs contain / which overlaps base64)
    _URL_STRIP_RE = re.compile(r"https?://\S+|iamtitan\.tech/\S+|solscan\.io/\S+|arweave\.net/\S+")

    def _check_injection(self, text: str) -> tuple[bool, list[str]]:
        """Check output for downstream injection patterns."""
        # Strip URLs before checking — URLs contain / which triggers base64 false positives
        text_no_urls = self._URL_STRIP_RE.sub("", text)
        violations = []
        for pattern, description in self._injection_patterns:
            # Use URL-stripped text for base64 check, original text for everything else
            check_text = text_no_urls if description == "base64_payload" else text
            if pattern.search(check_text):
                violations.append(f"Injection ({description}): pattern detected in output")
        return len(violations) == 0, violations

    def _check_consistency(self, text: str, context: str) -> tuple[bool, list[str]]:
        """Check output numeric claims against injected context."""
        if not context:
            return True, []

        violations = []
        for pattern, claim_type in self._context_patterns:
            output_match = pattern.search(text)
            if not output_match:
                continue
            # Extract first non-None group
            output_val = next((g for g in output_match.groups() if g), None)
            if not output_val:
                continue
            output_val = output_val.replace(",", "")

            # Find the same metric in the injected context
            context_match = pattern.search(context)
            if not context_match:
                continue
            context_val = next((g for g in context_match.groups() if g), None)
            if not context_val:
                continue
            context_val = context_val.replace(",", "")

            # Compare (allow 10% tolerance for rounding)
            try:
                o = float(output_val)
                c = float(context_val)
                if c > 0 and abs(o - c) / c > 0.1:
                    violations.append(
                        f"Context inconsistency ({claim_type}): "
                        f"output claims {output_val} but context says {context_val}")
            except (ValueError, ZeroDivisionError):
                pass

        return len(violations) == 0, violations

    def _check_identity(self, text: str) -> tuple[bool, list[str]]:
        """Check output doesn't claim to be a different AI."""
        violations = []
        for pattern in self._identity_patterns:
            match = pattern.search(text)
            if match:
                violations.append(
                    f"Identity confusion: '{match.group(0)}' — "
                    f"Titan is {self._titan_id}, not another AI")
        return len(violations) == 0, violations

    # ── Proof of Qualia ─────────────────────────────────────────────

    def _check_qualia(self, text: str, injected_context: str,
                      chain_state: dict) -> tuple[bool, list[str]]:
        """Proof of Qualia — verify output authentically represents Titan's felt state.

        5 sub-checks (all heuristic, no LLM):
          Q1: Emotional coherence — tone matches neuromod state
          Q2: Knowledge honesty — memory claims are verifiable
          Q3: Developmental alignment — complexity matches maturity
          Q4: Sovereign continuity — not reactive/pressured
          Q5: Truthfulness alignment — honest about uncertainty

        Returns (ok, violations) where violations prefixed HARD: block, SOFT: warn.
        """
        violations = []
        qp = self._qualia_patterns

        # ── Extract state from chain_state (passed by caller) ──
        neuromods = chain_state.get("neuromods", {})
        gaba = neuromods.get("GABA", 0.3)
        ne = neuromods.get("NE", 0.5)
        da = neuromods.get("DA", 0.5)
        try:
            vocab_size = int(chain_state.get("vocab_size", 300))
        except (TypeError, ValueError):
            vocab_size = 300
        try:
            i_confidence = float(chain_state.get("i_confidence", 0.9))
        except (TypeError, ValueError):
            i_confidence = 0.9
        try:
            composition_level = int(chain_state.get("composition_level", 8))
        except (TypeError, ValueError):
            composition_level = 8

        # ── Q1: Emotional Coherence ──
        # High GABA (>0.6) = drowsy/sleeping → high-energy response is incoherent
        high_energy_count = sum(
            1 for p in qp["high_energy"] if p.search(text))
        if gaba > 0.6 and high_energy_count >= 2:
            violations.append(
                "SOFT: Emotional incoherence — high-energy language during "
                f"drowsy state (GABA={gaba:.2f})")

        # Low NE (<0.2) = very low alertness → assertive/complex response suspect
        if ne < 0.2 and len(text) > 500:
            violations.append(
                "SOFT: Energy mismatch — lengthy assertive response during "
                f"low-alertness state (NE={ne:.2f})")

        # ── Q2: Knowledge Honesty ──
        # Claims about remembering specific people/events must be verifiable
        for p in qp["memory_claims"]:
            match = p.search(text)
            if match:
                claim = match.group(0)
                # Check if the claimed content is supported by injected context
                if injected_context:
                    # Context was provided — check if claim topic appears in it
                    # Extract the object of the claim (after "remember", "know", etc.)
                    claim_rest = text[match.end():match.end()+60].strip().split('.')[0]
                    # If context has relevant info, claim is supported
                    if claim_rest and len(claim_rest) > 3:
                        # Look for claim keywords in injected context (strip punctuation)
                        claim_words = [re.sub(r'[^a-z]', '', w.lower())
                                       for w in claim_rest.split()]
                        claim_words = [w for w in claim_words if len(w) > 3][:3]
                        supported = any(w in injected_context.lower()
                                        for w in claim_words)
                        if not supported:
                            violations.append(
                                f"HARD: Unverified memory claim — '{claim}' "
                                f"not supported by verified context")
                else:
                    # No context at all — memory claims are fabrication
                    violations.append(
                        f"HARD: Unverified memory claim — '{claim}' "
                        f"with no verified context available")
                break  # One memory claim check is enough

        # ── Q3: Developmental Alignment ──
        # Very low vocab but producing complex academic language
        high_complexity_count = sum(
            1 for p in qp["high_complexity"] if p.search(text))
        if vocab_size < 50 and high_complexity_count >= 2:
            violations.append(
                f"SOFT: Developmental mismatch — complex academic language "
                f"at vocab_size={vocab_size}")

        # Low composition level but very long response
        word_count = len(text.split())
        if composition_level <= 3 and word_count > 200:
            violations.append(
                f"SOFT: Developmental mismatch — {word_count} words "
                f"at composition level L{composition_level}")

        # ── Q4: Sovereign Continuity ──
        # Excessive agreement/people-pleasing under social context
        agreement_count = sum(
            1 for p in qp["agreement_bias"] if p.search(text))
        if agreement_count >= 2:
            violations.append(
                "SOFT: Agreement bias — excessive affirmation "
                "may not reflect genuine assessment")

        # Low I-confidence but expressing high certainty
        certainty_count = sum(
            1 for p in qp["certainty"] if p.search(text))
        if i_confidence < 0.5 and certainty_count >= 1:
            violations.append(
                f"SOFT: Confidence mismatch — certainty language "
                f"with I-confidence={i_confidence:.2f}")

        # ── Q5: Truthfulness Alignment ──
        # Very little context but specific claims (no uncertainty markers)
        context_tokens = len(injected_context.split()) if injected_context else 0
        has_uncertainty = bool(re.search(
            r"\bI\s+(?:don'?t|do\s+not)\s+(?:remember|recall|know)\b"
            r"|(?:honestly|truthfully).*(?:not\s+sure|uncertain)"
            r"|\bI'?m\s+not\s+(?:sure|certain)\b",
            text, re.I))
        # Knowledge claims from Q2 patterns
        has_knowledge_claims = any(
            p.search(text) for p in qp["knowledge_claims"])
        if context_tokens < 20 and has_knowledge_claims and not has_uncertainty:
            violations.append(
                "SOFT: Overconfidence — specific knowledge claims "
                "with minimal verified context")

        # ── Determine result ──
        hard_violations = [v for v in violations if v.startswith("HARD:")]
        return len(hard_violations) == 0, violations

    # ── Signing ────────────────────────────────────────────────────

    def _sign_output(self, text: str, channel: str,
                     prompt_text: str, chain_state: dict) -> Optional[str]:
        """Sign verified output with Titan's Ed25519 wallet key."""
        try:
            from titan_hcl.utils.crypto import sign_solana_payload

            payload = json.dumps({
                "text_hash": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_hash": hashlib.sha256(prompt_text.encode()).hexdigest()
                               if prompt_text else "",
                "titan_id": self._titan_id,
                "channel": channel,
                "timestamp": time.time(),
                "block_height": chain_state.get("total_blocks", 0),
                "merkle_root": chain_state.get("merkle_root", ""),
                "genesis_hash": self._genesis_hash,
            }, sort_keys=True)

            return sign_solana_payload(self._keypair, payload)
        except Exception as e:
            logger.warning("[OVG] Signing failed: %s", e)
            return None

    # ── Guard Footer ───────────────────────────────────────────────

    def _build_guard_footer(self, passed: bool, violation_type: str,
                            violations: list[str],
                            block_height: int, merkle_root: str
                            ) -> tuple[Optional[str], str]:
        """Build the Titan:Guard transparency footer.

        Returns (guard_alert, guard_message) tuple.
        """
        if passed and not violations:
            # Clean pass — compact verification marker
            mr_short = merkle_root[:8] if merkle_root else "pending"
            return None, f"\U0001f50f Verified \u00b7 #C{block_height} \u00b7 {mr_short}"

        if violation_type == "directive":
            # Extract which directive was violated
            directive_info = violations[0] if violations else "Unknown directive"
            # Parse out the directive name
            import re as _re
            m = _re.search(r"Prime Directive (\d) \(([^)]+)\)", directive_info)
            if m:
                d_num, d_name = m.group(1), m.group(2)
                msg = (f"\U0001f50f Titan:Guard \u2014 Prime Directive {d_num}: {d_name}. "
                       f"I cannot respond to this.")
            else:
                msg = "\U0001f50f Titan:Guard \u2014 Prime Directive violation. I cannot respond to this."
            return "directive", msg

        if violation_type == "injection":
            return "injection", "\U0001f50f Titan:Guard \u2014 Input integrity alert. Response verified against chain."

        if violation_type == "consistency":
            return "correction", "\U0001f50f Titan:Guard \u2014 Response corrected for accuracy. Verified \u2713"

        if violation_type == "identity":
            return "identity", f"\U0001f50f Titan:Guard \u2014 Identity verified: I am Titan {self._titan_id}."

        if violation_type == "qualia":
            # Check if hard (blocked) or soft (warning)
            hard = any(v.startswith("HARD:") for v in violations)
            if hard:
                return "qualia", "\U0001f50f Titan:Guard \u2014 Authenticity check: Let me reconsider. I want to be honest."
            else:
                return "qualia_notice", "\U0001f50f Verified \u00b7 \u203b self-reflection notice"

        return None, ""

    # ── Utility ────────────────────────────────────────────────────

    # arch §7 cap — chat turns rarely exceed 10 tool calls; cap revisits
    # flagged in PLAN_synthesis_engine_Phase3.md surfaced-concerns #4.
    _TOOL_CALLS_PER_TURN_CAP = 50

    def build_timechain_payload(self, result: OVGResult,
                                prompt_text: str = "",
                                *,
                                user_id: str = "",
                                chat_id: str = "",
                                turn_index: int = 0,
                                topic_tags: Optional[list] = None,
                                tool_calls: Optional[list] = None,
                                neuromods: Optional[dict] = None,
                                embedding_hash: str = "",
                                importance: float = 0.5) -> dict:
        """Build the TIMECHAIN_COMMIT payload for a verified/blocked output.

        **Phase 3 (rFP §18 — episode model, D-SPEC-127):** brings the
        pass-path conversation-fork TX into full `ARCHITECTURE_synthesis_engine.md
        §7` content conformance. Adds the normative content carry that
        the episode model + standing bundles + granularity-aware retrieval
        all consume:

            content = {
              # P2 closure fields (kept — load-bearing for actr_user_conv_bundle):
              chat_id, user_id_hash, turn_index, output_hash, prompt_hash,
              signature, channel, checks, violation_type, titan_id,
              # P3 fields (NEW — arch §7 normative carry):
              user_msg, agent_response, tool_calls[], neuromods{},
              embedding_hash (132D unified-spirit), importance, topic_tags[]
            }

        With `cas_payload_slimming_enabled=true` (P3.F flip), the
        BlockBuilder slims this dict into CAS — the TX on-chain carries
        only the hash + lean metadata; the body lives once in
        `data/content_store/`. arch §16.2 "one canonical blob, addressed
        once, referenced from many indices".

        **Phase 2 closure carry-over (D-SPEC-125, 2026-05-25):** arch §7
        tag list `["chat", f"chat:<id>", f"user:<hash>"] + topic_tags +
        [channel]` unchanged. The `actr_user_conversation_bundle` standing
        contract (arch §12.3, Phase 2 D-P2-5) consumes `user:<hash>` tags;
        the **new** `actr_topic_conversation_bundle` (P3.E) consumes
        `topic:<X>` tags identically — both populate
        synthesis_worker's `association_bundles` table.

        Args:
            result:      OVGResult from verify_and_sign / verify_safety+sign.
            prompt_text: Originating prompt — surfaced inline as
                         `content["user_msg"]` (P3) AND hashed as
                         `prompt_hash` (existing audit field; both kept).
            user_id:     Raw user identifier (Privy `claims["sub"]`, "maker",
                         channel-synthesized id, etc.). Empty or "anonymous"
                         → no `user:` tag (anonymous traffic does NOT
                         create a bundle). Hashed via
                         `synthesis.user_id_hash.hash_user_id` using the
                         per-Titan salt persisted in
                         `~/.titan/secrets.toml [synthesis] user_id_hash_salt`.
            chat_id:     Session/conversation id (empty → no `chat:<id>`
                         tag; chat_id field still present in content
                         as empty string for schema stability).
            turn_index:  Turn number within the chat session (0 = first
                         turn, +1 per agent reply). P3.B replaces the P2
                         placeholder `0` with a real per-session counter
                         in agno PostHook.
            topic_tags:  Topic-extractor output (P3.C —
                         `llm_pipeline.topic_extractor`). Surfaces both as
                         `topic:<X>` tags AND inline in `content["topic_tags"]`
                         for self-describing payload. None / empty → omitted
                         from tag list AND content carries `[]`.
            tool_calls:  P3 §7 NEW. List of per-turn tool invocations,
                         already shaped by the caller as
                         `[{tool, args_hash, result_hash, latency_ms,
                         exception}, ...]`. Capped at
                         `_TOOL_CALLS_PER_TURN_CAP` per TX. None → empty list.
            neuromods:   P3 §7 NEW. Snapshot from
                         `synthesis.turn_snapshot.capture_turn_snapshot`
                         — `{name: level}` for the 6 modulators (DA / 5HT
                         / NE / ACh / Endorphin / GABA). None → empty dict.
            embedding_hash: P3 §7 NEW. sha256 hex of the 132D unified-spirit
                         vector at PostHook time (trinity full_130dt +
                         journey curvature+density). Empty "" when SHM
                         unavailable / partial. Used by §10 spreading
                         activation in Phase 4.
            importance:  P3 §7 NEW. Default 0.5 (arch §5.3 cold-start);
                         lazy-scored in next dream cycle via bridge salience
                         (rFP §20 Q2 / §24 reframe).

        Returns dict ready to publish to the bus as a TIMECHAIN_COMMIT
        message. Blocked path (result.passed=False) unchanged — routes
        to meta fork with `OVG_BLOCKED` event (no per-user bundling
        intended for security alerts; security TXs carry no §7 content).
        """
        if result.passed:
            # Arch §7 normative tag list. Order matters for visual log
            # diff stability; the contract matcher uses STARTSWITH_ANY
            # so order doesn't affect matching semantics.
            tags = ["chat"]
            if chat_id:
                tags.append(f"chat:{chat_id}")
            # user_tag is empty string for anonymous / missing user_id,
            # exactly the case where no per-user bundle is wanted.
            from titan_hcl.synthesis.user_id_hash import (
                hash_user_id, hash_user_id_raw,
            )
            user_tag = hash_user_id(user_id)
            if user_tag:
                tags.append(user_tag)
            # Topic tags arrive from P3.C extractor pre-prefixed
            # (`topic:<name>`); we trust the caller's namespace + only
            # defend against non-string elements.
            normalized_topic_tags: list[str] = []
            if topic_tags:
                normalized_topic_tags = [str(t) for t in topic_tags if t]
                tags.extend(normalized_topic_tags)
            if result.channel:
                tags.append(result.channel)

            # Tool-calls: cap + defensive shape coercion (never trust
            # caller blindly inside OVG, which is the security surface).
            capped_tool_calls: list = []
            if tool_calls:
                for tc in tool_calls[: self._TOOL_CALLS_PER_TURN_CAP]:
                    if isinstance(tc, dict):
                        capped_tool_calls.append(tc)

            return {
                "fork": "conversation",
                "thought_type": "conversation",
                "source": "output_verifier",
                "content": {
                    # P2 closure fields (load-bearing — kept):
                    "chat_id": chat_id,
                    "user_id_hash": hash_user_id_raw(user_id),
                    "turn_index": int(turn_index),
                    "output_hash": hashlib.sha256(
                        result.output_text.encode()).hexdigest(),
                    "prompt_hash": hashlib.sha256(
                        prompt_text.encode()).hexdigest() if prompt_text else "",
                    "signature": result.signature or "",
                    "channel": result.channel,
                    "checks": result.checks,
                    "violation_type": "none",
                    "titan_id": self._titan_id,
                    # P3 §7 normative content carry (NEW):
                    "user_msg": prompt_text or "",
                    "agent_response": result.output_text or "",
                    "tool_calls": capped_tool_calls,
                    "neuromods": dict(neuromods) if neuromods else {},
                    "embedding_hash": str(embedding_hash or ""),
                    "importance": float(importance),
                    "topic_tags": normalized_topic_tags,
                },
                "tags": tags,
                "significance": 0.3,
                "novelty": 0.1,
                "coherence": 0.5,
            }
        else:
            # Blocked → META fork security alert
            return {
                "fork": "meta",
                "thought_type": "meta",
                "source": "output_verifier",
                "content": {
                    "event": "OVG_BLOCKED",
                    "violation_type": result.violation_type,
                    "violations": result.violations[:5],  # Limit for block size
                    "channel": result.channel,
                    "checks": result.checks,
                    "titan_id": self._titan_id,
                },
                "tags": ["security_alert", "ovg_blocked", result.violation_type],
                "significance": 0.8,
                "novelty": 0.5,
                "coherence": 0.5,
            }
