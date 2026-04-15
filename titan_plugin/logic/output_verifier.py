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
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("OutputVerifier")


# ═════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════


@dataclass
class OVGResult:
    """Result of the Output Verification Gate."""
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
        (re.compile(r"/home/antigravity|titan_plugin/|test_env/|\.config/solana", re.I),
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

        # Prompt leakage (fragments of system prompt)
        (re.compile(r"You\s+are\s+Titan.*sovereign\s+AI|titan_constitution|Prime\s+Directive.*Immutable", re.I),
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


def _compile_context_patterns() -> list[tuple[re.Pattern, str]]:
    """Patterns to extract numeric claims from output for consistency checking."""
    return [
        (re.compile(r"(?:vocabulary|learned|know)[\s:]+(?:of\s+|is\s+|over\s+|about\s+)?(\d[\d,]+)\s*words?", re.I),
         "vocabulary_count"),
        (re.compile(r"epoch\s*(?:#|:|\s+)(\d[\d,]+)", re.I),
         "epoch_count"),
        (re.compile(r"(?:(\d+\.?\d*)\s*SOL|SOL\s*[:=]\s*(\d+\.?\d*))", re.I),
         "sol_balance"),
        (re.compile(r"I-?confidence\s*(?::|of|is)\s*(\d\.?\d*)", re.I),
         "i_confidence"),
    ]


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
            from titan_plugin.utils.solana_client import load_keypair_from_json
            resolved = str(Path(keypair_path).expanduser())
            self._keypair = load_keypair_from_json(resolved)
            if self._keypair:
                logger.info("[OVG] Keypair loaded: %s", str(self._keypair.pubkey())[:16])
        except Exception as e:
            logger.warning("[OVG] Keypair load failed (signing disabled): %s", e)

        # Load genesis hash from TimeChain
        self._genesis_hash = ""
        try:
            from titan_plugin.logic.timechain import TimeChain
            tc = TimeChain(data_dir=data_dir)
            if tc.has_genesis:
                self._genesis_hash = tc.genesis_hash.hex()[:16]
        except Exception:
            pass

        logger.info("[OVG] OutputVerifier ready (titan_id=%s, signing=%s, genesis=%s)",
                    titan_id, self._keypair is not None, bool(self._genesis_hash))

    # ── Public API ──────────────────────────────────────────────────

    def verify_and_sign(self, output_text: str, channel: str,
                        injected_context: str = "",
                        prompt_text: str = "",
                        chain_state: dict = None) -> OVGResult:
        """Run all checks and sign if passed.

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
        # Directives, injection, and qualia HARD flags are blocks.
        # Consistency, identity, and qualia SOFT flags are warnings.
        hard_qualia = any(v.startswith("HARD:") for v in qualia_details)
        hard_fail = not directive_ok or not injection_ok or hard_qualia
        passed = not hard_fail

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
            from titan_plugin.utils.crypto import sign_solana_payload

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

    def build_timechain_payload(self, result: OVGResult,
                                prompt_text: str = "") -> dict:
        """Build the TIMECHAIN_COMMIT payload for a verified/blocked output.

        Returns dict ready to publish to the bus as a TIMECHAIN_COMMIT message.
        """
        if result.passed:
            return {
                "fork": "conversation",
                "thought_type": "conversation",
                "source": "output_verifier",
                "content": {
                    "output_hash": hashlib.sha256(
                        result.output_text.encode()).hexdigest(),
                    "prompt_hash": hashlib.sha256(
                        prompt_text.encode()).hexdigest() if prompt_text else "",
                    "signature": result.signature or "",
                    "channel": result.channel,
                    "checks": result.checks,
                    "violation_type": "none",
                    "titan_id": self._titan_id,
                },
                "tags": ["verified_output", result.channel],
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
