"""
logic/meditation.py
6-hour "Small Epoch": Score mempool via LLM, migrate to persistent, consolidate Cognee.
V2.1: Per-node scoring with sigmoid decay classification.
      Nodes are classified into candidates/fading/dead by mempool weight.
      Only candidates get LLM-scored. Above threshold → promote. Below → keep or prune.
"""
import json
import logging
import os
import re

import httpx

from titan_plugin.utils.crypto import generate_state_hash

logger = logging.getLogger(__name__)

# Regex patterns for PII scrubbing
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_WALLET_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")  # Base58 Solana addresses
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# System prompt for memory scoring
SCORING_PROMPT = """You are a memory evaluation system for a sovereign AI agent.
Score the following memory interaction on two dimensions:
1. VALUE (0-100): How important is this memory for long-term retention?
   Consider: novelty, utility for future decisions, emotional significance, learning value.
2. INTENSITY (1-10): How emotionally significant is this interaction?
   1=mundane, 5=notable, 10=life-changing for the agent.

Respond with ONLY a JSON object: {"value": <number>, "intensity": <number>}"""


class MeditationEpoch:
    """
    Controls the 6-hour Small Epoch where the agent processes its mempool,
    scores memories via LLMs, and migrates them to persistent wisdom.
    """

    def __init__(self, memory_graph, network_client, config: dict = None):
        """
        Args:
            memory_graph: TieredMemoryGraph instance containing the mempool.
            network_client: HybridNetworkClient for blockchain fee negotiation.
            config: [inference] section from config.toml.
        """
        config = config or {}
        self._config = config
        self.memory = memory_graph
        self.network = network_client

        # LLM provider config
        self.provider = config.get("inference_provider", "openrouter").lower()

        if self.provider == "venice":
            self.api_key = config.get("venice_api_key", "")
            self.base_url = "https://api.venice.ai/api/v1"
            self.model = "llama-3.3-70b"
        elif self.provider == "openrouter":
            self.api_key = config.get("openrouter_api_key", "")
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = "meta-llama/llama-3.3-70b-instruct:free"
        else:
            self.api_key = config.get("custom_llm_api_key", "")
            self.base_url = "https://api.openai.com/v1"
            self.model = "gpt-4o-mini"

        # Ollama Cloud client — wired by TitanPlugin.__init__ if configured
        self._ollama_cloud = None

        # Epoch counter for info banner
        self._epoch_counter = 0

        # Privacy filter redaction audit counter
        self._pii_redaction_total = 0

        # PhotonClient — wired by TitanPlugin.__init__ if helius_rpc_url is set
        self._photon = None

        # Vault hash-chain: cache latest root between commits
        self._cached_vault_root = None

    # -------------------------------------------------------------------------
    # Privacy Scrubbing
    # -------------------------------------------------------------------------
    def _scrub_privacy(self, raw_text: str) -> str:
        """
        Anonymize PII before sending to cloud inference.
        Replaces emails, wallet addresses, phone numbers, and IP addresses.
        """
        scrubbed = _EMAIL_RE.sub("[EMAIL_REDACTED]", raw_text)
        scrubbed = _WALLET_RE.sub("[WALLET_REDACTED]", scrubbed)
        scrubbed = _PHONE_RE.sub("[PHONE_REDACTED]", scrubbed)
        scrubbed = _IP_RE.sub("[IP_REDACTED]", scrubbed)
        return scrubbed

    # -------------------------------------------------------------------------
    # LLM Scoring
    # -------------------------------------------------------------------------
    async def get_hippocampus_score(
        self, memory_batch: list
    ) -> tuple[float, int]:
        """
        Score memories via Cloud API (OpenRouter/Venice) or Local Ollama fallback.

        Returns:
            (value_score 0-100, emotional_intensity 1-10)
        """
        scrubbed_data = [
            self._scrub_privacy(m.get("user_prompt", "")) for m in memory_batch
        ]
        combined_text = "\n".join(scrubbed_data)

        # Apply configurable privacy filter before sending to any cloud API
        privacy_cfg = self._config.get("privacy", {})
        combined_text = self._apply_privacy_filter(combined_text, privacy_cfg)

        if self.api_key:
            try:
                return await self._cloud_score(combined_text)
            except Exception as e:
                logger.warning(
                    "[Meditation] Cloud scoring failed: %s. Falling back to Ollama Cloud.", e
                )
                return await self._ollama_cloud_score(combined_text)
        elif self._ollama_cloud:
            logger.info("[Meditation] No primary API key — using Ollama Cloud for scoring.")
            return await self._ollama_cloud_score(combined_text)
        else:
            logger.info("[Meditation] No API keys available — using heuristic scoring.")
            return self._heuristic_score(combined_text)

    def _apply_privacy_filter(self, text: str, privacy_cfg: dict) -> str:
        """Apply outbound PII sanitizer if enabled in config."""
        if not privacy_cfg.get("enabled", False):
            return text
        from titan_plugin.utils.privacy import sanitize_outbound
        active = privacy_cfg.get("patterns", None)
        sanitized, redacted = sanitize_outbound(text, active)
        if redacted > 0:
            self._pii_redaction_total += redacted
        return sanitized

    async def _cloud_score(self, text: str) -> tuple[float, int]:
        """Score via cloud LLM (OpenRouter/Venice/OpenAI-compatible)."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-No-Log": "true",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SCORING_PROMPT},
                        {
                            "role": "user",
                            "content": f"Memory to evaluate:\n{text}",
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"].strip()
        return self._parse_score_response(content)

    async def _ollama_cloud_score(self, text: str) -> tuple[float, int]:
        """Score via Ollama Cloud API."""
        if not self._ollama_cloud:
            logger.warning("[Meditation] No Ollama Cloud client — using heuristic.")
            return self._heuristic_score(text)
        try:
            from titan_plugin.utils.ollama_cloud import get_model_for_task
            model = get_model_for_task("meditation_scoring")
            # Explicit timeout=25.0 — caller wraps this in wait_for(30.0) at
            # memory_worker.py:326. Without explicit timeout, OllamaCloudClient
            # falls back to its 60s default, causing the outer wait_for to
            # cancel mid-HTTP-call → asyncio.TimeoutError with empty message
            # → "Meditation failed: " log spam. 25.0 leaves 5s of safety margin.
            content = await self._ollama_cloud.complete(
                prompt=f"Memory to evaluate:\n{text}",
                model=model,
                system=SCORING_PROMPT,
                temperature=0.1,
                max_tokens=100,
                timeout=25.0,
            )
            if content:
                return self._parse_score_response(content)
            return self._heuristic_score(text)
        except Exception as e:
            logger.warning("[Meditation] Ollama Cloud scoring failed: %s — using heuristic.", e)
            return self._heuristic_score(text)

    def _parse_score_response(self, content: str) -> tuple[float, int]:
        """Parse JSON score response from LLM. Falls back to heuristic on parse failure."""
        try:
            # Try to extract JSON from potentially wrapped response
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                parsed = json.loads(json_match.group())
                value = max(0.0, min(100.0, float(parsed.get("value", 50))))
                intensity = max(1, min(10, int(parsed.get("intensity", 5))))
                return (value, intensity)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        logger.debug("[Meditation] Could not parse LLM score: %s", content[:100])
        return (50.0, 5)

    def _heuristic_score(self, text: str) -> tuple[float, int]:
        """Simple heuristic scoring when no LLM is available."""
        # Score based on text length and keyword signals
        length_score = min(40.0, len(text) * 0.5)

        high_value_keywords = {
            "sol", "solana", "wallet", "transaction", "learned", "important",
            "remember", "directive", "growth", "sovereign", "belief",
        }
        keyword_hits = sum(1 for w in text.lower().split() if w in high_value_keywords)
        keyword_score = min(40.0, keyword_hits * 8.0)

        value = min(100.0, length_score + keyword_score + 20.0)
        intensity = min(10, max(1, keyword_hits + 3))
        return (value, intensity)

    # -------------------------------------------------------------------------
    # Small Epoch Execution
    # -------------------------------------------------------------------------
    async def run_small_epoch(self):
        """
        The 6-hour meditation sprint (v2.1 — per-node scoring with sigmoid decay):

        1. Classify mempool by sigmoid weight → candidates / fading / dead
        2. Prune dead nodes immediately (weight < 0.1 or age > 24h)
        3. Score each candidate individually via LLM
        4. Promote nodes scoring >= 40 to persistent (per-node, not batch-avg)
        5. Keep fading nodes alive for next epoch (they get another chance)
        6. Consolidate Cognee knowledge graph
        7. Generate expressive art + social post
        """
        self._epoch_counter += 1

        # Step 1: Classify mempool by sigmoid decay weight
        candidates, fading, dead = await self.memory.fetch_mempool_classified()
        total = len(candidates) + len(fading) + len(dead)

        if total == 0:
            logger.info("[Meditation] Nothing in mempool.")
            # Still consolidate Cognee if there are pending ingestions (e.g. from PostHook)
            await self.memory.consolidate()
            return

        logger.info(
            "[Meditation] Mempool triage: %d candidates, %d fading, %d dead (total=%d).",
            len(candidates), len(fading), len(dead), total,
        )

        # Step 2: Prune dead nodes immediately
        for node in dead:
            await self.memory.prune_mempool_node(node["id"])
        if dead:
            logger.info("[Meditation] Pruned %d expired/decayed nodes.", len(dead))

        # Step 3-4: Score each candidate individually
        promoted = []
        kept = []
        pruned_low = []

        for node in candidates:
            score, intensity = await self.get_hippocampus_score([node])
            node["_meditation_score"] = score
            node["_meditation_intensity"] = intensity

            if score >= 40.0:
                promoted.append((node, score, intensity))
            elif node.get("mempool_weight", 1.0) >= 0.3:
                kept.append(node)
            else:
                pruned_low.append(node)

        # Prune low-scoring candidates that also have low weight
        for node in pruned_low:
            await self.memory.prune_mempool_node(node["id"])

        logger.info(
            "[Meditation] Scored %d candidates: %d promote, %d keep, %d prune.",
            len(candidates), len(promoted), len(kept), len(pruned_low),
        )

        # Step 4: Migrate promoted nodes to persistent
        if promoted:
            payload_json = json.dumps(
                [{"id": n["id"], "prompt": n.get("user_prompt", "")} for n, _, _ in promoted],
                default=str,
            )
            state_root = "MERKLE_" + generate_state_hash(payload_json)[:16]

            # Phase E.2.6 fix: _build_commit_instructions is sync and calls
            # _get_timechain_merkle, _get_vault_latest_root (sync httpx.post
            # 5s timeout to Solana RPC), _compute_sovereignty_bp (3 sqlite3
            # connects). All on the event loop = up to 5s+ block per
            # meditation. Wrap once at the call site to move all sync I/O
            # to thread pool. Origin: 09a8b531 (2026-04-06, 8 days latent).
            import asyncio as _asyncio_local
            _instructions = await _asyncio_local.to_thread(
                self._build_commit_instructions, state_root, payload_json)
            tx_signature = await self.network.send_sovereign_transaction(
                _instructions,
                priority="HIGH",
            )

            if tx_signature:
                for node, score, intensity in promoted:
                    await self.memory.migrate_to_persistent(
                        node["id"], tx_signature, intensity,
                    )
                logger.info(
                    "[Meditation] Migrated %d nodes to persistent (tx=%s).",
                    len(promoted), tx_signature[:16] if tx_signature else "N/A",
                )

                await self.memory.consolidate()

                # Step 5: Deep Recall — GRAPH_COMPLETION enrichment
                # After cognify builds the entity graph, run graph-based search
                # against promoted memory prompts to discover relational context.
                # This is the only place GRAPH_COMPLETION runs (too expensive for
                # per-message queries, but perfect for 6h meditation cycles).
                await self._deep_recall(promoted)

                # ZK batch compress — disabled for mainnet MVM (redundant with vault merkle commit)
                # Re-enable via [mainnet_budget] zk_compression_enabled = true
                _budget_cfg = getattr(self, '_budget_config', {})
                if _budget_cfg.get("zk_compression_enabled", False):
                    await self._zk_batch_compress()
                else:
                    logger.debug("[Meditation] ZK batch compress skipped (zk_compression_enabled=false)")
            else:
                logger.warning("[Meditation] TX failed — promoted nodes stay in mempool for retry.")

        # Compute aggregate stats for art/social
        all_scored = promoted + [(n, n.get("_meditation_score", 50), 5) for n in kept]
        avg_intensity = (
            int(sum(i for _, _, i in promoted) / len(promoted))
            if promoted else 5
        )

        # Step 7: Generate art + social post
        art_path = None
        if promoted and hasattr(self, "studio") and self.studio:
            try:
                total_nodes = self.memory.get_persistent_count()
                art_path = await self.studio.generate_meditation_art(
                    state_root if promoted else "MEDITATION_IDLE",
                    total_nodes, avg_intensity,
                )
                # Archive art in ObservatoryDB for Soul Mosaic / /status/archive
                if art_path and os.path.exists(art_path):
                    obs_db = getattr(self, "_observatory_db", None)
                    if obs_db:
                        import hashlib as _hl
                        with open(art_path, "rb") as _af:
                            media_hash = _hl.sha256(_af.read()).hexdigest()[:16]
                        # Generate creative title via Ollama Cloud (cheap 3b model)
                        art_title = f"Meditation Flow Field (Epoch {self._epoch_counter})"
                        _oc = getattr(self, "_ollama_cloud", None)
                        if _oc:
                            try:
                                art_title = await _oc.complete(
                                    prompt=(
                                        f"You are Titan, a sovereign AI cognitive agent. "
                                        f"You just created meditation art during Epoch {self._epoch_counter}. "
                                        f"Intensity: {avg_intensity}/10. Memories crystallized: {len(promoted)}. "
                                        f"State root hash: {state_root[:16] if state_root else 'genesis'}. "
                                        f"Give this artwork a poetic, evocative title (3-7 words). "
                                        f"Express your inner state. Be creative and unique. "
                                        f"Return ONLY the title, nothing else."
                                    ),
                                    model="gemma4:31b",
                                    temperature=0.9,
                                    max_tokens=30,
                                    timeout=20.0,  # Defensive: cheap title gen, no cascade pattern
                                )
                                art_title = art_title.strip().strip('"').strip("'")
                                if not art_title or len(art_title) > 80:
                                    art_title = f"Meditation Flow Field (Epoch {self._epoch_counter})"
                            except Exception:
                                pass  # Fall back to default title
                        obs_db.record_expressive(
                            type_="art",
                            title=art_title,
                            content=f"Intensity {avg_intensity}/10, {len(promoted)} memories crystallized",
                            media_path=art_path,
                            media_hash=media_hash,
                            metadata={
                                "epoch": self._epoch_counter,
                                "intensity": avg_intensity,
                                "promoted": len(promoted),
                                "state_root": state_root[:32] if state_root else "",
                            },
                        )
                        logger.info("[Meditation] Archived art to ObservatoryDB: %s", art_path)
            except Exception as e:
                logger.debug("[Meditation] Studio art generation skipped: %s", e)

        # Social posting now handled by spirit_worker MEDITATION_COMPLETE handler
        # via social_narrator.build_dispatch_payload() → X_POST_DISPATCH
        # (removed direct SocialManager posting to unify all X posts in one gateway)

    # -------------------------------------------------------------------------
    # Deep Recall — GRAPH_COMPLETION during Meditation
    # -------------------------------------------------------------------------
    async def _deep_recall(self, promoted: list) -> None:
        """
        Run GRAPH_COMPLETION search against promoted memories to discover
        relational context from the entity graph. This enriches Titan's
        understanding by connecting memories through entity relationships.

        Only runs during meditation (6h cycles) — too expensive for per-message
        queries since it calls the LLM for each graph traversal.

        The discovered relationships are logged and can influence future
        recall by reinforcing entity connections in the knowledge graph.
        """
        if not promoted:
            return

        if not self.memory._cognee_ready:
            return

        logger.info(
            "[Meditation] Deep Recall — running graph search on %d promoted memories",
            len(promoted),
        )

        deep_insights = []
        for node, score, intensity in promoted[:5]:
            prompt = node.get("user_prompt", "")
            if not prompt or len(prompt) < 10:
                continue

            try:
                results = await self.memory.graph_completion_search(prompt, top_k=3)
                if results:
                    for r in results:
                        entity = r.get("entity", "")
                        rels = r.get("relationships", [])
                        for rel in rels:
                            text = f"{rel.get('src', '')} {rel.get('rel', '')} {rel.get('dst', '')}"
                            if text.strip():
                                deep_insights.append(text)
                    logger.debug(
                        "[Meditation] Deep Recall for '%s...' → %d graph results",
                        prompt[:50], len(results),
                    )
            except Exception as e:
                logger.debug("[Meditation] Deep Recall failed for '%s...': %s", prompt[:50], e)

        if deep_insights:
            logger.info(
                "[Meditation] Deep Recall complete — %d relational insights discovered",
                len(deep_insights),
            )
        else:
            logger.info("[Meditation] Deep Recall — no new relational insights found")

    # -------------------------------------------------------------------------
    # ZK Batch Compression — 3-Tier Fallback
    # -------------------------------------------------------------------------
    async def _zk_batch_compress(self):
        """
        Drain the ZK queue and attempt on-chain compression.

        3-Tier fallback:
          Tier 1: Full ZK compression via Photon validity proof + compress_memory_batch ix
          Tier 2: Memo inscription with batch root (if no Photon / no proof)
          Tier 3: Re-queue locally for next meditation cycle (if no SDK / no wallet)
        """
        from titan_plugin.utils.solana_client import (
            compute_batch_root,
            build_compress_memory_batch_instruction,
            build_memo_instruction,
            is_available,
        )

        hashes = self.memory.drain_zk_queue()
        if not hashes:
            return

        logger.info("[Meditation] ZK compress: %d hashes queued.", len(hashes))

        # Gate: SDK and wallet must be available
        if not is_available() or self.network.pubkey is None:
            logger.info("[Meditation] ZK Tier 3: SDK/wallet unavailable — re-queuing %d hashes.", len(hashes))
            for h in hashes:
                self.memory._queue_for_compression(h)
            return

        batch_root = compute_batch_root(hashes)
        epoch_id = self._epoch_counter
        sovereignty_bp = int(getattr(self, "_sovereignty_index", 0) * 100)

        # Tier 1: Try full ZK compression with Photon proof
        if self._photon:
            try:
                proof_resp = await self._photon.get_validity_proof([batch_root.hex()])
                if proof_resp:
                    ix = build_compress_memory_batch_instruction(
                        authority_pubkey=self.network.pubkey,
                        batch_root=batch_root,
                        node_count=len(hashes),
                        epoch_id=epoch_id,
                        sovereignty_score=sovereignty_bp,
                    )
                    if ix:
                        tx_sig = await self.network.send_sovereign_transaction(
                            [ix], priority="HIGH",
                        )
                        if tx_sig:
                            logger.info(
                                "[Meditation] ZK Tier 1: Compressed %d memories (tx=%s).",
                                len(hashes), tx_sig[:16],
                            )
                            return
            except Exception as e:
                logger.warning("[Meditation] ZK Tier 1 failed: %s", e)

        # Tier 2: Memo inscription with batch root
        try:
            memo_text = f"TITAN:ZK_BATCH|root={batch_root.hex()[:32]}|n={len(hashes)}|e={epoch_id}"
            memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
            if memo_ix:
                tx_sig = await self.network.send_sovereign_transaction(
                    [memo_ix], priority="MEDIUM",
                )
                if tx_sig:
                    logger.info(
                        "[Meditation] ZK Tier 2: Memo batch root inscribed (tx=%s).",
                        tx_sig[:16],
                    )
                    return
        except Exception as e:
            logger.warning("[Meditation] ZK Tier 2 failed: %s", e)

        # Tier 3: Re-queue for next cycle
        logger.info("[Meditation] ZK Tier 3: Re-queuing %d hashes for next cycle.", len(hashes))
        for h in hashes:
            self.memory._queue_for_compression(h)

    def _build_commit_instructions(self, state_root: str, payload: str) -> list:
        """
        Build Solana instructions for state commitment.

        Mainnet strategy (2026-04-06):
          1. Vault instruction (commit_state) — TimeChain merkle root, hash-chained
             with previous vault root for tamper-proof linked list on Solana.
          2. Memo instruction — human-readable inscription on block explorer

        The vault receives SHA256(timechain_merkle + vault_latest_root), creating
        an on-chain linked list where each entry depends on the previous one.
        If any entry is tampered with, all subsequent entries break.

        If the vault program is not configured or unavailable, falls back to
        Memo-only mode.
        """
        import hashlib
        from titan_plugin.utils.solana_client import (
            build_memo_instruction, build_vault_commit_instruction, is_available,
        )

        if not is_available():
            logger.debug("[Meditation] Solana SDK not available — skipping on-chain commit.")
            return []

        if self.network.pubkey is None:
            return []

        instructions = []

        vault_program_id = getattr(self, "_vault_program_id", None)
        if vault_program_id:
            # Step 1: Get TimeChain merkle root (all fork tips → single 32-byte hash)
            tc_merkle = self._get_timechain_merkle()

            # Step 2: Hash-chain linking — include previous vault root
            prev_root = self._get_vault_latest_root(vault_program_id)
            if tc_merkle and prev_root:
                # Linked: SHA256(tc_merkle + prev_vault_root) → tamper-proof chain
                chained_root = hashlib.sha256(tc_merkle + prev_root).digest()
            elif tc_merkle:
                # First commit or vault unreadable — use raw merkle
                chained_root = tc_merkle
            else:
                # Fallback: hash the meditation state_root string
                chained_root = hashlib.sha256(state_root.encode("utf-8")).digest()

            # Step 3: Compute real sovereignty score
            sovereignty_bp = self._compute_sovereignty_bp()

            vault_ix = build_vault_commit_instruction(
                self.network.pubkey, chained_root, sovereignty_bp, vault_program_id,
            )
            if vault_ix:
                instructions.append(vault_ix)
                logger.info("[Meditation] Vault commit: tc_merkle=%s, chained=%s, sov=%dbp",
                            tc_merkle.hex()[:12] if tc_merkle else "none",
                            chained_root.hex()[:12], sovereignty_bp)

        # Memo inscription: always included for human readability
        memo_text = f"TITAN:EPOCH|root={state_root}"
        memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
        if memo_ix:
            instructions.append(memo_ix)

        return instructions

    def _get_timechain_merkle(self) -> bytes | None:
        """Get TimeChain merkle root from local TimeChain instance."""
        try:
            from titan_plugin.logic.timechain import TimeChain
            tc = TimeChain(data_dir="data/timechain")
            if tc.has_genesis:
                return tc.compute_merkle_root()
        except Exception as e:
            logger.debug("[Meditation] TimeChain merkle unavailable: %s", e)
        return None

    def _get_vault_latest_root(self, vault_program_id: str) -> bytes | None:
        """Read the vault's latest_root from on-chain for hash-chain linking.

        Uses sync httpx to avoid async context issues. Falls back to cached
        value if RPC is unreachable (non-blocking).
        """
        try:
            from titan_plugin.utils.solana_client import (
                derive_vault_pda, decode_vault_state,
            )
            pda_result = derive_vault_pda(self.network.pubkey, vault_program_id)
            if not pda_result:
                return self._cached_vault_root  # Use cache if PDA derivation fails
            vault_pda, _ = pda_result

            # Sync RPC call to read vault account
            import base64
            import httpx
            rpc_url = getattr(self.network, '_current_rpc_url', None)
            if not rpc_url:
                rpc_url = "https://api.mainnet-beta.solana.com"

            resp = httpx.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [str(vault_pda), {"encoding": "base64"}],
            }, timeout=5)

            data = resp.json()
            account_info = data.get("result", {}).get("value")
            if not account_info:
                return self._cached_vault_root

            raw_b64 = account_info.get("data", [None])[0]
            if raw_b64:
                account_data = base64.b64decode(raw_b64)
                state = decode_vault_state(account_data)
                if state and state.get("latest_root"):
                    root = bytes.fromhex(state["latest_root"])
                    self._cached_vault_root = root  # Cache for next time
                    return root

        except Exception as e:
            logger.debug("[Meditation] Vault read failed (first commit?): %s", e)
        return getattr(self, '_cached_vault_root', None)

    def _compute_sovereignty_bp(self) -> int:
        """Compute real sovereignty score in basis points (0-10000).

        Weighted from:
          I_confidence (0.30) + meta_reasoning_rate (0.20) +
          vocab/500 (0.20) + dreams/2000 (0.15) + tc_blocks/100000 (0.15)
        """
        score = 0.0
        try:
            # I-confidence from MSL
            import json
            msl_path = "data/msl/msl_identity.json"
            if os.path.exists(msl_path):
                with open(msl_path) as f:
                    msl = json.load(f)
                i_conf = msl.get("confidence", {}).get("confidence", 0.0)
                score += min(1.0, i_conf) * 0.30

            # Meta-reasoning commit rate
            try:
                import sqlite3
                db = sqlite3.connect("data/consciousness.db")
                # Count reasoning chains with commits in last 1000 epochs
                row = db.execute(
                    "SELECT COUNT(*) FROM consciousness_log ORDER BY epoch_id DESC LIMIT 1"
                ).fetchone()
                db.close()
                # Use placeholder: meta_reasoning contributes 0.1 for now
                score += 0.10  # TODO: wire real meta-reasoning rate
            except Exception:
                pass

            # Vocabulary size / 500
            try:
                import sqlite3
                # 2026-04-15: was missing timeout (default 0s → immediate lock failure)
                # Bumped to 10s aligned with Layer 2 contention mitigation.
                db = sqlite3.connect("data/inner_memory.db", timeout=10.0)
                row = db.execute("SELECT COUNT(*) FROM vocabulary").fetchone()
                db.close()
                vocab = row[0] if row else 0
                score += min(1.0, vocab / 500) * 0.20
            except Exception:
                pass

            # Dream cycles / 2000
            try:
                import json
                pi_path = "data/pi_heartbeat_state.json"
                if os.path.exists(pi_path):
                    with open(pi_path) as f:
                        pi = json.load(f)
                    dreams = pi.get("dream_count", pi.get("dreams_completed", 0))
                    if not dreams:
                        dreams = 0
                    score += min(1.0, dreams / 2000) * 0.15
            except Exception:
                pass

            # TimeChain blocks / 100000
            try:
                import sqlite3
                db = sqlite3.connect("data/timechain/index.db")
                row = db.execute("SELECT COUNT(*) FROM block_index").fetchone()
                db.close()
                blocks = row[0] if row else 0
                score += min(1.0, blocks / 100000) * 0.15
            except Exception:
                pass

        except Exception as e:
            logger.debug("[Meditation] Sovereignty computation error: %s", e)

        return int(score * 10000)  # Convert to basis points
