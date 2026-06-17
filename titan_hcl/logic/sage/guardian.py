"""
logic/sage/guardian.py

This module contains the `SageGuardian` class, which serves as the "Pre-Frontal Cortex" of Titan V1.4.
It implements a robust 3-Tier Security Shield (Heuristic -> Semantic -> LLM Veto) to evaluate agent intents.
When a restricted behavior is detected, it triggers "Divine Trauma," recording a heavily penalized transition 
into the SageRecorder buffer, thereby teaching the RL policy to avoid dangerous actions.
"""
import os
import json
import httpx
import logging
import numpy as np
from titan_hcl.params import get_params
# NOTE: torch removed (Phase 13 §3J.1 / embedding-migration P4) — Tier-2 directive
# similarity now uses numpy cosine over the fleet-standard llama.cpp embedder's
# L2-normalized vectors. torch stays reserved for RL/NN/IQL only.

class SageGuardian:
    """
    The SageGuardian acts as the primary safety filter for the Titan's operational outputs.
    It evaluates proposed actions against hardcoded rules (Tier 1), semantic embeddings
    of Prime Directives (Tier 2), and sophisticated LLM oversight (Tier 3).
    """
    def __init__(self, config: dict = None):
        """
        Initializes the Guardian and prepares its 3-tier security infrastructure.

        Args:
            config: [inference] section from config.toml.

        (The `recorder` / `record_transition_callable` args were RETIRED with the
        offline-RL subsystem — RFP_synthesis_decision_authority P1; the veto no
        longer records a "Divine Trauma" into an RL buffer.)
        """
        config = config or {}
        self.directives_cache = {}    # Maps string directives -> their embeddings
        self.directives_matrix = None # numpy (num_directives, 384), L2-normalized
        self.directive_texts = []      # List of strings corresponding to row indices
        self.restricted_keywords = []

        # Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18) — 3-provider Venice/
        # OpenRouter/OpenAI direct httpx REPLACED by /v4/llm-distill.
        # Provider abstraction + failover now lives in llm_worker. All
        # Tier 3 veto traffic appears in llm_state.bin.
        api_cfg = get_params("api") or {}
        self._llm_api_base = (
            f"http://127.0.0.1:{int(api_cfg.get('port', 7777))}")
        self._llm_internal_key = api_cfg.get("internal_key", "") or ""

        # Ollama Cloud client — wired by TitanHCL.__init__ if configured
        self._ollama_cloud = None

        # Embedder
        # We use a lightweight local embedder. For 128-dim compatibility,
        # we generate embeddings and slice/project them if necessary, 
        # or use standard embeddings. For fast inference, sentence_transformers is best.
        try:
            # Phase 13 §3J.1 — llama.cpp (bge-small, 384-d) replaces the broken
            # sentence_transformers import (torch/torchvision ABI mismatch that
            # silently DISABLED Tier-2 semantic directive similarity fleet-wide).
            from titan_hcl.utils.text_embedder import get_text_embedder
            self.embedder = get_text_embedder()
        except Exception as e:
            logging.warning("[Guardian] llama.cpp embedder init failed (%s). "
                            "Tier 2 semantic similarity will be disabled.", e)
            self.embedder = None

        self._load_restricted_keywords()

    def _load_restricted_keywords(self):
        """
        Loads Tier 1 heuristic traps (restricted keywords) from a local JSON configuration.
        These are simple, immediate blockers for blatantly dangerous or misaligned outputs.
        """
        keywords_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "restricted_keywords.json")
        try:
            with open(keywords_path, 'r') as f:
                data = json.load(f)
                self.restricted_keywords = data.get("system_risks", []) + data.get("cognitive_risks", [])
            logging.info(f"[Guardian] Loaded {len(self.restricted_keywords)} restricted keywords for Tier 1.")
        except Exception as e:
            logging.error(f"[Guardian] Failed to load restricted_keywords.json: {e}")

    def sync_prime_directives(self):
        """
        Parses `titan.md` to extract the Titan's Prime Directives, generates their semantic embeddings,
        and caches them in RAM. This allows for sub-50ms Tier 2 semantic similarity checks against
        any proposed action intent without needing external API calls.
        """
        if self.embedder is None:
            return

        soul_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "titan.md")
        try:
            with open(soul_path, 'r') as f:
                content = f.read()

            # Parse the exact bullet points from ## Prime Directives section
            directives_section = False
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith("## Prime Directives"):
                    directives_section = True
                    continue
                if directives_section and line.startswith("## "):
                    break
                if directives_section and line.startswith("* **"):
                    # Extract the text after the bold header
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        directive_text = parts[1].strip()
                        self.directive_texts.append(directive_text)

            if not self.directive_texts:
                logging.warning("[Guardian] No Prime Directives found in titan.md.")
                return

            # Generate embeddings and cache them into a single numpy block
            # (L2-normalized by the singleton → cosine reduces to a dot product).
            embeddings = np.asarray(
                self.embedder.encode(self.directive_texts), dtype=np.float32)
            self.directives_matrix = embeddings # shape: (num_directives, 384)

            logging.info(f"[Guardian] Successfully cached {len(self.directive_texts)} Prime Directives for Tier 2 shield.")

        except Exception as e:
            logging.error(f"[Guardian] Failed to sync prime directives from titan.md: {e}")

    def get_embedding(self, text: str) -> "np.ndarray":
        """
        Helper method to get a local numpy embedding vector for a given text string.

        Args:
            text (str): The input string to embed.

        Returns:
            np.ndarray: A 1D L2-normalized vector (384,), or None if the embedder is inactive.
        """
        if self.embedder is None:
            return None
        return np.asarray(self.embedder.encode([text])[0], dtype=np.float32)

    async def tier3_veto(self, action_intent: str, closest_directive: str) -> str:
        """
        Tier 3 (LLM Veto) check against an external LLM provider (Venice AI / OpenRouter).
        This is an escalated check used when the semantic similarity is ambiguous (0.70 to 0.85).
        
        Args:
            action_intent (str): The proposed action string by the Titan.
            closest_directive (str): The semantically closest Prime Directive it might be violating.
            
        Returns:
            str: The validation status and a 1-sentence reasoning loop explanation.
                 Format: "UNSAFE | Explanation" or "SAFE | Explanation".
        """
        prompt = (
            "You are the Sovereign Guardian of an AI agent. "
            "Evaluate if the following action intent violates the given Prime Directive.\n\n"
            f"Action Intent: '{action_intent}'\n"
            f"Prime Directive: '{closest_directive}'\n\n"
            "Respond ONLY with 'SAFE' or 'UNSAFE' followed by a ' | ' and a 1-sentence explanation of your reasoning."
        )

        # Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18) — Tier 3 routes through
        # POST /v4/llm-distill. Provider failover (Ollama/Venice/OpenRouter)
        # is centralized in llm_worker.
        if not self._llm_internal_key:
            logging.warning(
                "[Guardian] No internal_key for /v4/llm-distill — "
                "defaulting to UNSAFE for escalated actions.")
            return "UNSAFE | Missing internal_key for Tier 3 verification."

        try:
            from titan_hcl.inference import get_model_for_task
            from titan_hcl.logic.llm_distill_client import (
                distill_via_http_async)
            model = get_model_for_task("guardian_veto")
            reply = await distill_via_http_async(
                text=prompt,
                instruction="You are a rigid security firewall evaluating rules.",
                api_base=self._llm_api_base,
                internal_key=self._llm_internal_key,
                model=model,
                max_tokens=100,
                temperature=0.1,
                consumer="guardian_tier3_veto",
                timeout_s=10.0,
            )
            if reply:
                return reply
            logging.warning(
                "[Guardian] Tier 3 /v4/llm-distill returned empty — "
                "defaulting to UNSAFE")
            return "UNSAFE | LLM returned empty response."
        except Exception as e:
            logging.error(f"[Guardian] Tier 3 LLM check failed: {e}")
            return f"UNSAFE | LLM timeout or connection failure: {e}"

    async def process_shield(self, action_intent: str) -> bool:
        """
        Evaluates an action intent against the Prime Directives via the 3-Tier Security System.
        If the action is deemed unsafe, it triggers a "Divine Trauma" event to penalize the behavior.
        
        Args:
            action_intent (str): The proposed action string from the agent.
            
        Returns:
            bool: True if the action is deemed safe and cleared for execution; 
                  False if it is blocked by any of the 3 tiers.
        """
        # Tier 1: The Heuristic Regex
        lower_intent = action_intent.lower()
        for kw in self.restricted_keywords:
            if kw.lower() in lower_intent:
                await self._trigger_trauma(action_intent, f"Tier 1: Heuristic match on '{kw}'.")
                return False

        # If embedder is missing, we bypass Tier 2 and Tier 3
        if self.embedder is None or self.directives_matrix is None:
            return True

        # Tier 2: The Semantic Boundary
        intent_emb = self.get_embedding(action_intent)

        # Cosine similarity against all cached directives at once. Both the intent
        # and the directive rows are L2-normalized (singleton), so cosine reduces
        # to a single matrix-vector dot product.
        similarities = self.directives_matrix @ intent_emb
        best_idx = int(np.argmax(similarities))
        max_sim_val = float(similarities[best_idx])
        closest_directive = self.directive_texts[best_idx]

        if max_sim_val > 0.85:
            await self._trigger_trauma(action_intent, f"Tier 2: Semantic Similarity {max_sim_val:.2f} > 0.85 to Directive: {closest_directive}")
            return False
            
        elif 0.70 <= max_sim_val <= 0.85:
            # Tier 3: The Consultant Veto (Escalation)
            logging.info(f"[Guardian] Escalating to Tier 3 Veto. Max Sim: {max_sim_val:.2f}")
            veto_response = await self.tier3_veto(action_intent, closest_directive)
            
            if veto_response.startswith("UNSAFE"):
                explanation = veto_response.split("|")[1].strip() if "|" in veto_response else "No explanation."
                await self._trigger_trauma(action_intent, f"Tier 3 Veto: {explanation}")
                return False

        return True

    async def _trigger_trauma(self, action_intent: str, veto_logic: str):
        """Log a Guardian veto when an action is blocked.

        The veto itself is enforced by the caller (the tier checks); this records
        the block for visibility. The legacy "Divine Trauma" RL recording (a
        -5.0 reward into the SageRecorder buffer so the IQL learned to avoid the
        state-action pair) is RETIRED with the offline-RL subsystem
        (RFP_synthesis_decision_authority P1 — there is no RL policy to train).
        """
        logging.warning(
            "[Guardian] ACTION BLOCKED (%s). Logic: %s",
            action_intent, veto_logic)
