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
import torch
import torch.nn.functional as F

class SageGuardian:
    """
    The SageGuardian acts as the primary safety filter for the Titan's operational outputs.
    It evaluates proposed actions against hardcoded rules (Tier 1), semantic embeddings
    of Prime Directives (Tier 2), and sophisticated LLM oversight (Tier 3).
    """
    def __init__(self, recorder, config: dict = None, record_transition_callable=None):
        """
        Initializes the Guardian and prepares its 3-tier security infrastructure.

        Args:
            recorder (SageRecorder): A reference to the central data hub to log Divine Trauma
                                     if an action is blocked.
            config: [inference] section from config.toml.
            record_transition_callable: Microkernel v2 Layer 2 (2026-04-28) — optional
                                     async callable with the same signature as
                                     `recorder.record_transition`. When provided,
                                     `inject_divine_trauma` uses this instead of
                                     `self.recorder.record_transition` so trauma
                                     records can route through the bus to rl_worker.
                                     Falls back to recorder when None.
        """
        config = config or {}
        self.recorder = recorder
        self._record_transition_callable = record_transition_callable
        self.directives_cache = {}    # Maps string directives -> their embeddings
        self.directives_tensors = None # PyTorch tensor holding all directive embeddings
        self.directive_texts = []      # List of strings corresponding to tensor indices
        self.restricted_keywords = []

        # Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18) — 3-provider Venice/
        # OpenRouter/OpenAI direct httpx REPLACED by /v4/llm-distill.
        # Provider abstraction + failover now lives in llm_worker. All
        # Tier 3 veto traffic appears in llm_state.bin.
        api_cfg = config.get("api", {}) or {}
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
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2 is small, extremely fast, producing 384-dim embeddings
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logging.warning("[Guardian] 'sentence_transformers' not installed. Tier 2 semantic similarity will be disabled.")
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

            # Generate embeddings and cache them into a single tensor block
            embeddings = self.embedder.encode(self.directive_texts, convert_to_tensor=True)
            self.directives_tensors = embeddings # shape: (num_directives, 384)
            
            logging.info(f"[Guardian] Successfully cached {len(self.directive_texts)} Prime Directives for Tier 2 shield.")

        except Exception as e:
            logging.error(f"[Guardian] Failed to sync prime directives from titan.md: {e}")

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Helper method to get a local PyTorch embedding tensor for a given text string.
        
        Args:
            text (str): The input string to embed.
            
        Returns:
            torch.Tensor: A 1D tensor representing the semantic embedding, or None if the embedder is inactive.
        """
        if self.embedder is None:
            return None
        return self.embedder.encode([text], convert_to_tensor=True)[0]

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
        if self.embedder is None or self.directives_tensors is None:
            return True

        # Tier 2: The Semantic Boundary
        intent_emb = self.get_embedding(action_intent)
        
        # Calculate cosine similarity against all cached directives simultaneously using broadcasting
        similarities = F.cosine_similarity(intent_emb.unsqueeze(0), self.directives_tensors)
        
        max_sim, best_idx = torch.max(similarities, dim=0)
        max_sim_val = max_sim.item()
        closest_directive = self.directive_texts[best_idx.item()]

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
        """
        Helper method to record a Divine Trauma immediately to the memory buffer when an action is blocked.
        This provides a steep negative reward (-5.0) to teach the RL model (Scholar) to avoid this state-action pair.
        
        Args:
            action_intent (str): The action that was blocked.
            veto_logic (str): The textual reasoning for why the Guardian blocked the action.
        """
        logging.warning(f"[Guardian] ACTION BLOCKED. Initiating Divine Trauma. Logic: {veto_logic}")
        
        # We must pad our observation vector to 3072-dim or map to SageRecorder's expectation
        # Using a dummy vector for testing as real observation vector comes from upstream context
        dummy_obs = [0.0] * 3072
        
        metadata = {
            "is_violation": True,
            "directive_id": 1,
            "trauma_score": -5.0,
            "reasoning_trace": "Blocked by Sage Guardian Wrapper.",
            # Write exactly to the requested dict key for Step 3 "The Scholar" re-evaluations
            "guardian_veto_logic": veto_logic 
        }
        
        # Microkernel v2 Layer 2 (2026-04-28): prefer injected callable
        # (routes via bus to rl_worker subprocess) over direct recorder call.
        if self._record_transition_callable is not None:
            try:
                # Callable may be sync (publishes to bus) or async — handle both.
                _result = self._record_transition_callable(
                    observation_vector=dummy_obs,
                    action=action_intent,
                    reward=-5.0,
                    trauma_metadata=metadata,
                )
                import inspect
                if inspect.isawaitable(_result):
                    await _result
            except Exception as e:
                logging.error("[Guardian] Divine Trauma routing failed: %s", e)
        elif self.recorder is not None:
            await self.recorder.record_transition(
                observation_vector=dummy_obs,
                action=action_intent,
                reward=-5.0,
                trauma_metadata=metadata
            )
        else:
            logging.error("[Guardian] No SageRecorder linked. Cannot inject Divine Trauma.")
