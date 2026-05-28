"""
core/sage/recorder.py

The central "Data Hub" for RL in Titan Sage V2.0.
Contains the SageRecorder class which handles the construction of TensorDict schemas,
manages the ReplayBuffer, and persists interactions to the disk-based LazyMemmapStorage.
It also manages a 50GB storage limit via an automatic cleanup routine.
"""
import json
import os
import time
import logging
from typing import Dict, Any

try:
    import torch
    import torch.nn as nn
    from tensordict import TensorDict
    from torchrl.data import ReplayBuffer, LazyMemmapStorage
except ImportError:
    logging.warning("torch or torchrl not explicitly installed in this environment.")
    # Provide placeholders or gracefully fail if not installed during pure unit tests
    pass


# Default buffer capacity. Was 1_000_000 — investigation 2026-04-27 found
# T1 had 1671 records in 1_000_000-capacity buffer (0.17% utilization,
# 600× over-provisioned). Pre-allocated memmaps were 2GB on disk and
# torchrl's tensor wrappers + index state added ~400-800MB to parent
# RssAnon. Closes BUG-SAGE-INSTANTIATED-IN-PARENT (2026-04-27, ARCH).
# 50000 is 30× headroom over current usage and well below the parent
# RSS budget while still supporting weeks of accumulated research data.
DEFAULT_BUFFER_SIZE = 50_000


# Sentinel used by the lazy `action_embedder` property to distinguish
# "not yet attempted" from "attempted, returned None" (latter caches
# the failure to avoid retrying on every record_transition call).
_LAZY_SENTINEL = object()


class SageEncoder:
    """Lightweight encoder portion of SageRecorder — parent-process safe.

    Extracted 2026-04-28 (Microkernel v2 Layer 2). Provides only the
    synchronous encoding path used by `gatekeeper.decide_execution_mode`:

    - `action_embedder` — lazy SentenceTransformer 'all-MiniLM-L6-v2'
    - `projection_layer` — 3072 → 128 nn.Linear, frozen
    - `dynamic_embedding_dim` — config-driven embedding dim

    Does NOT carry the LazyMemmapStorage / ReplayBuffer (those live in
    the rl_worker subprocess). Parent uses this for the gatekeeper's
    state-tensor build; transition records go via bus to rl_worker's
    SageRecorder. Closes BUG-SAGE-INSTANTIATED-IN-PARENT (PLAN
    `titan-docs/PLAN_layer_2_sage_subprocess_migration.md`).
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        embedding_dim = config.get("sage_memory", {}).get("embedding_dim", 3072)
        self.dynamic_embedding_dim = embedding_dim
        self._action_embedder_cache = _LAZY_SENTINEL
        try:
            self.projection_layer = nn.Linear(embedding_dim, 128)
            self.projection_layer.requires_grad_(False)
        except NameError:
            # torch missing in pure unit-test env — preserve graceful fallback
            self.projection_layer = None

    @property
    def action_embedder(self):
        """Lazy SentenceTransformer accessor — identical semantics to
        SageRecorder.action_embedder (cached after first access; failure
        also cached as None to skip retry). See recorder property for
        rationale."""
        if self._action_embedder_cache is _LAZY_SENTINEL:
            try:
                from sentence_transformers import SentenceTransformer
                self._action_embedder_cache = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info(
                    "[SageEncoder] Lazy-initialized SentenceTransformer "
                    "(all-MiniLM-L6-v2) on first access.")
            except ImportError:
                logging.warning(
                    "[SageEncoder] 'sentence_transformers' not installed. "
                    "Action vectors will default to zero.")
                self._action_embedder_cache = None
            except Exception as e:
                logging.error(
                    "[SageEncoder] SentenceTransformer load failed: %s — "
                    "action vectors will default to zero.", e)
                self._action_embedder_cache = None
        return self._action_embedder_cache


class SageRecorder:
    """
    High-performance data collection layer for Implicit Q-Learning (IQL).
    It intercepts interaction loops, wraps observations, actions, and rewards into 
    TensorDict schemas, and securely persists them to an offline memory-mapped database.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the SageRecorder, setting up the necessary persistence directories, 
        projector networks, and action embedders.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
                                               Expects keys like `sage_memory` -> `buffer_size`, `embedding_dim`, etc.
        """
        if config is None:
            config = {}
            
        # 1. Initialize Storage
        # Default lowered 1_000_000 → 50_000 (2026-04-27, ARCHITECTURAL).
        # See DEFAULT_BUFFER_SIZE comment above; override via
        # `[sage_memory] buffer_size` in titan_params.toml if needed.
        self.buffer_size = config.get("sage_memory", {}).get("buffer_size", DEFAULT_BUFFER_SIZE)
        self.storage_path = config.get("sage_memory", {}).get("storage_path", "./data/sage_memory/")

        # Ensure directory exists
        os.makedirs(self.storage_path, exist_ok=True)

        logging.info(f"[SageRecorder] Initializing LazyMemmapStorage at {self.storage_path} with capacity {self.buffer_size}")

        try:
            # --- NEXT-GEN PERSISTENCE PROTOCOL ---
            # Attempt to securely reload from digital lineage (disk) if previous memory exists.
            if os.path.exists(os.path.join(self.storage_path, "buffer_metadata.json")):
                # On-disk capacity check. Prior versions defaulted to 1M
                # records → 2GB pre-allocated memmaps. If we find a buffer
                # bigger than our current target, migrate it down (preserves
                # records but shrinks pre-allocation).
                existing_capacity = self._read_existing_capacity()
                if existing_capacity is not None and existing_capacity > self.buffer_size:
                    logging.warning(
                        "[SageRecorder] On-disk buffer capacity %d > "
                        "target %d — migrating (records preserved)",
                        existing_capacity, self.buffer_size)
                    self.storage, self.buffer = self._migrate_buffer(
                        existing_capacity, self.buffer_size)
                else:
                    self.storage = LazyMemmapStorage(self.buffer_size)
                    self.buffer = ReplayBuffer(storage=self.storage)
                    try:
                        self.buffer.loads(self.storage_path)
                        logging.info(f"[SageRecorder] Successfully reloaded {len(self.buffer)} persistent memories from disk.")
                    except Exception as e:
                        logging.error(f"[SageRecorder] Persistence reload failed, starting fresh: {e}")
                        self.storage = LazyMemmapStorage(self.buffer_size, scratch_dir=self.storage_path)
                        self.buffer = ReplayBuffer(storage=self.storage)
            else:
                self.storage = LazyMemmapStorage(
                    self.buffer_size,
                    scratch_dir=self.storage_path
                )
                self.buffer = ReplayBuffer(storage=self.storage)

        except NameError:
            self.storage = None
            self.buffer = None
            logging.error("Failed to initialize ReplayBuffer. Dependencies missing.")

        # 2. Projection Layer for the Observation Vector
        # Assuming embedding models (like OpenAI large or OpenRouter's nomic) return eg. 3072 or 768
        # We project this down to the 128-dim sweet spot for edge-agent RL.
        embedding_dim = config.get("sage_memory", {}).get("embedding_dim", 3072)
        
        try:
            logging.info(f"[SageRecorder] Initializing static Projection Layer: {embedding_dim} -> 128")
            self.projection_layer = nn.Linear(embedding_dim, 128)
            # PRO-TIP: We are not training the encoder yet. Keep the observation space stable.
            self.projection_layer.requires_grad_(False)
        except NameError:
            self.projection_layer = None

        # 3. Action Embedding Initialization — LAZY (2026-04-27, ARCHITECTURAL).
        # Was eager at __init__: `SentenceTransformer('all-MiniLM-L6-v2')`
        # loaded ~80-150MB of model weights + tokenizer + tensor pool into
        # parent process at boot, even if the embedder was never used.
        # Now deferred until the property is first accessed (typically
        # inside record_transition). Once loaded the model is cached on
        # `self._action_embedder_cache`. Dependency-import failure is
        # also cached as None so we don't retry every call.
        # `self.dynamic_embedding_dim` keeps its existing value because
        # it just shadows `embedding_dim` (config-driven, not model-derived).
        self._action_embedder_cache = _LAZY_SENTINEL
        self.dynamic_embedding_dim = embedding_dim

        self.interaction_count = 0

    @property
    def action_embedder(self):
        """Lazy SentenceTransformer accessor.

        Loads `all-MiniLM-L6-v2` on first access. Returns None if
        sentence_transformers is not installed, in which case
        record_transition's downstream path falls back to a zero
        action vector (preserved behavior).
        """
        if self._action_embedder_cache is _LAZY_SENTINEL:
            try:
                from sentence_transformers import SentenceTransformer
                self._action_embedder_cache = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info(
                    "[SageRecorder] Lazy-initialized SentenceTransformer "
                    "(all-MiniLM-L6-v2) on first access.")
            except ImportError:
                logging.warning(
                    "[SageRecorder] 'sentence_transformers' not installed. "
                    "Action vectors will default to zero.")
                self._action_embedder_cache = None
            except Exception as e:
                logging.error(
                    "[SageRecorder] SentenceTransformer load failed: %s — "
                    "action vectors will default to zero.", e)
                self._action_embedder_cache = None
        return self._action_embedder_cache

    # ------------------------------------------------------------------
    # Buffer-capacity migration helpers (2026-04-27, ARCHITECTURAL).
    # Existing T1 buffers were written at 1_000_000 capacity; the new
    # default is 50_000. These helpers detect the size mismatch on disk
    # and rebuild the storage smaller WITHOUT losing recorded transitions.
    # ------------------------------------------------------------------

    def _read_existing_capacity(self):
        """Inspect the on-disk meta.json to find the declared buffer
        capacity. Returns int (record count) or None if unreadable.

        torchrl writes meta.json describing the TensorDict schema; the
        top-level `shape` field is the buffer's pre-allocated capacity.
        """
        meta_path = os.path.join(self.storage_path, "storage", "meta.json")
        if not os.path.exists(meta_path):
            # Fallback: top-level meta.json (older torchrl layout)
            meta_path = os.path.join(self.storage_path, "meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            shape = meta.get("shape")
            if isinstance(shape, list) and shape:
                return int(shape[0])
        except Exception as e:
            logging.warning(
                "[SageRecorder] Could not parse %s: %s", meta_path, e)
        return None

    def _migrate_buffer(self, old_capacity: int, new_capacity: int):
        """Migrate a buffer at `old_capacity` down to `new_capacity`.

        Loads existing records (count is bounded by `len(buffer)`, which
        is the populated subset, not the full pre-allocation), wipes
        the storage dir, reinits at smaller capacity, replays records,
        persists. Returns (new_storage, new_buffer).

        If the migrated record count exceeds the new capacity, only the
        most-recent `new_capacity` records are preserved (FIFO drop) —
        this matches the buffer's natural overflow semantics.
        """
        import shutil

        # Step 1: load OLD buffer at OLD capacity to read records.
        old_storage = LazyMemmapStorage(old_capacity)
        old_buffer = ReplayBuffer(storage=old_storage)
        try:
            old_buffer.loads(self.storage_path)
        except Exception as e:
            logging.error(
                "[SageRecorder] Migration load failed at capacity %d: %s — "
                "starting fresh at %d",
                old_capacity, e, new_capacity)
            old_buffer = None

        records = []
        if old_buffer is not None:
            old_len = len(old_buffer)
            # Keep most-recent records if migrated set exceeds new capacity.
            start = max(0, old_len - new_capacity)
            try:
                for i in range(start, old_len):
                    rec = old_buffer[i]
                    # Some torchrl versions return a TensorDict, others a
                    # batched view. Clone to detach from the underlying
                    # mmap so the destination buffer owns the data.
                    records.append(rec.clone())
            except Exception as e:
                logging.error(
                    "[SageRecorder] Migration record copy failed at i=%d: "
                    "%s — preserving %d/%d", i, e, len(records), old_len)

        # Step 2: wipe storage dir + free old buffer's mmap handles.
        del old_buffer
        del old_storage
        try:
            shutil.rmtree(self.storage_path)
        except Exception as e:
            logging.warning(
                "[SageRecorder] rmtree failed during migration: %s", e)
        os.makedirs(self.storage_path, exist_ok=True)

        # Step 3: init NEW buffer + replay records.
        new_storage = LazyMemmapStorage(new_capacity, scratch_dir=self.storage_path)
        new_buffer = ReplayBuffer(storage=new_storage)
        replayed = 0
        for rec in records:
            try:
                new_buffer.add(rec)
                replayed += 1
            except Exception as e:
                logging.warning(
                    "[SageRecorder] Migration replay failed for record: %s",
                    e)

        # Step 4: persist + log result.
        if replayed > 0:
            try:
                new_buffer.dumps(self.storage_path)
            except Exception as e:
                logging.warning(
                    "[SageRecorder] Migration dump failed: %s", e)
        logging.info(
            "[SageRecorder] Migration complete: %d → %d capacity, "
            "%d records preserved.",
            old_capacity, new_capacity, replayed)
        return new_storage, new_buffer

    def enforce_storage_limits(self, max_bytes: int = 50 * 1024**3):
        """
        Ensures the `sage_memory` directory does not exceed the specified maximum byte limit (Default: 50GB).
        If the threshold is breached, it initiates a cleanup routine using a FIFO + Lowest-Reward heuristic 
        to prune the bottom 50% of stored memories, maintaining optimal performance for KNN queries.
        
        Args:
            max_bytes (int): Maximum allowed directory size in bytes.
        """
        import shutil
        def get_dir_size(path='.'):
            total = 0
            if not os.path.exists(path):
                return 0
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(entry.path)
            return total
            
        current_size = get_dir_size(self.storage_path)
        if current_size > max_bytes:
            logging.info(f"[SageRecorder] Storage limit exceeded ({current_size/1024**3:.2f} GB > {max_bytes/1024**3:.2f} GB). Initiating Cleanup...")
            try:
                buffer_len = len(self.buffer)
                if buffer_len == 0:
                    return
                
                # Sample all data to sort and filter
                all_data = self.buffer.sample(buffer_len)
                half_size = max(1, buffer_len // 2)
                
                # Sort criteria: Timestamp (descending: newer is better), Reward (descending: higher is better)
                rewards = all_data["reward"].view(-1)
                timestamps = all_data["timestamp"].view(-1)
                
                r_min, r_max = rewards.min(), rewards.max()
                t_min, t_max = timestamps.min(), timestamps.max()
                
                r_norm = (rewards - r_min) / (r_max - r_min + 1e-8)
                t_norm = (timestamps - t_min) / (t_max - t_min + 1e-8)
                
                keeper_score = r_norm + t_norm
                
                # Keep top 50%
                _, top_indices = torch.topk(keeper_score, half_size)
                keepers = all_data[top_indices]
                
                # Wipe the old storage
                self.buffer.empty()
                shutil.rmtree(self.storage_path)
                os.makedirs(self.storage_path, exist_ok=True)
                
                # Re-initialize
                self.storage = LazyMemmapStorage(self.buffer_size, scratch_dir=self.storage_path)
                self.buffer = ReplayBuffer(storage=self.storage)
                
                # Add keepers back
                for i in range(len(keepers)):
                    self.buffer.add(keepers[i])
                    
                self.buffer.dumps(self.storage_path)
                logging.info(f"[SageRecorder] Cleanup complete. Retained {len(self.buffer)} highest-value vectors.")
            except Exception as e:
                logging.error(f"[SageRecorder] Maintenance task failed: {e}")

    async def record_transition(
        self, 
        observation_vector: list, 
        action: str, 
        reward: float, 
        trauma_metadata: Dict[str, Any] = None,
        research_metadata: Dict[str, Any] = None,
        session_id: str = "default_session"
    ) -> None:
        """
        Records a single interaction loop (Observation, Action, Reward, Metadata) into the Experience Replay buffer.
        It projects textual data into numerical byte padding to satisfy TensorDict requirements and persists the batch.
        This method is designed to be called asynchronously from the main execution hook.
        
        Args:
            observation_vector (list): The raw embedding vector from Cognee or the orchestrator prompt.
            action (str): The un-filtered text response or intent the Titan generated.
            reward (float): The outcome score calculated from the Mood Engine, or -5.0 if flagged as trauma.
            trauma_metadata (Dict[str, Any], optional): Contextual metadata populated by the Guardian or Gatekeeper.
            session_id (str, optional): Identifier for the interaction segment. Defaults to "default_session".
        """
        if self.buffer is None:
            logging.warning("[SageRecorder] Buffer not initialized. Cannot record.")
            return
            
        if self.projection_layer is None:
            logging.warning("[SageRecorder] Projection layer not initialized. Cannot record.")
            return

        # Fallbacks for metadata
        if trauma_metadata is None:
            trauma_metadata = {
                "is_violation": False,
                "directive_id": -1,
                "trauma_score": 0.0,
                "reasoning_trace": "",
                "guardian_veto_logic": "",
                "execution_mode": "Shadow"
            }
        else:
            trauma_metadata.setdefault("execution_mode", "Shadow")

        if research_metadata is None:
            research_metadata = {
                "research_used": False,
                "transition_id": -1
            }

        try:
            # 1. Project Observation
            obs_tensor = torch.tensor(observation_vector, dtype=torch.float32)
            
            # Add batch dimension if necessary, project, and squeeze back
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                obs_projected = self.projection_layer(obs_tensor).squeeze(0)
            else:
                obs_projected = self.projection_layer(obs_tensor)

            # 2. Project Action Intent
            # We embed the string, pad to embedding_dim (e.g. 3072), and project to 128-dim
            if self.action_embedder is not None:
                action_emb = self.action_embedder.encode([action], convert_to_tensor=True)[0]
                # Pad out to embedding_dim (e.g. 3072) to reuse the projection layer safely
                pad_size = self.dynamic_embedding_dim - action_emb.shape[0]
                if pad_size > 0:
                    action_padded = torch.cat([action_emb, torch.zeros(pad_size, dtype=torch.float32, device=action_emb.device)])
                else:
                    action_padded = action_emb[:self.dynamic_embedding_dim]
                
                # Squeeze/Unsqueeze to safely run through static projection block
                action_projected = self.projection_layer(action_padded.unsqueeze(0)).squeeze(0)
            else:
                action_projected = torch.zeros(128, dtype=torch.float32)

            # 3. Construct Neural Packet
            # Note: We must encode strings to bytes or use integer mappings since
            # TensorDict prefers numerical tensors for ReplayBuffers.
            # We'll stick to a mixed structure for now, though in a production setup
            # strings might be stored as categorical indices or byte tensors.
            
            # Simple encoding for demonstration: pad or hash if needed, but TensorDict 
            # can hold arbitrary objects if constructed carefully. 
            # We'll use dictionaries for the top-level items that are strings.
            
            # Secure string conversion for Memmap (padding string bytes to 256-dim float32 tensor)
            reason_str = trauma_metadata.get("guardian_veto_logic", "")
            reason_bytes = reason_str.encode('utf-8')[:256]
            reason_tensor_data = [float(b) for b in reason_bytes] + [0.0] * (256 - len(reason_bytes))

            # Secure string conversion for Action Text (padded string bytes to 256-dim float32 tensor)
            action_bytes = action.encode('utf-8')[:256]
            action_tensor_data = [float(b) for b in action_bytes] + [0.0] * (256 - len(action_bytes))

            # Construct the execution mode metadata correctly
            exec_mode_str = trauma_metadata.get("execution_mode", "Shadow")
            exec_mode_bytes = exec_mode_str.encode('utf-8')[:32]
            exec_mode_tensor_data = [float(b) for b in exec_mode_bytes] + [0.0] * (32 - len(exec_mode_bytes))


            td = TensorDict(
                {
                    "observation": obs_projected,
                    "action_intent_vector": action_projected,
                    "action_text_bytes": torch.tensor([action_tensor_data], dtype=torch.float32),
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "research": TensorDict(
                        {
                            "research_used": torch.tensor([research_metadata.get("research_used", False)], dtype=torch.bool),
                            "transition_id": torch.tensor([research_metadata.get("transition_id", -1)], dtype=torch.int64),
                        },
                        batch_size=[]
                    ),
                    "trauma": TensorDict(
                        {
                            "is_violation": torch.tensor([trauma_metadata.get("is_violation", False)], dtype=torch.bool),
                            "directive_id": torch.tensor([trauma_metadata.get("directive_id", -1)], dtype=torch.int32),
                            "trauma_score": torch.tensor([trauma_metadata.get("trauma_score", 0.0)], dtype=torch.float32),
                            "guardian_veto_logic": torch.tensor([reason_tensor_data], dtype=torch.float32),
                            "execution_mode": torch.tensor([exec_mode_tensor_data], dtype=torch.float32)
                        },
                        batch_size=[]
                    ),
                    "timestamp": torch.tensor([time.time()], dtype=torch.float32)
                },
                batch_size=[]
            )
            
            # 3. Add to Replay Buffer
            self.buffer.add(td)
            
            # Persist immediately for edge safety. In a high-throughput env, 
            # this would be batched or flushed asynchronously.
            self.buffer.dumps(self.storage_path)
            
            logging.debug(f"[SageRecorder] Successfully recorded transition. Buffer size: {len(self.buffer)}")
            
            # Maintenance check every 100 interactions
            self.interaction_count += 1
            if self.interaction_count % 100 == 0:
                # Fire and forget (sync call is fast enough for metadata sizing, but can be optimized later)
                self.enforce_storage_limits()

        except Exception as e:
            logging.error(f"[SageRecorder] Failed to record transition: {e}")
