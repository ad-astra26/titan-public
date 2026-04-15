"""
core/sage/recorder.py

The central "Data Hub" for RL in Titan Sage V2.0.
Contains the SageRecorder class which handles the construction of TensorDict schemas,
manages the ReplayBuffer, and persists interactions to the disk-based LazyMemmapStorage.
It also manages a 50GB storage limit via an automatic cleanup routine.
"""
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
        # Default to 1,000,000 transitions if not configured
        self.buffer_size = config.get("sage_memory", {}).get("buffer_size", 1_000_000)
        self.storage_path = config.get("sage_memory", {}).get("storage_path", "./data/sage_memory/")
        
        # Ensure directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        logging.info(f"[SageRecorder] Initializing LazyMemmapStorage at {self.storage_path} with capacity {self.buffer_size}")
        
        try:
            # --- NEXT-GEN PERSISTENCE PROTOCOL ---
            # Attempt to securely reload from digital lineage (disk) if previous memory exists.
            if os.path.exists(os.path.join(self.storage_path, "buffer_metadata.json")):
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

        # 3. Action Embedding Initialization
        # Load a lightweight embedder for the Action intent
        try:
            from sentence_transformers import SentenceTransformer
            self.action_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.dynamic_embedding_dim = embedding_dim
            logging.info("[SageRecorder] Initialized SentenceTransformer for Action Intent projecting.")
        except ImportError:
            logging.warning("[SageRecorder] 'sentence_transformers' not installed. Action vectors will default to zero.")
            self.action_embedder = None
            
        self.interaction_count = 0

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
