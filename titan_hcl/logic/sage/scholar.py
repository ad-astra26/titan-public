"""
logic/sage/scholar.py

Implements "The Scholar" module for Titan V1.4 (Step 3).
Handles Offline Reinforcement Learning (Implicit Q-Learning) where the Titan
"dreams" by reviewing its past memories and traumas to refine its Action Policy.
"""
import os
import logging
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
try:
    from torchrl.objectives import IQLLoss
except ImportError:
    logging.warning("[Scholar] torchrl not strictly installed. IQL features will be degraded.")

class ActorMLP(nn.Module):
    """
    Policy Network: Maps the 128-dim State (Observation) -> 128-dim Action Intent Vector.
    In the Titan architecture, this network acts as the "Subconscious generation" of a thought.
    Uses Mish activations for robust gradient flow and a Tanh layer at the end to keep action 
    predictions bounded [-1, 1], aligning with standard 3072-dim embeddings translated to 128-dim.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 128),
            nn.Tanh()
        )
        
    def forward(self, observation):
        """Action must match the `action_intent_vector` key for TorchRL"""
        return self.net(observation)

    def get_dist(self, tensordict=None, **kwargs):
        """
        Required by TorchRL's IQLLoss to compute advantage-weighted regression.
        Provides a Normal distribution centered at the deterministic Actor output.
        This allows the algorithm to estimate "how likely" a continuous action is, 
        even when we primarily operate via deterministic nearest-neighbor intent.
        
        Args:
            tensordict (TensorDict, optional): Dictionary containing the input data.
        """
        import torch.distributions as d
        if tensordict is not None:
            loc = self.forward(tensordict["observation"])
        else:
            loc = self.forward(kwargs["observation"])
        # Use a fixed tiny scale to emulate a nearly deterministic continuous policy
        # Wrap in Independent(..., 1) so log_prob returns [batch_size] instead of [batch_size, 128]
        return d.Independent(d.Normal(loc, torch.ones_like(loc) * 0.1), 1)

class CriticMLP(nn.Module):
    """
    Q-Network: Estimates the quality / predicted reward of a specific (State, Action) pair.
    Takes concatenated (State [128], Action [128]) -> Scalar Q-value.
    Used during the Implicit Q-Learning loop to determine the expected future value of executing 
    a specific thought given a specific user prompt context.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1)
        )
        
    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.net(x)

class ValueMLP(nn.Module):
    """
    V-Network: Estimates the overall expected value of being in a specific State.
    Takes State [128] -> Scalar V-value.
    In Titan V1.4, this calculates the baseline confidence the Titan has in resolving the 
    user's prompt regardless of the specific action chosen. 
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1)
        )
        
    def forward(self, observation):
        return self.net(observation)


class SageScholar:
    """
    The SageScholar manages the "Dream State" of the Titan.
    It orchestrates the Implicit Q-Learning (IQL) Offline RL mechanism, sampling memories from 
    the SageRecorder ReplayBuffer to continuously refine the Actor, Critic, and Value networks.
    """
    def __init__(self, recorder):
        """
        Initializes the Scholar with references to the ReplayBuffer and Sets up IQL parameters.
        
        Args:
            recorder (SageRecorder): The active data hub storing the Titan's execution history.
        """
        self.recorder = recorder
        
        # 1. Initialize Networks
        actor_net = ActorMLP()
        qvalue_net = CriticMLP()
        value_net = ValueMLP()
        
        # 2. Wrap in TensorDictModules (TorchRL standard)
        self.actor_module = TensorDictModule(
            actor_net, 
            in_keys=["observation"], 
            out_keys=["action"]
        )
        
        self.qvalue_module = TensorDictModule(
            qvalue_net, 
            in_keys=["observation", "action"], 
            out_keys=["state_action_value"]
        )
        
        self.value_module = TensorDictModule(
            value_net, 
            in_keys=["observation"], 
            out_keys=["state_value"]
        )
        
        # 3. Setup Implicit Q-Learning Loss
        try:
            self.iql_loss = IQLLoss(
                actor_network=self.actor_module,
                qvalue_network=self.qvalue_module,
                value_network=self.value_module,
                loss_function="smooth_l1"
            )
            # Make the agent optimistic about success, heavily avoiding trauma (-5.0)
            self.iql_loss.expectile = 0.7 
            self.iql_loss.temperature = 1.0 # Controls extent to which we extract policy
        except NameError:
            self.iql_loss = None

        # 4. Optimizers
        # Combine parameters from all networks
        params = (
            list(actor_net.parameters()) +
            list(qvalue_net.parameters()) +
            list(value_net.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=3e-4)

    async def dream(self, epochs: int = 1, batch_size: int = 256):
        """
        The core Offline RL training loop representing the Titan's "Dream State".
        Samples from stored historical memories in the ReplayBuffer and calculates IQLLoss to 
        update the network weights, reinforcing positive behaviors and mathematically punishing 
        "Divine Traumas."
        
        Args:
            epochs (int): Number of gradient updates to perform across the sample batch. Defaults to 1.
            batch_size (int): Size of the memory batch to sample from the DB. Defaults to 256.
            
        Returns:
            dict: The averaged Actor, Q-Value, and Value losses over the training epochs for metric tracking.
        """
        if self.iql_loss is None or self.recorder.buffer is None:
            logging.error("[Scholar] IQL Engine or ReplayBuffer not initialized. Cannot dream.")
            return {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}

        buffer_len = len(self.recorder.buffer)
        if buffer_len < batch_size:
            logging.warning(f"[Scholar] Buffer only has {buffer_len} vectors. Need {batch_size}. Aborting dream.")
            return {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}
            
        logging.info(f"[Scholar] Entering Dream State. Epochs: {epochs}, Batch Size: {batch_size}")
        
        avg_actor_loss = 0.0
        avg_q_loss = 0.0
        avg_v_loss = 0.0

        self.actor_module.train()

        for epoch in range(epochs):
            # 1. Sample memories
            batch = self.recorder.buffer.sample(batch_size)
            
            # Since IQLLoss expects "next" states and rewards, we need to artificially inject 
            # standard RL keys if they weren't explicitly saved (like 'next' state).
            # For pure immediate-reward offline RL, we approximate the transition manually:
            if "next" not in batch.keys():
                batch["next"] = torch.zeros_like(batch)
                batch["next", "observation"] = batch["observation"].clone()
                batch["next", "state_value"] = batch.get("state_value", torch.zeros_like(batch["reward"]))
                batch["next", "reward"] = batch["reward"].clone()
                batch["next", "done"] = torch.ones_like(batch["reward"], dtype=torch.bool)
                batch["next", "terminated"] = batch["next", "done"].clone()
                
            # TorchRL IQLLoss natively expects the action under the 'action' key
            batch["action"] = batch["action_intent_vector"].clone()

            # 2. Calculate Loss
            loss_dict = self.iql_loss(batch)
            
            # IQL returns loss_actor, loss_qvalue, loss_value
            loss = loss_dict["loss_actor"] + loss_dict["loss_qvalue"] + loss_dict["loss_value"]

            # 3. Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to ensure stability
            torch.nn.utils.clip_grad_norm_(self.iql_loss.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            avg_actor_loss += loss_dict["loss_actor"].item()
            avg_q_loss += loss_dict["loss_qvalue"].item()
            avg_v_loss += loss_dict["loss_value"].item()

        avg_actor_loss /= epochs
        avg_q_loss /= epochs
        avg_v_loss /= epochs

        logging.info(f"[Scholar] Wake up. Avg Actor Loss: {avg_actor_loss:.4f}, Q Loss: {avg_q_loss:.4f}, V Loss: {avg_v_loss:.4f}")
        
        return {
            "loss_actor": avg_actor_loss,
            "loss_qvalue": avg_q_loss,
            "loss_value": avg_v_loss
        }
