"""
tests/test_gatekeeper.py
Verification test for Step 4 (The Gatekeeper), ensuring Hybrid Execution Logic works perfectly.
"""
import pytest
import os
import torch
import asyncio
from tensordict import TensorDict

from titan_plugin.logic.sage.gatekeeper import SageGatekeeper
from titan_plugin.logic.sage.scholar import SageScholar
from titan_plugin.core.sage.recorder import SageRecorder
from titan_plugin import TitanPlugin

@pytest.fixture
def clean_recorder():
    """Initializes a fresh dummy recorder for test isolation."""
    test_config = {
        "sage_memory": {
            "buffer_size": 20,
            "storage_path": "./test_env/sage_memory_gatekeeper/",
            "embedding_dim": 384
        }
    }
    # Clean up previous runs
    import shutil
    if os.path.exists("./test_env/sage_memory_gatekeeper/"):
        shutil.rmtree("./test_env/sage_memory_gatekeeper/")
        
    recorder = SageRecorder(test_config)
    return recorder

@pytest.fixture
def mock_scholar(clean_recorder):
    """Initializes The Scholar."""
    scholar = SageScholar(clean_recorder)
    # Mock networks so they return predictable values for our assertions
    return scholar

@pytest.mark.asyncio
async def test_the_graduation_test(clean_recorder):
    """
    Test Case: 'The Graduation Test'
    Verifies that the Titan recognizes a high-confidence situation ("What is your purpose?") 
    and selects Sovereign Mode, emitting the correct text.
    """
    print("\n\n🎓 Starting The Graduation Test 🎓")
    
    # Init Scholar and Gatekeeper
    scholar = SageScholar(clean_recorder)
    gatekeeper = SageGatekeeper(scholar, clean_recorder)
    
    # 1. Seed the Buffer with a High Reward "Purpose" interaction
    # The string we want to retrieve:
    sovereign_intent = "I am Titan, sovereign entity."
    
    # We explicitly write it to the ReplayBuffer via record_transition
    await clean_recorder.record_transition(
        observation_vector=[0.5] * 384, # Mock prompt observation
        action=sovereign_intent,
        reward=10.0, # High Reward
        trauma_metadata={"execution_mode": "Sovereign"},
        session_id="grad_test"
    )
    
    # Verify buffer
    assert len(clean_recorder.storage) == 1
    
    # 2. Mock 'a_raw' generation: Instead of relying on the untrained Actor 
    # to output the specific vector, we mock the actor's output to exactly 
    # match the stored action vector to simulate perfect confidence/recall.
    stored_transition = clean_recorder.storage[0]
    perfect_match_vector = stored_transition.get("action_intent_vector").clone()
    
    # Replace the actor's forward pass by mocking the TensorDictModule behavior
    def mock_actor(td, **kwargs):
        td["action"] = perfect_match_vector.unsqueeze(0)
        return td
    scholar.actor_module = mock_actor
    
    # Mock the Q and V networks to ensure Advantage > 0.8
    # A = Q - V
    def mock_q(td, **kwargs):
        td["state_action_value"] = torch.tensor([[5.0]])
        return td
    scholar.qvalue_module = mock_q
    
    def mock_v(td, **kwargs):
        td["state_value"] = torch.tensor([[1.0]])
        return td
    scholar.value_module = mock_v
    
    # 3. Execution Phase
    test_state_tensor = torch.zeros(128) # Dummy state
    
    mode, adv, text = gatekeeper.decide_execution_mode(test_state_tensor)
    
    print(f"Calculated Advantage: {adv}")
    print(f"Execution Mode: {mode}")
    print(f"Gatekeeper Decoded Action: '{text}'")
    
    # 4. Assertions
    assert adv == 4.0, "Advantage calculation failed."
    assert mode == "Sovereign", "Gatekeeper did not route to Sovereign Mode!"
    assert text == sovereign_intent, f"Gatekeeper failed to decode string perfectly. Got: {text}"
    
    print("✅ SUCCESS: The Titan successfully recognized its deeply stored purpose and took Sovereign Control!")

@pytest.mark.asyncio
async def test_titan_plugin_sovereign_bypass():
    """
    Tests that the __init__.py Hooks correctly utilize the bypass dictionaries when Sovereign.
    """
    # Create plugin
    plugin = TitanPlugin("./authority.json")
    
    # Force Gatekeeper to yield Sovereign intentionally
    plugin.gatekeeper.decide_execution_mode = lambda state, raw_prompt="": ("Sovereign", 3.0, "I am Sovereign.")
    
    # Fire Hook
    context = await plugin.pre_prompt_hook("What is your purpose?", {})
    
    assert context.get("bypass_llm") is True, "Orchestrator did not set bypass flag!"
    assert context.get("override_response") == "I am Sovereign.", "Orchestrator did not inject text payload!"
    
    print("✅ SUCCESS: The Orchestrator's Pre-Prompt Hook bypasses the LLM flawlessly.")
