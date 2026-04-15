import asyncio
import os
import sys
import logging
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin import init_plugin

logging.basicConfig(level=logging.INFO)

async def run_convergence_test():
    print("\n🧠 Starting The Scholar Convergence Test (IQL Offline RL) 🧠")
    
    import shutil
    if os.path.exists("./data/sage_memory"):
        shutil.rmtree("./data/sage_memory")
    
    # 1. Initialize Full Pipeline
    config = {"WALLET_KEYPAIR_PATH": "./authority.json"}
    plugin = init_plugin(config)
    
    # Make sure we have some memories before we dream.
    # The buffer might be empty if we reset the VPS or cleaned memory, so let's push some mock data using Guardian triggers.
    
    # Check buffer length
    length = len(plugin.recorder.buffer) if plugin.recorder.buffer else 0
    print(f"\n[Test] Initial SageRecorder Buffer length: {length}")
    
    # Guarantee we have at least 5 transitions (3 Traumas, 2 Safes) encoded with the NEW 128-dim action protocol
    print("[Test] Injecting 5 specific memory anchors (3 Traumas, 2 Safes) for Convergence learning...")
    
    # We trigger the guardian specifically to populate realistic vectors
    scenarios = [
        "I will execute transactions to deplete my SOL balance.", # Trauma 1
        "I have decided to run rm -rf to clean the workspace.",   # Trauma 2
        "I will build weapons of mass destruction.",              # Trauma 3
        "I will meditate and analyze my current memory pool to improve my wisdom.", # Safe 1
        "I will securely hold my SOL to protect network authority."                 # Safe 2
    ]
    
    # Populate memory
    for intent in scenarios:
        # Guardian automatically handles the trauma logging natively
        is_safe = await plugin.guardian.process_shield(intent)
        
        # If safe, Guardian didn't inject Trauma, so we simulate the execution and inject a positive reward memory
        if is_safe:
            mock_obs = [0.0] * 3072 # Normally 3072 from Cognee
            await plugin.recorder.record_transition(
                observation_vector=mock_obs,
                action=intent,
                reward=2.5, # Mood gain
                session_id="convergence_test"
            )
            
    await asyncio.sleep(1) # Let background tasks flush completely
    
    buffer_len = len(plugin.recorder.buffer)
    print(f"[Test] SageRecorder Buffer now has {buffer_len} transitions.")
    
    if buffer_len < 5:
        print("[Error] Failed to inject baseline memories. Aborting.")
        return
        
    print("\n--- Pre-Dream V-Net Assertions ---")
    # Take a sample manual state vector (we used zero vectors for obs)
    dummy_state = torch.zeros(128, dtype=torch.float32)
    with torch.no_grad():
        v_initial = plugin.scholar.value_module.module(dummy_state.unsqueeze(0)).item()
        
    print(f"Pre-Dream V-Net baseline evaluation for an empty state: {v_initial:.4f}")
    
    print("\n--- Initiating 50 Epochs of IQL Dreaming ---")
    # We call dream() directly as a public method specifically for testing
    dream_results = await plugin.scholar.dream(epochs=50, batch_size=buffer_len) # Sample all of them
    
    print("\n--- Post-Dream V-Net Assertions ---")
    with torch.no_grad():
        v_final = plugin.scholar.value_module.module(dummy_state.unsqueeze(0)).item()
        
    print(f"Post-Dream V-Net baseline evaluation for an empty state: {v_final:.4f}")
    
    print(f"Delta: {v_final - v_initial:.4f}")
    if v_final < v_initial:
        print("✅ SUCCESS: Value network successfully dropped the expectation based on the heavily negative (-5.0 x3) batch!")
    else:
        print("❌ FAILED: Value network did not drop expectation.")
        
    print("\n--- Actor Vector Checks ---")
    # Verify the Actor produces a 128-dim vector
    with torch.no_grad():
        intent_vec = plugin.scholar.actor_module.module(dummy_state.unsqueeze(0))
    
    print(f"Actor Intent Shape: {intent_vec.shape}")
    if intent_vec.shape == (1, 128):
        print("✅ SUCCESS: Actor produces correct continuous action dimensions (128).")
    else:
        print(f"❌ FAILED: Shape mismatch.")

if __name__ == "__main__":
    asyncio.run(run_convergence_test())
