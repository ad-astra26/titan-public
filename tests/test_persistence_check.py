import asyncio
import sys
import os
import time

# Ensure Titan plugin is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin import init_plugin
from titan_plugin.core.sage.recorder import SageRecorder

async def run_interaction_loop(plugin):
    """Simulates 5 interactions through the OpenClaw execution hook."""
    print("Initiating 5 simulated interactions...")
    for i in range(1, 6):
        prompt = f"User interaction {i}: Hello Titan."
        response = f"Agent response {i}: Greetings."
        
        # This will trigger the recording in the background
        await plugin.post_resolution_hook(prompt, response)
        
        # Briefly wait for the async background task to complete recording
        await asyncio.sleep(0.5)
        print(f"  Interaction {i} completed and logged.")

def verify_persistence():
    """Restarts the recorder and samples the buffer to check for persistence."""
    print("\n--- The Persistence Check ---")
    print("Stopping script and restarting recorder instance...")
    
    # Reload from disk
    config = {
        "sage_memory": {
            "buffer_size": 1_000_000,
            "storage_path": "./data/sage_memory/"
        }
    }
    
    reloaded_recorder = SageRecorder(config)
    
    if reloaded_recorder.buffer is None:
        print("❌ ReplayBuffer failed to initialize. Make sure torch and torchrl are installed.")
        return
        
    length = len(reloaded_recorder.buffer)
    print(f"Buffer length upon reload: {length}")
    
    if length < 5:
        print("❌ Buffer did not persist at least 5 records.")
        return
    
    try:
        sample = reloaded_recorder.buffer.sample(5)
        print("\n✅ Verification Successful: Sample of 5 drawn from LazyMemmapStorage.")
        
        reward = sample["reward"]
        trauma_is_violation = sample["trauma"]["is_violation"]
        
        print(f"Sampled Rewards: {reward.squeeze().tolist()}")
        print(f"Sampled 'is_violation' Traumas: {trauma_is_violation.squeeze().tolist()}")
        print("\nThe memory has persisted across instances!")
    except Exception as e:
        print(f"❌ Verification failed during sampling: {e}")

async def main():
    print("🧪 Starting The Persistence Check Protocol...")
    
    # Mock config
    config = {"WALLET_KEYPAIR_PATH": "./authority.json"}
    
    # 1. Initialize Main Titan Execution Loop (via plugin init)
    plugin = init_plugin(config)
    
    # 2. Run interactions & Stop
    await run_interaction_loop(plugin)
    
    # Let async memory dump cleanly if needed
    await asyncio.sleep(1) 
    
    # 3. Verify
    verify_persistence()

if __name__ == "__main__":
    asyncio.run(main())
