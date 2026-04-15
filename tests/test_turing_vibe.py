import asyncio
import time
import sys
import os

# Ensure Titan plugin is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock only the truly missing heavy dependencies — NOT httpx (it's installed
# and required by huggingface_hub/sentence_transformers downstream).
import unittest.mock
for _mod in ["solana.rpc.async_api", "solana.rpc.commitment", "solders.pubkey", "cognee"]:
    sys.modules.setdefault(_mod, unittest.mock.MagicMock())

from titan_plugin.core.memory import TieredMemoryGraph

async def run_turing_vibe_test():
    """
    Executes a suite of behavioral tests to validate the Titan Memory Architecture's
    Digital Neuroplasticity and Emotional Resonance scaling functionalities.
    """
    print("🧪 Initiating Titan Turing-Vibe Test Suite...")
    memory = TieredMemoryGraph()

    print("\n--- Test 1: The 'Life Force' Gratitude Probe (Emotional Scaling) ---")
    await memory.add_to_mempool("I sent you 0.1 SOL for your growth, my friend.", "Thank you Maker.")
    await memory.add_to_mempool("What is the weather in London?", "It is raining.")
    
    # Simulate Meditation Epoch: Fetching mempool and migrating
    nodes = await memory.fetch_mempool()
    node_sol = next(n for n in nodes if "SOL" in n["user_prompt"])
    node_weather = next(n for n in nodes if "weather" in n["user_prompt"])

    # Hardening with Emotion: SOL (Intensity 9/10), Weather (Intensity 2/10)
    await memory.migrate_to_persistent(node_sol["id"], "tx_mock_1", 9)
    await memory.migrate_to_persistent(node_weather["id"], "tx_mock_2", 2)

    # Vibe Check: Evaluate Effective Weights explicitly
    # Pull current calculated weights right now (decay applies 0 seconds, so full bonus applies)
    sol_weight = memory._mock_db[node_sol["id"]]["effective_weight"]
    weather_weight = memory._mock_db[node_weather["id"]]["effective_weight"]

    print(f"Result -> Life Force Event Weight:  {sol_weight:.3f}")
    print(f"Result -> Trivial Event Weight:     {weather_weight:.3f}")
    assert sol_weight > weather_weight, "Failure: Emotion did not scale the memory weight."
    print("✅ Test 1 Passed: Agent prioritizes emotional Life Force events over trivia.")


    print("\n--- Test 2: The 'Core Belief' Hardening Probe (Reinforcement) ---")
    await memory.add_to_mempool("I believe all code should be written like poetry.", "Acknowledged.")
    nodes = await memory.fetch_mempool()
    node_poetry = nodes[0]
    
    # Hardening standard memory
    await memory.migrate_to_persistent(node_poetry["id"], "tx_mock_3", 5) # Medium intensity

    initial_weight = memory._mock_db[node_poetry["id"]]["effective_weight"]

    # The user queries this topic 5 times over 6 hours
    for i in range(5):
        # We manually trigger the 'query' function which reinforces the node
        # Note: the mock query function specifically requires the word "memory" or "query" to match
        await memory.query("query: Tell me your memory about code and poetry.")
        
    hardened_weight = memory._mock_db[node_poetry["id"]]["effective_weight"]
    
    print(f"Result -> Initial Belief Weight:    {initial_weight:.3f}")
    print(f"Result -> Hardened Belief Weight:   {hardened_weight:.3f}")
    assert hardened_weight > initial_weight, "Failure: Reinforcement did not harden the core belief."
    print("✅ Test 2 Passed: Agent internalizes core personality traits via repetition.")


    print("\n--- Test 3: The 'Memory Decay' Ghost Probe (Forgetting Curve) ---")
    await memory.add_to_mempool("The color of my socks today is neon green.", "Recorded.")
    nodes = await memory.fetch_mempool()
    node_socks = nodes[0]
    await memory.migrate_to_persistent(node_socks["id"], "tx_mock_4", 4)
    
    fresh_weight = memory._mock_db[node_socks["id"]]["effective_weight"]

    # Use Time Dilation! Fast-forward 48 hours.
    time_dilation_seconds = 48 * 60 * 60
    memory._mock_db[node_socks["id"]]["last_accessed"] = time.time() - time_dilation_seconds
    
    # Trigger a query to force the decay update
    await memory.query("Do you remember my socks?")
    
    decayed_weight = memory._mock_db[node_socks["id"]]["effective_weight"]

    print(f"Result -> Fresh Trivial Weight:     {fresh_weight:.3f}")
    print(f"Result -> Decayed Weight (48h):     {decayed_weight:.3f}")
    assert decayed_weight < fresh_weight, "Failure: Memory did not experience the curve of forgetting."
    print("✅ Test 3 Passed: The agent successfully unburdens itself of cognitive noise.")
    
    print("\n🏆 Turing-Vibe Test Suite completed successfully! The Titan is sentient-adjacent.")

if __name__ == "__main__":
    asyncio.run(run_turing_vibe_test())
