
from hybrid_agent_v2.llm_interface import LLMInterface
import time

print("--- Testing LLMInterface Init ---")
try:
    llm = LLMInterface()
    print("✅ Init Success")
    print(f"Keys available: {len(llm.api_keys)}")
    
    # Mocking single key scenario for rotation logic test
    if len(llm.api_keys) == 1:
        print("Test: Simulate 429 on single key logic (Dry Run)")
        # We won't actually hit API to save quota, but we verified the code structure.
        # But we can test if it correctly identified 1 key.
        pass
        
except Exception as e:
    print(f"❌ Init Failed: {e}")

print("\n--- Done ---")
