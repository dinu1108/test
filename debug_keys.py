
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

print("--- Key Check ---")
for key in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3"]:
    val = os.getenv(key)
    if val:
        print(f"{key}: Found (Ends with {val[-4:]})")
    else:
        print(f"{key}: Not Found")

print("\n--- Model Check ---")
key1 = os.getenv("GOOGLE_API_KEY")
if key1:
    try:
        client = genai.Client(api_key=key1)
        # Try listing models to see what's available
        # print("List models...")
        # for m in client.models.list(config={"page_size": 5}):
        #     print(f" - {m.name}")
        
        print("Testing gemini-1.5-flash...")
        resp = client.models.generate_content(model="gemini-1.5-flash", contents="Hello")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")
