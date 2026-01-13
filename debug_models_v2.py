from google import genai
from config import load_api_key, get_all_api_keys
import os

load_api_key()
keys = get_all_api_keys()
client = genai.Client(api_key=keys[0])

print("--- Available Models ---")
try:
    models = client.models.list()
    for m in models:
        print(f"Name: {m.name}, DisplayName: {m.display_name}, Supported: {m.supported_methods}")
except Exception as e:
    print(f"Error listing models: {e}")
