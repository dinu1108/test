import google.generativeai as genai
from config import load_api_key
import os

load_api_key()
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Listing available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
