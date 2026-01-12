import os

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (one directory up from this config file's package)
# config.py is in hybrid_agent_v2/, so root is one level up.
# However, user requested strictly looking for root of "main_v2.py".
# Assuming current working directory is project root or we traverse up.

def load_api_key():
    # Strategy: Find .env in the parent directory of this file's directory
    # c:\...\auto_highlight_extractor\hybrid_agent_v2\config.py
    # -> c:\...\auto_highlight_extractor\.env
    
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[Config] Loaded .env from: {env_path}")
    else:
        # Fallback: Try current working directory
        cwd_env = Path(os.getcwd()) / ".env"
        if cwd_env.exists():
            load_dotenv(dotenv_path=cwd_env)
            print(f"[Config] Loaded .env from CWD: {cwd_env}")
        else:
            print("[Config] ⚠️ Warning: .env file not found.")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("[Config] ❌ GOOGLE_API_KEY not found in environment variables.")
