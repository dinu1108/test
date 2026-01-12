import os
from pathlib import Path

def init():
    # Use explicit absolute path based on CWD
    base = Path(os.getcwd())
    raw = base / "raw_data"
    edit = base / "edit_data"
    
    print(f"[Init] Creating {base}...")
    base.mkdir(exist_ok=True)
    
    print(f"[Init] Creating {raw}...")
    raw.mkdir(exist_ok=True)
    
    print(f"[Init] Creating {edit}...")
    edit.mkdir(exist_ok=True)
    
    # Validation
    if raw.exists() and edit.exists():
        print("SUCCESS: Directories created.")
    else:
        print("FAILURE: Directories not found.")

if __name__ == "__main__":
    init()
