
import shutil
import os
from pathlib import Path

backup_dir = Path("offline_model_backup")
backup_dir.mkdir(exist_ok=True)

to_backup = [
    "auto_train_manager.py",
    "train_style.py",
    "pattern_learner.py",
    "hybrid_agent_v2/knowledge_base.py"
]

print(f"📂 Backing up to: {backup_dir.absolute()}")

for file_path in to_backup:
    src = Path(file_path)
    if src.exists():
        dst = backup_dir / src.name
        shutil.copy2(src, dst)
        print(f"  ✅ Copied {src.name}")
    else:
        print(f"  ⚠️ File not found: {file_path}")

print("Done.")
