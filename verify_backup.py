
import os
from pathlib import Path

backup_dir = Path("offline_model_backup")
files = [
    "auto_train_manager.py",
    "train_style.py",
    "pattern_learner.py",
    "hybrid_agent_v2/knowledge_base.py"
]

log = ["--- Backup Verification ---"]
if not backup_dir.exists():
    log.append("❌ Backup directory NOT FOUND")
else:
    log.append(f"📂 Backup Dir: {backup_dir.absolute()}")
    contents = list(backup_dir.glob("*"))
    log.append(f"📄 Files in Backup Dir ({len(contents)}):")
    for f in contents:
        log.append(f" - {f.name} ({f.stat().st_size} bytes)")

    log.append("\n--- Source Check ---")
    for fname in files:
        src = Path(fname)
        src_exists = src.exists()
        log.append(f"Source '{fname}': {'✅ Found' if src_exists else '❌ MISSING'}")

with open("backup_verification.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("Verification complete.")
