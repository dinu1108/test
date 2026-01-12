
import os
import glob
from pathlib import Path

log_dir = Path("training_center")
logs = list(log_dir.glob("*.log"))

out_path = "debug_log_content.md"

if logs:
    target_log = logs[0]
    print(f"Reading first log found: {target_log}")
    
    with open(target_log, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Log Content: {target_log.name}\n\n```text\n")
        f.write(content)
        f.write("\n```\n")
    print(f"Log content written to {out_path}")
else:
    print("No log files found in training_center.")
