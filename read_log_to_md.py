
import os
import shutil

log_path = r"training_center\Khan_nefLhScHPb0.log"
out_path = "debug_output.md"

if os.path.exists(log_path):
    print(f"Copying {log_path} to {out_path}")
    headers = ["# Error Log Content"]
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# Error Log Content\n\n```text\n")
        f.write(content)
        f.write("\n```\n")
else:
    with open(out_path, 'w') as f:
        f.write("Log file not found.")
