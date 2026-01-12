
import os
log_path = r"training_center\Khan_nefLhScHPb0.log"
if os.path.exists(log_path):
    print(f"Reading {log_path}:")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read())
else:
    print("Log file not found.")

import pkg_resources
print("\nInstalled Packages:")
for p in pkg_resources.working_set:
    print(f"{p.project_name}=={p.version}")
