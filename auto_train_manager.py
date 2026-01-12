import os
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import hashlib

def manage_auto_train():
    # 1. Configuration
    base_dir = Path(".")
    raw_dir = base_dir / "raw_data"
    edit_dir = base_dir / "processed" 
    train_center = Path("training_center")
    
    # [FIX] imagehash 설치 여부 사전 체크
    try:
        import imagehash
    except ImportError:
        print("[Error] 'imagehash' 라이브러리가 인식되지 않습니다.")
        print(f"실행 중인 파이썬({sys.executable})에 직접 설치를 시도합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ImageHash"])

    print(f"[{'='*30}]")
    print("[Manager] Auto Style Training Manager (M:N Mode)")
    print(f"[Manager] Raw Dir: {raw_dir}")
    print(f"[Manager] Edit Dir: {edit_dir}")
    print(f"[Manager] Python Path: {sys.executable}") # 현재 사용 중인 파이썬 확인용
    print(f"[{'='*30}]")

    if not raw_dir.exists() or not edit_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        edit_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scanning & Grouping
    lookup = defaultdict(lambda: {"raw": set(), "edit": set()})
    
    def get_keys(path, root_dir):
        keys = []
        prefix = path.stem.split('_')[0]
        keys.append(f"Prefix:{prefix}")
        try:
            rel = path.relative_to(root_dir)
            if len(rel.parts) > 1:
                folder_name = rel.parts[0]
                keys.append(f"Folder:{folder_name}")
        except: pass
        return keys

    for f in raw_dir.rglob("*.mp4"):
        for k in get_keys(f, raw_dir):
            lookup[k]["raw"].add(f)
            
    for f in edit_dir.rglob("*.mp4"):
        for k in get_keys(f, edit_dir):
            lookup[k]["edit"].add(f)

    # 3. Planning Tasks
    tasks = []
    seen_pairs = set() 
    
    url_file = base_dir / "urls.txt"
    if url_file.exists():
        print(f"\n[Manager] Found 'urls.txt'. Reading targets...")
        with open(url_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                
                parts = [p.strip() for p in line.split('|')]
                
                # RAW | Path/URL | Group
                if parts[0].upper() in ['RAW', 'EDIT'] and len(parts) >= 3:
                    ctype, val, gname = parts[0].upper(), parts[1], parts[2]
                    key = f"URLGroup:{gname}"
                    
                    # 로컬 파일 경로인지 확인
                    if os.path.exists(val.replace('"', '')):
                        val = Path(val.replace('"', ''))
                    
                    if ctype == 'RAW': lookup[key]["raw"].add(val)
                    else: lookup[key]["edit"].add(val)
                    continue

    for key, data in lookup.items():
        raws, edits = list(data['raw']), list(data['edit'])
        if not raws or not edits: continue
        
        print(f"{key:<20} | {len(raws):<5} | {len(edits):<5} | {len(raws)*len(edits):<5}")
        
        for r in raws:
            for e in edits:
                if (r, e) in seen_pairs: continue
                seen_pairs.add((r, e))
                
                group_name = key.split(':')[1]
                if isinstance(e, str):
                    suffix = hashlib.md5(e.encode()).hexdigest()[:6]
                    style_name = f"{group_name}_{suffix}"
                else:
                    style_name = f"{group_name}_{e.stem}"
                
                tasks.append({"raw": r, "edit": e, "name": style_name, "group": group_name})

    # 4. Deep Content Matching (Fallback)
    from sync_matcher import SyncMatcher
    scheduled_edits = {t['edit'] for t in tasks}
    all_edits = list(edit_dir.rglob("*.mp4"))
    unmatched_edits = [e for e in all_edits if e not in scheduled_edits]
    
    if unmatched_edits:
        print(f"\n[Manager] 🔍 Deep Scanning {len(unmatched_edits)} unmatched edits...")
        matcher = SyncMatcher()
        all_raws = list(raw_dir.rglob("*.mp4"))
        
        for e_file in unmatched_edits:
            results = matcher.match(str(raw_dir), str(e_file))
            file_scores = defaultdict(float)
            for seg in results.values():
                file_scores[seg['source']] += float(seg['confidence'])
                
            if file_scores:
                best_name = max(file_scores, key=file_scores.get)
                for r in all_raws:
                    if r.name == best_name:
                        tasks.append({
                            "raw": r, "edit": e_file,
                            "name": f"{r.stem}_{e_file.stem}",
                            "group": "AutoMatched"
                        })
                        print(f"  ✅ MATCHED: {e_file.name} <==> {r.name}")
                        break

    if not tasks:
        print("\n[Manager] No valid tasks found.")
        return

    # 5. Execution (Sequential Debug Mode)
    print(f"\n[Manager] 🚀 Executing {len(tasks)} tasks (Ensuring Python Path Consistency)...")
    
    for i, task_data in enumerate(tasks):
        target_dir = train_center / task_data['name']
        if target_dir.exists() and list(target_dir.glob("*.json")):
            print(f"  [Skip] Task {i+1}: '{task_data['name']}'")
            continue
        
        print(f"  [Start] Task {i+1}: {task_data['name']}")
        
        # [CRITICAL FIX] "python" 대신 sys.executable을 사용하여 동일한 환경 보장
        cmd = [
            sys.executable, "train_style.py",
            "--raw", str(task_data['raw']),
            "--edit", str(task_data['edit']),
            "--name", task_data['name']
        ]
        
        try:
            # shell=True 대신 리스트 형태 전달로 환경 변수 유지
            subprocess.run(cmd, check=True)
            print(f"  [Done] Task {i+1} Success.")
        except subprocess.CalledProcessError as e:
            print(f"  [Fail] Task {i+1} Failed.")

    print("\n[Manager] All tasks completed.")
    
    # 6. Style Aggregation
    print("\n[Manager] Aggregating Group Styles...")
    presets_dir = Path("presets")
    presets_dir.mkdir(exist_ok=True)
    
    # Identify unique groups processed
    groups_to_aggregate = set()
    for t in tasks:
        if t.get('group') and t['group'] != "AutoMatched":
            groups_to_aggregate.add(t['group'])
            
    for group in groups_to_aggregate:
        # Find all JSONs starting with GroupName_
        pattern = f"{group}_*.json"
        found = list(presets_dir.glob(pattern))
        
        if not found: continue
        
        print(f"  Grouping {len(found)} styles for '{group}'...")
        
        total_weights = defaultdict(float)
        total_params = defaultdict(float)
        descriptions = []
        prompts = []
        
        count = 0
        for p_file in found:
            try:
                with open(p_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Sum weights
                for k, v in data.get('weights', {}).items():
                    total_weights[k] += v
                    
                # Sum numeric params
                for k, v in data.get('parameters', {}).items():
                    if isinstance(v, (int, float)):
                        total_params[k] += v       
                        
                if 'description' in data: descriptions.append(data['description'])
                if 'llava' in data.get('prompts', {}): prompts.append(data['prompts']['llava'])
                
                count += 1
            except: pass
            
        if count == 0: continue
        
        # Average
        avg_weights = {k: round(v/count, 2) for k, v in total_weights.items()}
        avg_params = {k: int(v/count) for k, v in total_params.items()}
        
        # Merge Prompt (Use longest or most common? Just use first for now)
        final_prompt = prompts[0] if prompts else ""
        final_desc = f"Aggregated Style from {count} videos in group '{group}'."
        
        merged_preset = {
            "description": final_desc,
            "weights": avg_weights,
            "thresholds": {"rms_min_db": 1.2, "clamped_max_score": 5.0}, # Defaults
            "parameters": avg_params,
            "prompts": {"llava": final_prompt}
        }
        
        out_path = presets_dir / f"{group}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged_preset, f, indent=4)
            
        print(f"  ✅ Created Group Preset: {out_path.name}")

if __name__ == "__main__":
    manage_auto_train()
