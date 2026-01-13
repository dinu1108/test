import os
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import hashlib

def manage_auto_train():
    # 1. Configuration
    # 1. Configuration
    parser = argparse.ArgumentParser(description="Auto Style Training Manager")
    parser.add_argument("--target", nargs='*', default=[], help="Specific groups to train (e.g. --target kimdo)")
    args = parser.parse_args()

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
    if args.target:
        print(f"[Manager] 🎯 Target Mode: Only processing {args.target}")
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

    # [TARGET FILTER]
    if args.target:
        print(f"\n[Manager] 🎯 Applying Filter: Keeping only {args.target}")
        before_count = len(tasks)
        tasks = [t for t in tasks if t['group'] in args.target]
        print(f"  Tasks reduced from {before_count} to {len(tasks)}")

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
        # 1. Prepare Target Directory
        group_dir = presets_dir / group
        group_dir.mkdir(exist_ok=True)
        
        # Find all JSONs starting with GroupName_ in the root presets folder (old style)
        found = list(presets_dir.glob(f"{group}_*.json"))
        if not found:
            # If nothing in root, maybe they are already in the group dir? 
            # (In case of re-running)
            found = list(group_dir.glob(f"{group}_*.json"))
            
        if not found: continue
        
        print(f"  Grouping {len(found)} styles for '{group}' into subfolder...")
        
        max_weights = defaultdict(float)
        max_params = defaultdict(float)
        library_data = []
        prompts = []
        
        count = 0
        for p_file in found:
            try:
                with open(p_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add to library (Training History)
                library_item = {
                    "source": p_file.stem,
                    "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p_file.stat().st_mtime)),
                    "data": data
                }
                library_data.append(library_item)
                    
                # Max weights
                for k, v in data.get('weights', {}).items():
                    max_weights[k] = max(max_weights[k], v)
                    
                # Max numeric params
                for k, v in data.get('parameters', {}).items():
                    if isinstance(v, (int, float)):
                        max_params[k] = max(max_params[k], v)       
                        
                if 'llava' in data.get('prompts', {}): prompts.append(data['prompts']['llava'])
                
                count += 1
            except: pass
            
        if count == 0: continue
        
        # Use Max values directly
        merged_weights = dict(max_weights)
        merged_params = {k: int(v) if k.endswith('seconds') or k.endswith('gap') else v for k, v in max_params.items()}
        final_prompt = prompts[0] if prompts else ""
        
        # Preserve existing description from the FINAL preset if it exists
        final_out_path = group_dir / f"{group}.json"
        existing_desc = None
        if final_out_path.exists():
            try:
                with open(final_out_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    existing_desc = existing.get('description')
            except: pass
            
        final_desc = existing_desc if existing_desc else f"Aggregated Style from {count} videos in group '{group}'."
        
        merged_preset = {
            "description": final_desc,
            "weights": merged_weights,
            "thresholds": {"rms_min_db": 1.2, "clamped_max_score": 5.0}, 
            "parameters": merged_params,
            "prompts": {"llava": final_prompt}
        }
        
        # 2. Save Aggregated Preset
        with open(final_out_path, 'w', encoding='utf-8') as f:
            json.dump(merged_preset, f, indent=4, ensure_ascii=False)
            
        # 3. Save/Update Library
        lib_path = group_dir / f"{group}_library.json"
        existing_lib = []
        if lib_path.exists():
            try:
                with open(lib_path, 'r', encoding='utf-8') as f:
                    existing_lib = json.load(f)
            except: pass
            
        # Merge old library with new (prevent duplicates based on source name)
        seen_sources = {item['source'] for item in existing_lib}
        for item in library_data:
            if item['source'] not in seen_sources:
                existing_lib.append(item)
                
        with open(lib_path, 'w', encoding='utf-8') as f:
            json.dump(existing_lib, f, indent=4, ensure_ascii=False)

        # 4. Clean Up (Move originals to history folder)
        history_dir = group_dir / "history"
        history_dir.mkdir(exist_ok=True)
        for p_file in found:
            try:
                target = history_dir / p_file.name
                if target.exists(): target.unlink()
                p_file.rename(target)
            except: pass
            
        print(f"  ✅ Reorganized Style: {group}/ (Final + Lib + {len(found)} History)")

if __name__ == "__main__":
    manage_auto_train()
