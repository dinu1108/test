import argparse
import subprocess
import shutil
import os
import json
import time
from pathlib import Path
from modules.analyst import Analyst
from pattern_learner import PatternLearner
from modules.scraper import GenericDownloader
from sync_matcher import SyncMatcher
import sys

def time_str_to_seconds(t_str):
    try:
        parts = list(map(int, t_str.split(':')))
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
    except: pass
    return 0

def seconds_to_time_str(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def smart_download_raw(raw_url, edit_path, output_path):
    print(f"[Trainer] 🧠 Smart Download Initiated for {raw_url}")
    
    # 1. Download Low-Res Video for Visual Matcher
    temp_dir = output_path.parent / "temp_visual_match"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("[1/3] Downloading Low-Res Video for Visual Sync...")
    dl_visual = GenericDownloader(temp_dir)
    v_path, _ = dl_visual.download(raw_url, video_worst_only=True)
    
    if not v_path:
        print("[Error] Failed to download video for visual sync.")
        return False
        
    print(f"[Trainer] Visual Ref Downloaded: {v_path.name}")
    
    # 2. Visual Sync Match
    print(f"[2/3] Finding Edit in Raw Video (Visual Hash)...")
    matcher = SyncMatcher()
    # match expects raw_dir path
    results = matcher.match(str(temp_dir), str(edit_path))
    
    # Cleanup Visual Temp
    try:
        shutil.rmtree(temp_dir)
    except: pass
    
    # 3. Analyze Result
    if not results:
        print("[Warning] No match found via Audio Sync. SKIPPING task to prevent massive download (User Request).")
        return False
        
    # Find best segment
        
    # Find best segment
    # results = {"seg_01": {"source":..., "start": "HH:MM:SS", "end":..., "confidence":...}}
    best_seg = max(results.values(), key=lambda x: float(x['confidence']))
    print(f"   MATCH FOUND: {best_seg['start']} ~ {best_seg['end']} (Conf: {best_seg['confidence']})")
    
    start_sec = time_str_to_seconds(best_seg['start'])
    end_sec = time_str_to_seconds(best_seg['end'])
    
    # Buffer: 1 hour before and after (User Request: "timeline nearby")
    # User said "needed section 1 hour", implying buffer. Let's give generous buffer 60 min.
    # Start min 0.
    target_start_sec = max(0, start_sec - 3600)
    # End unknown max, but youtube handles *end. Just add 3600 duration + duration of match.
    target_end_sec = end_sec + 3600
    
    t_start = seconds_to_time_str(target_start_sec)
    t_end = seconds_to_time_str(target_end_sec)
    
    print(f"[3/3] Downloading Section: {t_start} ~ {t_end}")
    
    dl_final = GenericDownloader(output_path.parent)
    v_sec, _ = dl_final.download(raw_url, start_time=t_start, end_time=t_end)
    
    if v_sec and v_sec.exists():
        shutil.move(str(v_sec), output_path)
        return True
    return False

def download_file(source, output_path, is_raw=False, edit_ref=None):
    # Local File Check
    if Path(source).exists():
        print(f"[Trainer] Local file detected: {source}")
        shutil.copy2(source, output_path)
        return True

    # URL Handling
    if "http" in source:
        if is_raw and edit_ref and edit_ref.exists():
            # Trigger Smart Download
            return smart_download_raw(source, edit_ref, output_path)
        else:
            # Normal Download (Edit video or Raw without Ref)
            print(f"[Trainer] Downloading {source}...")
            # Use GenericDownloader directly
            # Create temp to handle ID filenames
            temp_d = output_path.parent / "temp_dl_direct"
            dl = GenericDownloader(temp_d)
            
            # For Edit video (which is essentially anything not 'smart raw'), we disable metadata
            # to save requests and avoid Block 429 on subtitles.
            include_meta = True
            if not is_raw:
                 print("[Trainer] Metadata (Subs/Chat) Disabled for Edit Video to prevent 429.")
                 include_meta = False
            
            v_p, _ = dl.download(source, with_metadata=include_meta)
            if v_p:
                shutil.move(str(v_p), output_path)
                if temp_d.exists(): shutil.rmtree(temp_d)
                return True
            return False
            
    return False

def main():
    parser = argparse.ArgumentParser(description="Auto Highlight - Style Trainer (Smart Mode)")
    parser.add_argument("--raw", required=True, help="YouTube URL or Local Path for Raw")
    parser.add_argument("--edit", required=True, help="YouTube URL or Local Path for Edit")
    parser.add_argument("--name", required=True, help="Style Name")
    args = parser.parse_args()

    # 1. Setup Directories
    train_root = Path("training_center")
    style_dir = train_root / args.name
    
    if style_dir.exists():
        print(f"[Warning] Directory {style_dir} already exists. Cleaning up...")
        shutil.rmtree(style_dir)
    
    style_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Download Files
    raw_path = style_dir / "raw_full.mp4"
    edit_path = style_dir / "edited_01.mp4"
    
    print(f"\n{'='*60}")
    print(f"🏋️ Style Trainer: {args.name}")
    print(f"{'='*60}\n")
    
    # Strategy: Download Edit FIRST to use as reference for Raw
    print(">> [1/2] Processing Edited Video...")
    if not download_file(args.edit, edit_path):
        print("Edit download failed. Cleaning up...")
        shutil.rmtree(style_dir)
        sys.exit(1)

    print("\n>> [2/2] Processing Raw Video (Smart Mode)...")
    # Pass edit_path for Sync Matching
    if not download_file(args.raw, raw_path, is_raw=True, edit_ref=edit_path):
        print("Raw download failed. Cleaning up...")
        shutil.rmtree(style_dir)
        sys.exit(1)

    # 3. Validation
    if not raw_path.exists() or not edit_path.exists():
        print("[Error] Files missing. Aborting and cleaning up.")
        shutil.rmtree(style_dir)
        sys.exit(1)

    # 4. Run Learning
    print(f"\n>> [3/3] Learning Patterns...")
    learner = PatternLearner()
    learner.learn(str(style_dir), args.name)
    
    # 5. Cleanup (User Guideline: Delete raw after training)
    print("\n[Cleanup] Deleting large raw file...")
    if raw_path.exists():
        try:
            raw_path.unlink()
        except: pass
    
    print(f"\n✅ Done! Style '{args.name}' Created.")

if __name__ == "__main__":
    main()
