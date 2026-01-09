import argparse
import sys
import os
import json
import traceback
from pathlib import Path
from modules.analyst import Analyst
from modules.editor import Editor

def load_preset(style_name):
    """Load preset from JSON file."""
    preset_path = Path(f"presets/{style_name}.json")
    if not preset_path.exists():
        print(f"[Main] Warning: Preset '{style_name}' not found. Using internal defaults.")
        # Return a robust default if file missing
        return {
            "weights": {"audio_rms": 0.3, "audio_slope": 0.2, "chat_velocity": 0.3},
            "parameters": {"stage1_top_k": 300, "stage2_top_k": 80}
        }
    
    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            print(f"[Main] Loading Preset: {style_name.upper()}")
            return json.load(f)
    except Exception as e:
        print(f"[Main] Error loading preset: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Auto Highlight Extractor (Hybrid AI)")
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--style", default="energetic", help="Editing Style: energetic (default), talkative, skillful")
    args = parser.parse_args()

    # 1. Load Preset
    config = load_preset(args.style)

    # 2. Setup Modules
    analyst = Analyst(config=config)
    editor = Editor()

    print(f"\n{'='*60}")
    print(f"🚀 Auto Highlight Extractor (Style: {args.style.upper()})")
    print(f"{'='*60}\n")

    try:
        # Step 1: Download
        print(">> [1/4] Downloading Video & Chat...")
        video_path, audio_path, chat_path = analyst.download_video(args.url)
        if not video_path:
            print("[Error] Download failed.")
            return

        # Step 2: Analyze
        print("\n>> [2/4] Analyzing Content (Hybrid AI)...")
        highlights = analyst.get_highlights(video_path, chat_path)
        
        if not highlights:
            print("[Info] No highlights found matching criteria.")
            return

        # Step 3: Edit
        print(f"\n>> [3/4] Editing {len(highlights)} Sequences...")
        editor.create_full_recap(video_path, highlights, style_name=args.style)

        print("\n>> [4/4] Process Complete! Check 'clips' folder.")

    except KeyboardInterrupt:
        print("\n[Aborted] User stopped the process.")
    except Exception as e:
        print(f"\n[Critical Error] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
