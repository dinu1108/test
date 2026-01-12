import argparse
import json
import os
import cv2
import imagehash
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict

class SyncMatcher:
    def __init__(self):
        self.hash_size = 8
        self.threshold = 12 # Hamming Distance threshold (0-64)
        
    def _get_phash(self, frame):
        """Compute pHash for a single frame."""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return imagehash.phash(pil_img, hash_size=self.hash_size)
        except Exception as e:
            return None

    def _extract_fingerprints(self, edit_path):
        """Extract visual fingerprints from 3m, 7m, 12m marks."""
        fingerprints = []
        cap = cv2.VideoCapture(str(edit_path))
        if not cap.isOpened():
            print(f"[Matcher] Error: Cannot open edit video {edit_path}")
            return []
            
        # Target timestamps (seconds): 3m, 7m, 12m
        # If video is shorter, use 20%, 50%, 80% marks
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0
        
        targets = [180.0, 420.0, 720.0] # 3m, 7m, 12m
        
        # Adjust for short videos
        if duration < 800: # Less than ~13 min
            targets = [duration * 0.2, duration * 0.5, duration * 0.8]
            
        print(f"[Matcher] Extracting Visual Fingerprints at {targets}s...")
        
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                h = self._get_phash(frame)
                if h:
                    fingerprints.append({
                        "time": t,
                        "hash": h
                    })
        cap.release()
        return fingerprints

    def match(self, raw_dir, edit_file):
        """
        Scan raw video using Fast Scan (10s jumps).
        Returns best match segment.
        """
        # 1. Edit Fingerprinting
        fps = self._extract_fingerprints(edit_file)
        if not fps:
            print("[Matcher] Failed to extract fingerprints from edit.")
            return {}
            
        print(f"[Matcher] ✅ Extracted {len(fps)} visual fingerprints.")
        
        # 2. Raw Scanning
        raw_path = Path(raw_dir)
        # Scan folder for video files
        supported_exts = {'.mp4', '.m4a', '.ts', '.mkv', '.webm'}
        # Note: m4a usually has no video, but user might pass m4a if they forgot. 
        # But we need video. We'll try to find video files.
        
        raw_files = [p for p in raw_path.iterdir() if p.suffix.lower() in supported_exts]
        
        for r_file in raw_files:
            print(f"  -> Visual Scanning {r_file.name}...")
            cap = cv2.VideoCapture(str(r_file))
            if not cap.isOpened(): continue
            
            # Optimization: Fast Scan (jump 10s)
            step_sec = 10.0 
            step_ms = step_sec * 1000
            
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps_rate = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps_rate if fps_rate > 0 else 0
            
            current_ms = 0
            
            # Using loop with set(POS_MSEC) is slower than reading, 
            # but for 10s jumps, set() is necessary.
            # To optimize seek, we just grab frames.
            
            print(f"     Scanning {duration/3600:.1f}h content...")
            
            best_match = None
            min_dist = 999
            
            while current_ms < duration * 1000:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
                ret, frame = cap.read()
                if not ret: break
                
                # Check against ALL fingerprints
                # Strategy: If ANY fingerprint matches, we check others to confirm?
                # Simpler: Just find single strong match for the FIRST fingerprint (or any).
                
                curr_hash = self._get_phash(frame)
                if not curr_hash: 
                    current_ms += step_ms
                    continue
                
                # Compare
                for fp in fps:
                    dist = curr_hash - fp['hash']
                    
                    if dist <= self.threshold:
                        print(f"   [Debug] Visual Match? Dist={dist} at Raw={current_ms/1000:.1f}s vs Edit={fp['time']:.1f}s")
                        
                        # Verify with other FPS if possible?
                        # For now, Early Exit on first good match
                        if dist < min_dist:
                            min_dist = dist
                            # Calculate Raw Start Time
                            # Raw_Current = Edit_Time + Start_Offset
                            # Start_Offset = Raw_Current - Edit_Time
                            raw_start = (current_ms / 1000.0) - fp['time']
                            best_match = {
                                "source": r_file.name,
                                "start": raw_start,
                                "confidence": 1.0 - (dist / 64.0) # Approx confidence
                            }
                            
                        if dist <= 8: # Super strong match
                            print(f"   🚀 Strong Match Found! Stopping scan.")
                            cap.release()
                            return self._format_result(best_match, duration)
                            
                current_ms += step_ms
                
                if int(current_ms/1000) % 600 == 0:
                    print(f"     ... scanned {current_ms/1000/60:.0f} min", end='\r')
            
            cap.release()
            if best_match:
                return self._format_result(best_match, duration)
                
        return {}

    def _format_result(self, match, raw_total_dur):
        s_sec = max(0, match['start'])
        e_sec = min(raw_total_dur, s_sec + 3600) # Default 1h segment? Or just end?
        # User wants section download. Let's give 1 hour + buffer.
        
        return {
             "segment_01": {
                "source": match['source'],
                "start": self._sec_to_time(s_sec),
                "end": self._sec_to_time(e_sec),
                "confidence": f"{match['confidence']:.2f}"
            }
        }

    def _sec_to_time(self, sec):
        h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--edit", required=True)
    args = parser.parse_args()
    matcher = SyncMatcher()
    print(matcher.match(args.raw_dir, args.edit))
