import argparse
import json
import os
import subprocess
import numpy as np
import librosa
import scipy.signal
from pathlib import Path
from tqdm import tqdm
import requests
import base64
import statistics
import glob

class PatternLearner:
    def __init__(self):
        self.tmp_dir = Path("learner_tmp")
        self.tmp_dir.mkdir(exist_ok=True)
        self.sr = 8000 # Low SR for faster correlation
        self.ana_sr = 16000 # Analysis SR
        self.MAX_DIFF_SEC = 5.0 # Allowed drift tolerance

    def extract_audio(self, video_path, output_path):
        if not output_path.exists():
            print(f"[Learner] Extracting audio from {video_path.name}...")
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", str(self.sr), "-ac", "1", str(output_path)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

    def find_intervals(self, raw_wav, edited_wav):
        print("[Learner] Loading audio for synchronization...")
        y_raw, _ = librosa.load(str(raw_wav), sr=self.sr)
        y_edit, _ = librosa.load(str(edited_wav), sr=self.sr)

        chunk_dur = 5.0
        chunk_len = int(chunk_dur * self.sr)
        intervals = []
        
        last_raw_idx = -1
        current_len = 0
        current_start_raw = -1
        
        total_chunks = len(y_edit) // chunk_len
        
        print(f"[Learner] Syncing {total_chunks} blocks (Optimized Sequential)...")
        
        # Normalize
        y_raw_norm = (y_raw - np.mean(y_raw)) / (np.std(y_raw) + 1e-6)
        
        # Robust Sequential Sync
        # We process manually to avoid Windows Multiprocessing issues.
        # Scipy's FFT is efficient enough with a smart search window.
        
        for i in tqdm(range(0, total_chunks, 2), desc="Syncing"):
            s_e = i * chunk_len
            e_e = s_e + chunk_len
            chunk = y_edit[s_e:e_e]
            chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-6)
            
            # Dynamic Search Window
            if last_raw_idx != -1:
                pred = last_raw_idx + (chunk_len * 2)
                win = 30 * self.sr # 30s Window (Widened for robustness)
                search_start = max(0, pred - win)
                search_end = min(len(y_raw), pred + win)
                
                y_search = y_raw_norm[search_start:search_end]
                if len(y_search) < len(chunk): continue
                
                cc = scipy.signal.correlate(y_search, chunk, mode='valid', method='fft')
                local_max = np.argmax(cc)
                global_idx = search_start + local_max
                
            else:
                # Global Search (Only for first lock)
                # This is the heavy part.
                cc = scipy.signal.correlate(y_raw_norm, chunk, mode='valid', method='fft')
                local_max = np.argmax(cc)
                global_idx = local_max
            
            # Check continuity with tolerance
            expected_pos = last_raw_idx + (chunk_len * 2) if last_raw_idx != -1 else -1
            
            # Heuristic: Check if matches prediction
            is_contiguous = False
            if last_raw_idx != -1:
                diff = abs(global_idx - expected_pos)
                if diff < (self.sr * self.MAX_DIFF_SEC):
                    is_contiguous = True
            
            if is_contiguous:
                current_len += (chunk_len * 2)
                last_raw_idx = global_idx
            else:
                # Segment break
                if current_start_raw != -1:
                     intervals.append((current_start_raw/self.sr, (current_start_raw + current_len)/self.sr))
                current_start_raw = global_idx
                current_len = chunk_len
                last_raw_idx = global_idx

        if current_start_raw != -1:
             intervals.append((current_start_raw/self.sr, (current_start_raw + current_len)/self.sr))
             
        # Merge close intervals
        merged = []
        if intervals:
            intervals.sort()
            curr_s, curr_e = intervals[0]
            for next_s, next_e in intervals[1:]:
                # Merge if gap < 10s
                if next_s <= curr_e + 10.0:
                    curr_e = max(curr_e, next_e)
                else:
                    merged.append((curr_s, curr_e))
                    curr_s, curr_e = next_s, next_e
            merged.append((curr_s, curr_e))
            
        print(f"[Learner] Identified {len(merged)} cut segments.")
        return merged

    def profile_features(self, raw_wav_full, intervals):
        print("[Learner] Profiling Audio Features (RMS, Slope, Pitch)...")
        y, sr = librosa.load(str(raw_wav_full), sr=self.ana_sr)
        
        stats = {"rms_max": [], "rms_slope": [], "pitch_var": [], "preroll": [], "postroll": []}
        
        for s, e in tqdm(intervals, desc="Analyzing"):
            s_idx, e_idx = int(s*sr), int(e*sr)
            if e_idx > len(y): e_idx = len(y)
            if e_idx - s_idx < sr: continue
            
            seg = y[s_idx:e_idx]
            
            rms = librosa.feature.rms(y=seg, hop_length=2048)[0]
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
            
            peak_idx = np.argmax(rms_norm)
            peak_time = librosa.frames_to_time(peak_idx, sr=sr, hop_length=2048)
            
            slopes = np.diff(rms_norm)
            win = 10
            p_start = max(0, peak_idx - win)
            local_slope = np.mean(np.maximum(0, slopes[p_start:peak_idx])) if peak_idx > 0 else 0
            
            zcr = librosa.feature.zero_crossing_rate(y=seg)[0]
            zcr_var = np.var(zcr)
            
            stats["rms_max"].append(np.max(rms_norm))
            stats["rms_slope"].append(local_slope)
            stats["pitch_var"].append(zcr_var)
            stats["preroll"].append(peak_time)
            stats["postroll"].append((e-s) - peak_time)
            
        return stats

    def llava_persona(self, raw_video, intervals):
        if not intervals: return "Standard Highlight"
        print("[Learner] LLaVA Extracting Persona...")
        longest = max(intervals, key=lambda x: x[1]-x[0])
        mid = (longest[0] + longest[1]) / 2
        
        cmd = ["ffmpeg", "-y", "-ss", str(mid), "-i", str(raw_video), "-vframes", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
        try:
            r = subprocess.run(cmd, capture_output=True)
            if len(r.stdout) > 0:
                b64 = base64.b64encode(r.stdout).decode('utf-8')
                p = "Analyze this livestream frame. Describe the streamer's reaction and engagement style in one sentence."
                res = requests.post("http://localhost:11434/api/generate", json={"model":"llava", "prompt":p, "images":[b64], "stream":False})
                return res.json().get('response', 'Engaging Moment').strip()
        except: return "General Gaming Highlight"

    def learn(self, center_dir, output_name):
        center = Path(center_dir)
        raws = list(center.glob("raw_full.mp4"))
        edits = list(center.glob("edited_*.mp4"))
        
        if not raws or not edits:
            print("[Error] Missing raw_full.mp4 or edited_*.mp4 in directory.")
            return

        raw_p = raws[0]
        edit_p = edits[0] 
        
        print(f"=== Learning Style from: {center.name} ===")
        print(f"Raw: {raw_p.name}")
        print(f"Target: {edit_p.name}")
        
        raw_wav = self.extract_audio(raw_p, self.tmp_dir / "raw.wav")
        edit_wav = self.extract_audio(edit_p, self.tmp_dir / "edit.wav")
        
        intervals = self.find_intervals(raw_wav, edit_wav)
        if not intervals:
            print("Failed to sync.")
            return
            
        stats = self.profile_features(raw_p, intervals) 
        persona = self.llava_persona(raw_p, intervals)
        
        avg_slope = statistics.mean(stats['rms_slope']) if stats['rms_slope'] else 0
        avg_pitch_var = statistics.mean(stats['pitch_var']) if stats['pitch_var'] else 0
        
        w_rms = 0.3
        w_slope = 0.2
        w_zcr = 0.1
        
        if avg_slope > 0.05: w_slope = 0.4; w_rms = 0.2
        if avg_pitch_var > 0.02: w_zcr = 0.3
        
        preset = {
            "description": f"Learned from {center.name}. {persona}",
            "weights": {
                "audio_rms": w_rms,
                "audio_slope": w_slope,
                "audio_zcr": w_zcr,
                "chat_velocity": 0.2,
                "visual_clip": 0.2
            },
            "thresholds": {
                "rms_min_db": 1.2,
                "clamped_max_score": 5.0
            },
            "parameters": {
                "stage1_top_k": 300,
                "merge_gap_seconds": int(statistics.mean(stats['preroll']) * 2) + 60
            },
            "prompts": {
                "llava": f"This is a highlight similar to: '{persona}'. Is the content engaging in the same way? Answer YES or NO."
            }
        }
        
        out = Path(f"presets/{output_name}.json")
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(preset, f, indent=4)
            
        print(f"succesfully saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("center", help="Path to training_center/[style] folder")
    parser.add_argument("--name", help="Output style name (default: folder name)")
    args = parser.parse_args()
    
    style_name = args.name if args.name else Path(args.center).name
    
    learner = PatternLearner()
    learner.learn(args.center, style_name)
