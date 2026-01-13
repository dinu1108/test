import subprocess
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import math

class VisualScout:
    def __init__(self, temp_dir="temp_visual"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def analyze_video(self, video_path, fps=0.5):
        """
        Extract thumbnails at a low FPS and calculate visual energy (Entropy + Motion).
        Caching results to avoid re-analyzing long videos.
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        cache_path = self.temp_dir / f"{video_id}_visual_cache.json"

        # [Check Cache]
        if cache_path.exists():
            print(f"🕵️ [VisualScout] Found cached visual data for '{video_id}'. Skipping extraction.")
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        "times": np.array(data["times"]),
                        "entropy": np.array(data["entropy"]),
                        "motion": np.array(data["motion"]),
                        "visual_energy": np.array(data["visual_energy"])
                    }
            except:
                print("⚠️ [VisualScout] Cache corrupted. Re-analyzing...")

        # 0. Get duration for adaptive FPS
        def _get_duration(v_path):
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(v_path)]
            try:
                return float(subprocess.check_output(cmd).decode().strip())
            except: return 0.0

        duration = _get_duration(video_path)
        # Adaptive FPS: 12h @ 0.5fps is too much. 
        # If > 2h, use 0.1fps (1 frame/10s). If > 6h, use 0.05fps (1 frame/20s).
        if duration > 21600: # > 6h
            fps = 0.05 
        elif duration > 7200: # > 2h
            fps = 0.1
            
        print(f"\n🕵️ [VisualScout] Analyzing Visual Energy ({fps} fps, adaptive)...")
        
        # 1. Extract thumbnails using FFmpeg with GPU acceleration & Keyframe-only decoding
        thumb_pattern = self.temp_dir / "thumb_%05d.jpg"
        
        # ✅ FIX: GPU 가속 시도 후 실패 시 CPU fallback
        cmd_gpu = [
            "ffmpeg", "-y", "-nostdin",
            "-hwaccel", "cuda", 
            "-discard", "nokey", # Skip P/B frames (Massive speedup!)
            "-i", str(video_path),
            "-vf", f"fps={fps},scale=320:-1", 
            "-q:v", "5", 
            str(thumb_pattern)
        ]
        
        cmd_cpu = [
            "ffmpeg", "-y", "-nostdin",
            "-discard", "nokey",
            "-i", str(video_path),
            "-vf", f"fps={fps},scale=320:-1", 
            "-q:v", "5", 
            str(thumb_pattern)
        ]
        
        try:
            subprocess.run(cmd_gpu, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            print("   ⚡ GPU acceleration enabled (CUDA)")
        except Exception as e:
            print(f"   ⚠️ GPU acceleration failed, falling back to CPU: {e}")
            try:
                subprocess.run(cmd_cpu, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                print("   🐢 Using CPU mode (slower)")
            except Exception as e2:
                print(f"❌ [VisualScout] Extraction failed: {e2}")
                return None

        # 2. Process thumbnails in parallel (Loading & Metadata Scanning)
        thumbs = sorted(list(self.temp_dir.glob("thumb_*.jpg")))
        
        def _calculate_entropy(img):
            """히스토그램 기반 이미지 엔트로피 계산 (정보량 측정)"""
            hist = img.histogram()
            hist_sum = sum(hist)
            if hist_sum == 0: return 0
            probs = [float(h) / hist_sum for h in hist]
            return -sum([p * math.log(p, 2) for p in probs if p != 0])

        def _load_and_scan(args):
            i, t_file = args
            curr_time = i / fps
            
            entropy_val = 0
            img_arr = None
            
            try:
                img = Image.open(t_file).convert('L')
                # 1. 히스토그램 엔트로피 (정보 변곡점 탐지)
                entropy_val = _calculate_entropy(img)
                
                # 2. 모션 분석용 저해상도 배열
                if i % 2 == 0:
                    img_small = img.resize((64, 64))
                    img_arr = np.array(img_small, dtype=np.int16)
            except Exception as e:
                pass
                
            return curr_time, entropy_val, img_arr, t_file

        print(f"   🧵 [VisualScout] Scanning {len(thumbs)} thumbnails in parallel...")
        task_args = list(enumerate(thumbs))
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            all_results = list(tqdm(executor.map(_load_and_scan, task_args), total=len(thumbs), desc="Scanning Metadata"))
        
        times, entropy, motion = [], [], []
        prev_arr = None
        for i, (t, e, arr, t_file) in enumerate(all_results):
            times.append(t)
            entropy.append(e)
            
            if i % 2 == 0:
                if arr is not None and prev_arr is not None:
                    motion.append(np.abs(arr - prev_arr).mean())
                else: motion.append(0)
                if arr is not None: prev_arr = arr
            else:
                motion.append(motion[-1] if motion else 0)
            
            try: os.remove(t_file)
            except: pass

        # Normalize scores
        entropy = np.array(entropy)
        motion = np.array(motion)
        
        if len(entropy) > 0:
            entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-6)
        if len(motion) > 0:
            motion = (motion - motion.min()) / (motion.max() - motion.min() + 1e-6)
            
        print(f"📊 [VisualScout] Done. Visual Peaks found: {len(np.where(motion > 0.5)[0])}")
        
        # [Save Cache]
        try:
            cache_data = {
                "times": times.tolist(),
                "entropy": entropy.tolist(),
                "motion": motion.tolist(),
                "visual_energy": (entropy * 0.4 + motion * 0.6).tolist()
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            print(f"💾 [VisualScout] Results cached to '{cache_path.name}'")
        except: pass

        return {
            "times": np.array(times),
            "entropy": entropy,
            "motion": motion,
            "visual_energy": (entropy * 0.4 + motion * 0.6)
        }

if __name__ == "__main__":
    scout = VisualScout()
