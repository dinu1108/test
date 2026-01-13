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

class PatternLearner:
    def __init__(self):
        self.tmp_dir = Path("learner_tmp")
        self.tmp_dir.mkdir(exist_ok=True)
        self.sr = 8000 # 동기화용 낮은 샘플링 레이트
        self.ana_sr = 16000 # 특징 분석용 샘플링 레이트
        self.MAX_DIFF_SEC = 5.0 # 허용 오차 시간

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
        
        print(f"[Learner] Syncing {total_chunks} blocks (Full-Scan Mode)...")
        
        y_raw_norm = (y_raw - np.mean(y_raw)) / (np.std(y_raw) + 1e-6)
        
        # [수정포인트 1] range 스텝을 1로 변경하여 전수 조사 (데이터 누수 방지)
        for i in tqdm(range(0, total_chunks), desc="Syncing"):
            s_e = i * chunk_len
            e_e = s_e + chunk_len
            chunk = y_edit[s_e:e_e]

            # [수정포인트 2] 무음 구간 건너뛰기로 연산 효율화
            if np.max(np.abs(chunk)) < 0.01: continue 

            chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-6)
            
            if last_raw_idx != -1:
                pred = last_raw_idx + chunk_len
                win = 40 * self.sr # 검색 윈도우 40초로 확장
                search_start = max(0, pred - win)
                search_end = min(len(y_raw), pred + win)
                
                y_search = y_raw_norm[search_start:search_end]
                if len(y_search) < len(chunk): continue
                
                cc = scipy.signal.correlate(y_search, chunk, mode='valid', method='fft')
                local_max = np.argmax(cc)
                global_idx = search_start + local_max
            else:
                cc = scipy.signal.correlate(y_raw_norm, chunk, mode='valid', method='fft')
                global_idx = np.argmax(cc)
            
            # 연속성 체크
            expected_pos = last_raw_idx + chunk_len if last_raw_idx != -1 else -1
            is_contiguous = False
            if last_raw_idx != -1:
                diff = abs(global_idx - expected_pos)
                if diff < (self.sr * self.MAX_DIFF_SEC):
                    is_contiguous = True
            
            if is_contiguous:
                current_len += chunk_len
                last_raw_idx = global_idx
            else:
                if current_start_raw != -1:
                     intervals.append((current_start_raw/self.sr, (current_start_raw + current_len)/self.sr))
                current_start_raw = global_idx
                current_len = chunk_len
                last_raw_idx = global_idx

        if current_start_raw != -1:
             intervals.append((current_start_raw/self.sr, (current_start_raw + current_len)/self.sr))
             
        # 인접 구간 병합 (10초 이내)
        merged = []
        if intervals:
            intervals.sort()
            curr_s, curr_e = intervals[0]
            for next_s, next_e in intervals[1:]:
                if next_s <= curr_e + 10.0:
                    curr_e = max(curr_e, next_e)
                else:
                    merged.append((curr_s, curr_e))
                    curr_s, curr_e = next_s, next_e
            merged.append((curr_s, curr_e))
            
        print(f"[Learner] Identified {len(merged)} cut segments.")
        return merged

    def profile_features(self, raw_wav_full, intervals):
        print("[Learner] Profiling Audio Features...")
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

    def get_frame_at(self, video_path, time_sec):
        cmd = ["ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path), "-vframes", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if len(r.stdout) > 0:
                return base64.b64encode(r.stdout).decode('utf-8')
        except: pass
        return None

    def llava_persona(self, raw_video, intervals, stats):
        if not intervals or not stats: return "Standard Highlight"
        print("[Learner] LLaVA Extracting Persona (Multi-Frame High-Energy Analysis)...")
        
        # [수정포인트 3] 에너지가 가장 높은 상위 3개 구간 추출
        candidates = []
        for i in range(len(intervals)):
            candidates.append({
                "slope": stats["rms_slope"][i],
                "start": intervals[i][0],
                "peak_offset": stats["preroll"][i]
            })
        candidates.sort(key=lambda x: x['slope'], reverse=True)
        
        images = []
        for c in candidates[:3]:
            abs_time = c['start'] + c['peak_offset']
            b64 = self.get_frame_at(raw_video, abs_time)
            if b64: images.append(b64)
            
        if not images: return "General Gaming Highlight"

        # [수정포인트 4] LLaVA 오판 방지용 강제 가이드 프롬프트
        prompt = (
            "Analyze this streamer's style. This is a VIDEO GAME BROADCAST (e.g. Dead by Daylight). "
            "Describe the streamer's reaction: Is he shouting, laughing, or making intense faces? "
            "Identify the 'Vibe' in ONE sentence (e.g., Chaotic, High-energy, Screaming, Comedic)."
        )
        
        try:
            res = requests.post("http://localhost:11434/api/generate", 
                                json={"model":"llava", "prompt":prompt, "images":images, "stream":False}, timeout=30)
            desc = res.json().get('response', 'Engaging Moment').strip()
            print(f"   -> Persona Identified: {desc[:60]}...")
            return desc
        except Exception as e:
            return "Generic High-Energy Streamer"

    def learn(self, center_dir, output_name):
        center = Path(center_dir)
        raws = list(center.glob("raw_full.mp4"))
        edits = list(center.glob("edited_*.mp4"))
        
        if not raws or not edits:
            print("[Error] Missing raw_full.mp4 or edited_*.mp4.")
            return

        raw_p, edit_p = raws[0], edits[0] 
        raw_wav = self.extract_audio(raw_p, self.tmp_dir / "raw.wav")
        edit_wav = self.extract_audio(edit_p, self.tmp_dir / "edit.wav")
        
        intervals = self.find_intervals(raw_wav, edit_wav)
        if not intervals: return
            
        stats = self.profile_features(raw_p, intervals) 
        persona = self.llava_persona(raw_p, intervals, stats)
        
        # [수정포인트 5] 김도님 스타일 가중치 최적화
        avg_slope = statistics.mean(stats['rms_slope']) if stats['rms_slope'] else 0
        w_rms, w_slope, w_zcr = 0.3, 0.2, 0.1
        
        if avg_slope > 0.04: # 소리 상승폭이 큰 경우 (비명/리액션형)
            w_slope = 0.7 
            w_rms = 0.1
            print(f"[Learner] 🚀 High-energy profile detected. Slope Weight boosted to 0.7")

        # [수정포인트 6] 편집 템포 강제 클램프 (30~40초)
        calc_gap = int(statistics.mean(stats['preroll']) * 1.5) + 10 if stats['preroll'] else 35
        final_merge_gap = max(30, min(calc_gap, 40))

        preset = {
            "description": f"Learned from {center.name}. {persona}",
            "weights": {
                "audio_rms": w_rms, "audio_slope": w_slope, "audio_zcr": w_zcr,
                "chat_velocity": 0.2, "visual_clip": 0.2
            },
            "thresholds": {
                "rms_min_db": 0.5, # 감도 대폭 향상
                "clamped_max_score": 5.0
            },
            "parameters": {
                "stage1_top_k": 300,
                "merge_gap_seconds": final_merge_gap
            },
            "prompts": {
                "llava": f"This is a highlight similar to: '{persona}'. Is the content engaging? Answer YES or NO."
            }
        }
        
        out = Path(f"presets/{output_name}.json")
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(preset, f, indent=4, ensure_ascii=False)
        print(f"✅ Successfully saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("center")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    style_name = args.name if args.name else Path(args.center).name
    PatternLearner().learn(args.center, style_name)
