import librosa
import numpy as np
import json
import os
import subprocess
import torch
import requests
import base64
import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class Analyst:
    def __init__(self, processed_dir="processed", config=None):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True)
        
        self.config = config if config else {}
        self.weights = self.config.get("weights", {})
        self.params = self.config.get("parameters", {})
        self.prompts = self.config.get("prompts", {})
        self.thresholds = self.config.get("thresholds", {})
        
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.STAGE1_TOP_K = self.params.get("stage1_top_k", 300)
        self.STAGE2_TOP_K = self.params.get("stage2_top_k", 80)
        self.FINAL_TOP_K = self.params.get("stage3_top_k", 30)

    def download_video(self, url):
        try:
            output_template = str(self.processed_dir / "%(id)s.%(ext)s")
            cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4", "--write-sub", "--write-auto-sub", "--sub-lang", "ko,en", "--write-comments", "-o", output_template, url]
            subprocess.run(cmd, check=True)
            files = list(self.processed_dir.glob("*.mp4")) # Simple glob for demo
            if not files: return None, None, None
            # Find the most recently modified file
            video_path = max(files, key=os.path.getctime)
            chat_path = str(video_path).replace(".mp4", ".live_chat.json")
            if not os.path.exists(chat_path): chat_path = None
            return str(video_path), None, chat_path
        except: return None, None, None

    def _load_clip(self):
        if self.model: return
        from transformers import CLIPProcessor, CLIPModel
        print(f"[Analyst] Loading CLIP ({self.device})...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _unload_clip(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if self.device == "cuda": torch.cuda.empty_cache()

    def analyze_audio_advanced(self, video_path):
        cache_path = self.processed_dir / f"{Path(video_path).name}.audio_v14.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f: return pickle.load(f)

        print("[Analyst] Audio Feature Extraction (Slope/ZCR)...")
        temp_wav = self.processed_dir / "temp_audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", str(temp_wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            y, sr = librosa.load(str(temp_wav), sr=16000)
            rms = librosa.feature.rms(y=y, hop_length=2048)[0]
            zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=2048)[0]
            
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
            zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-6)
            
            slope = np.diff(rms_norm, prepend=0)
            slope = np.maximum(0, slope) # positive slope only
            
            data = {"rms": rms_norm, "slope": slope, "zcr": zcr_norm, "times": librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=2048)}
            with open(cache_path, 'wb') as f: pickle.dump(data, f)
            if temp_wav.exists(): temp_wav.unlink()
            return data
        except: return None

    def calculate_scores(self, audio_data):
        w_rms = self.weights.get("audio_rms", 0.3)
        w_slope = self.weights.get("audio_slope", 0.3)
        w_zcr = self.weights.get("audio_zcr", 0.2)
        
        raw = (audio_data['rms'] * w_rms) + (audio_data['slope'] * w_slope) + (audio_data['zcr'] * w_zcr)
        
        final = np.zeros_like(raw)
        max_s = self.thresholds.get("clamped_max_score", 5.0)
        
        # Vectorized accumulation approximation or loop
        # Loop for temporal dependency
        for t in range(2, len(raw)):
            val = raw[t] + (raw[t-1] * 0.7) + (raw[t-2] * 0.4)
            final[t] = min(val, max_s)
        return final, audio_data['times']

    def extract_frames(self, video_path, candidates):
        # returns (time, image_array, raw_score)
        valid = []
        def _job(c):
            t = c['time']
            cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "224x224", "pipe:1"]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=5)
                if len(r.stdout) > 0:
                    frm = np.frombuffer(r.stdout, dtype=np.uint8).reshape(224,224,3)
                    return (t, frm, c['score'])
            except: pass
            return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_job, c) for c in candidates]
            for f in as_completed(futs):
                res = f.result()
                if res: valid.append(res)
        valid.sort(key=lambda x: x[0])
        return valid

    def run_clip_stage(self, frames):
        self._load_clip()
        from torchvision import transforms
        norm = transforms.Normalize(mean=[0.48, 0.45, 0.40], std=[0.26, 0.26, 0.27])
        
        pos = ["face of a streamer", "person expressing emotion", "exciting game moment"]
        neg = ["loading screen", "black screen", "static menu"]
        
        txt_tok = self.processor(text=pos+neg, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            txt_feat = self.model.get_text_features(**txt_tok)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            
        results = []
        bs = 64
        for i in range(0, len(frames), bs):
            batch = frames[i:i+bs]
            imgs = np.stack([b[1] for b in batch])
            t_imgs = torch.from_numpy(imgs).permute(0,3,1,2).float().to(self.device)/255.0
            t_imgs = norm(t_imgs)
            
            with torch.no_grad():
                i_feat = self.model.get_image_features(pixel_values=t_imgs)
                i_feat /= i_feat.norm(dim=-1, keepdim=True)
                probs = (100.0 * i_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()
                
            for j, p in enumerate(probs):
                if p[:len(pos)].max() > p[len(pos):].max():
                    # Weighted Fusion
                    visual_score = p[:len(pos)].max() * 100
                    final = (batch[j][2] * 20) + (visual_score * self.weights.get("visual_clip", 0.1))
                    results.append({'time': batch[j][0], 'score': final})
        
        self._unload_clip()
        return results

    def run_llava_stage(self, video_path, candidates):
        verified = []
        prompt = self.prompts.get("llava", "Is this a highlight?")
        url = "http://localhost:11434/api/generate"
        
        print(f"[Analyst] LLaVA Checking {len(candidates)} candidates...")
        for c in tqdm(candidates, desc="LLaVA"):
            t = c['time']
            # Re-extract full quality frame
            cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-vframes", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=5)
                if len(r.stdout) > 0:
                    b64 = base64.b64encode(r.stdout).decode('utf-8')
                    # yes/no check
                    res = requests.post(url, json={"model":"llava", "prompt": prompt, "images":[b64], "stream":False}, timeout=20)
                    if "yes" in res.json().get('response', '').lower():
                        # Desc check
                        d_res = requests.post(url, json={"model":"llava", "prompt": "Describe the emotion and situation briefly.", "images":[b64], "stream":False})
                        c['summary'] = d_res.json().get('response', 'Highlight')
                        verified.append(c)
            except: pass
        return verified

    def get_highlights(self, video_path, chat_path=None):
        # 1. Audio Analysis
        data = self.analyze_audio_advanced(video_path)
        if not data: return []
        
        # 2. Score
        scores, times = self.calculate_scores(data)
        
        # 3. Hybrid Filter (Top-K + Soft)
        indices = np.argsort(scores)[::-1]
        top_k = indices[:self.STAGE1_TOP_K]
        
        soft_val = self.params.get("soft_threshold_score", 0.8)
        soft = np.where(scores >= soft_val)[0]
        
        candidates_idx = np.union1d(top_k, soft)
        candidates_idx.sort()
        
        # Debounce
        candidates = []
        last_t = -999
        for i in candidates_idx:
            t = times[i]
            if t - last_t > 15:
                candidates.append({'time': t, 'score': scores[i]})
                last_t = t
                
        print(f"[Stage 1] {len(candidates)} Candidates")
        
        # 4. CLIP
        frames = self.extract_frames(video_path, candidates)
        clip_cands = self.run_clip_stage(frames)
        clip_cands.sort(key=lambda x: x['score'], reverse=True)
        clip_passed = clip_cands[:self.STAGE2_TOP_K]
        print(f"[Stage 2] {len(clip_passed)} Candidates")
        
        # 5. LLaVA
        final = self.run_llava_stage(video_path, clip_passed)
        print(f"[Stage 3] {len(final)} Verified Highlights")
        
        final.sort(key=lambda x: x['time'])
        
        # Format
        output = []
        for f in final:
            output.append({
                'start': max(0, f['time'] - 60),
                'end': f['time'] + 120,
                'summary': f.get('summary', 'AI Highlight'),
                'score': f['score']
            })
        return output
