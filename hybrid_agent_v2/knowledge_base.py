import chromadb
import whisper
import os
import re
import subprocess
import numpy as np
import chromadb.utils.embedding_functions as ef_functions
from pathlib import Path
from config import load_api_key

class VideoKnowledgeBase:
    def __init__(self, collection_name="video_memory"):
        load_api_key() # API 키 로드 (Whisper 등 다른 용도 위해 유지)
        
        self.persist_dir = Path("./hybrid_agent_v2/chroma_db")
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        
        # [변경] 구글 API 대신 로컬 모델(SentenceTransformer) 사용
        # 이 모델은 사용자님의 컴퓨터에서 직접 실행되어 할당량 제한이 없습니다.
        print("[KnowledgeBase] 📥 Loading Local Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding_function = ef_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # [주의] 임베딩 모델이 바뀌면 기존 컬렉션과 호환되지 않습니다.
        # 기존 데이터를 유지하려면 별도 마이그레이션이 필요하나, Factory 모드 특성상
        # 충돌 방지를 위해 기존 컬렉션을 삭제하고 새로 만드는 것이 안전할 수 있습니다.
        # 여기서는 get_or_create로 하되, 에러 발생 시 안내 메시지를 띄우는 것이 좋습니다.
        
        self.log_file = self.persist_dir / "rejection_logs.jsonl" # 실패 로그 파일
        
        # Initialize Collection
        self.init_collection(collection_name)
        
        print(f"[KnowledgeBase] 🏠 Local Embedding Mode (Unlimited) Ready.")
        self.whisper_model = None

    def log_rejection(self, video_id, candidate_data, reason, final_score):
        """탈락한 후보를 로그에 기록 (자동 튜닝용 Seed)"""
        import json
        from datetime import datetime
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "video_id": video_id,
            "candidate": candidate_data,
            "reason": reason,
            "final_score": final_score
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[KB] ⚠️ Log Error: {e}")

    def init_collection(self, collection_name):
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[KB] ⚠️ Collection Error: {e}")
            print("[KB] 기존 DB와 차원이 다를 수 있습니다. 'hybrid_agent_v2/chroma_db' 폴더를 삭제하고 다시 시도하세요.")
            raise e
        
        print(f"[KnowledgeBase] 🏠 Local Embedding Mode (Unlimited) Ready.")
        self.whisper_model = None

    def _load_whisper(self):
        if not self.whisper_model:
            print("[KnowledgeBase] Loading Whisper Model (base)...")
            self.whisper_model = whisper.load_model("base")

    def extract_audio(self, video_path):
        # [Safety Check] (Preserved from previous fix)
        video_path = Path(video_path)
        if not video_path.exists():
            potential_path = Path("raw_data") / video_path.name
            if potential_path.exists():
                print(f"[KB] 📍 Found prompt file in raw_data: {potential_path}")
                video_path = potential_path
        
        # [User Request] Absolute path & Verbose Debugging
        video_path_obj = video_path.absolute() # 절대 경로로 변경
        audio_path = video_path_obj.with_suffix(".wav")
        
        if not audio_path.exists():
            print(f"[KnowledgeBase] Extracting audio from {video_path_obj.name}...")
            # 리스트 형태의 인자는 공백/특수문자를 자동으로 처리하지만, 
            # 윈도우에서는 shell=True와 함께 문자열로 주는 것이 더 안전할 때가 있습니다.
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path_obj),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path)
            ]
            try:
                # stderr를 DEVNULL로 보내지 말고 출력하게 하여 에러 원인을 확인합니다.
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg 에러 발생! 파일명에 특수문자가 있는지, 혹은 ffmpeg가 설치되었는지 확인하세요.")
                raise e
        return str(audio_path)

    def ingest(self, video_path):
        """영상을 전사하고 벡터 DB에 인덱싱 (Rate Limit 대응)"""
        import time
        video_id = Path(video_path).stem
        existing = self.collection.get(where={"video_id": video_id}, limit=1)
        if existing['ids']:
            print(f"[KnowledgeBase] Video '{video_id}' already indexed. Skipping.")
            return

        self._load_whisper()
        audio_path = self.extract_audio(video_path)
        
        print(f"[KnowledgeBase] Transcribing '{video_id}'... (Outputting logs for progress)")
        # Whisper Python API doesn't have a native progress bar, using verbose=True to show activity.
        result = self.whisper_model.transcribe(audio_path, language="ko", verbose=True)
        
        ids, docs, metadatas = [], [], []
        
        from tqdm import tqdm
        print(f"[KnowledgeBase] Indexing {len(result['segments'])} segments into ChromaDB...")
        
        # Batch Size Reduced to 10 to avoid Rate Limits (Free Tier)
        BATCH_SIZE = 10
        
        for i, seg in enumerate(tqdm(result['segments'], desc="Indexing")):
            text = seg['text'].strip()
            if len(text) < 2: continue
            
            ids.append(f"{video_id}_{i}")
            docs.append(text)
            metadatas.append({
                "video_id": video_id,
                "start": float(seg['start']),
                "end": float(seg['end'])
            })
            
            if len(ids) >= BATCH_SIZE:
                self._add_batch(ids, docs, metadatas)
                ids, docs, metadatas = [], [], []
                # Local Embedding: No sleep needed

        if ids:
            self._add_batch(ids, docs, metadatas)
        print(f"[KnowledgeBase] Ingest Complete for {video_id}.")

    def _add_batch(self, ids, docs, metadatas):
        import time
        max_retries = 5
        base_delay = 10
        
        for attempt in range(max_retries):
            try:
                self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
                return # Success
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    delay = base_delay * (attempt + 1)
                    print(f"\n[KB] ⚠️ Rate Limit (429) hit. Sleeping for {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    print(f"\n[KB] ❌ DB Insert Error: {e}")
                    raise e
        print("\n[KB] ❌ Failed to insert batch after retries due to Rate Limits.")
        raise RuntimeError("ChromaDB Insert Failed: Rate Limit Exceeded")

    def get_context(self, video_id, start_time, end_time):
        """특정 시간 범위 내의 대본 추출 (수정 완료)"""
        results = self.collection.get(
            where={
                "$and": [
                    {"video_id": video_id},
                    {"start": {"$gte": float(start_time)}},
                    {"start": {"$lte": float(end_time)}}
                ]
            }
        )
        
        segments = []
        if results['ids']:
            for i in range(len(results['ids'])):
                segments.append({
                    "text": results['documents'][i],
                    "start": results['metadatas'][i]['start']
                })
        return sorted(segments, key=lambda x: x['start'])

    def clean_text_for_llm(self, text):
        text = re.sub(r'\[.*?\]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def get_optimized_transcript(self, video_path, threshold_percentile=80):
        """[Token Saver] V1 오디오 분석과 연동하여 대본 압축"""
        # 1. 로컬 V1 분석기 가동
        try:
            from modules.analyst import Analyst
        except ImportError:
            print("[KB] ⚠️ V1 Analyst module not found. Returning full text.")
            return "V1 Analyst module error."

        print("[KB] 📉 Running 'Token Saver' Pre-filtering (Audio Analysis)...")
        
        # Load Khan Preset Manually
        import json
        preset_path = Path("presets/Khan.json")
        if preset_path.exists():
            with open(preset_path, "r", encoding="utf-8") as f:
                chk_config = json.load(f)
        else:
            print("[KB] ⚠️ 'presets/Khan.json' not found. Using default config.")
            chk_config = {}

        analyst = Analyst(config=chk_config)
        audio_data = analyst.analyze_audio_advanced(video_path)
        
        if not audio_data:
            print("[KB] ⚠️ Audio analysis failed. Falling back to FULL transcript.")
            return self.get_full_transcript(video_path)
            
        scores, times = analyst.calculate_scores(audio_data)
        threshold = np.percentile(scores, threshold_percentile)
        active_indices = np.where(scores > threshold)[0]
        
        if len(active_indices) == 0: return "No active zones found."
            
        # 2. 활성 구간 병합 (Peak +/- 3분)
        active_times = times[active_indices]
        ranges = sorted([(max(0, t - 180), t + 180) for t in active_times])
        
        merged = []
        if ranges:
            curr_start, curr_end = ranges[0]
            for start, end in ranges[1:]:
                if start < curr_end:
                    curr_end = max(curr_end, end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = start, end
            merged.append((curr_start, curr_end))
            
        # 3. 데이터 추출 및 30초 단위 블록화
        video_id = Path(video_path).stem
        optimized_lines = []
        
        for start, end in merged:
            segs = self.get_context(video_id, start, end)
            
            # 30초 단위로 텍스트 묶기 (Text Slimming)
            current_block_id = -1
            for s in segs:
                block_id = int(s['start'] // 30)
                clean_txt = self.clean_text_for_llm(s['text'])
                if not clean_txt: continue
                
                if block_id != current_block_id:
                    m, sec = divmod(block_id * 30, 60)
                    h, m = divmod(m, 60)
                    timestamp = f"[{int(h):02d}:{int(m):02d}:{int(sec):02d}]"
                    optimized_lines.append(f"\n{timestamp} {clean_txt}")
                    current_block_id = block_id
                else:
                    optimized_lines[-1] += f" {clean_txt}"

        return "".join(optimized_lines)

    def get_full_transcript(self, video_path):
        """저장된 모든 문장을 가져와서 하나의 대본으로 만듭니다 (Fallback전용)"""
        video_id = Path(video_path).stem
        print(f"[KnowledgeBase] 📝 Fetching all segments for {video_id}...")
        results = self.collection.get(
            where={"video_id": video_id}
        )
        
        if not results['ids']:
            return ""

        # 시간순으로 정렬
        segments = []
        for i in range(len(results['ids'])):
            segments.append({
                "start": results['metadatas'][i]['start'],
                "text": results['documents'][i]
            })
        
        sorted_segs = sorted(segments, key=lambda x: x['start'])
        
        # 30초 단위로 묶어서 텍스트 양 최적화
        full_text = []
        current_block = -1
        for s in sorted_segs:
            block = int(s['start'] // 30)
            if block != current_block:
                m, sec = divmod(block * 30, 60)
                h, m = divmod(m, 60)
                full_text.append(f"\n[{int(h):02d}:{int(m):02d}:{int(sec):02d}] {s['text']}")
                current_block = block
            else:
                full_text[-1] += f" {s['text']}"
        
        return "".join(full_text)
