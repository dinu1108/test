
from pathlib import Path

class FactoryConfig:
    # --- Scoring Thresholds ---
    # 최종 점수 커트라인 (이 점수 이상만 영상으로 제작)
    GOLDEN_SCORE_THRESHOLD = 0.55 
    
    # --- Weights (황금 스코어 가중치) ---
    WEIGHTS = {
        'base': 0.1,          # 기본 오디오 신호 강도 (비중 최소화)
        'emotion': 0.4,       # 감정/재미 (가중치 유지)
        'info': 0.5,          # 정보 밀도/서사적 가치 (가중치 극대화)
        'context_break': 0.3, # 맥락 단절 (감점 유지)
        'payoff': 0.2         # 서사적 보상
    }

    # --- Editorial Rules ---
    # 연속성 판단 기준 (이 시간보다 짧은 간격이면 하나로 묶거나 패널티 완화)
    CONTINUITY_GAP = 3.0  
    
    # LLM 평가 시 살펴볼 앞뒤 맥락 범위 (초 단위)
    # 클립 앞뒤로 이만큼 더 읽어서 상황을 파악함
    CONTEXT_WINDOW_SEC = 120 
    
    # --- Timing (타임라인 확장) ---
    PREROLL = 60   # 컷 시작 지점 확장 (40s -> 60s)
    POSTROLL = 20  # 컷 종료 지점 확장 (초)

    # --- System ---
    # LLM 평가 중간 저장 파일명 및 디렉토리
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_FILE = CHECKPOINT_DIR / "temp_evals.json"
    
    # [NEW] Style Description for LLM
    DESCRIPTION = "General Video Highlight"

    # [NEW] Variable Thresholds for Editorial Agent
    NARRATIVE_PRIORITY_THRESHOLD = 3   # 이 점수 미만은 폐기
    SMART_MERGE_GAP = 120.0            # 이 시간(초) 이내면 앞 클립과 합침
    DEBOUNCE_SECONDS = 60.0            # V1 필터링 시 중복 제거 시간 간격

    AUTO_APPROVE = False
    SKIP_VISUAL = False
    ALLOW_VISUAL_FALLBACK = True # GPU 실패 시 CPU 자동 전환 허용
    
    # --- HW & Engine ---
    VIDEO_CODEC = "h264_nvenc" # "libx264" for CPU, "h264_nvenc" for NVIDIA GPU
    WHISPER_MODEL = "small"    # Default to "small" for throughput, "medium" for golden pass

    @classmethod
    def validate_preset(cls, data):
        """
        필수 프리셋 키가 누락되었는지 검증
        """
        required_keys = ['weights', 'parameters']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required preset key: {key}")
        return True

    @classmethod
    def load_preset(cls, json_path):
        import json
        import os

        if not os.path.exists(json_path):
            # [Fix] 만약 직접 경로를 찾지 못했다면 presets/ 폴더에서 찾아봅니다.
            presets_dir = os.path.join(os.getcwd(), "presets")
            potential_path = os.path.join(presets_dir, json_path)
            
            # 폴더인지 파일인지 확인
            if os.path.exists(potential_path):
                json_path = potential_path
            elif os.path.exists(potential_path + ".json"):
                json_path = potential_path + ".json"
            else:
                print(f"⚠️ Preset path not found: {json_path}")
                return

        # [NEW] 폴더 경로가 들어올 경우, 해당 폴더 내의 {folder_name}.json 찾기
        if os.path.isdir(json_path):
            folder_name = os.path.basename(json_path.rstrip(os.sep))
            target_json = os.path.join(json_path, f"{folder_name}.json")
            if os.path.exists(target_json):
                json_path = target_json
            else:
                print(f"⚠️ Could not find '{folder_name}.json' inside {json_path}")
                return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # [NEW] Load Description
            if 'description' in data:
                cls.DESCRIPTION = data['description']

            print(f"📂 Loading Preset: {cls.DESCRIPTION}")
            
            # Update Weights (Robust Mapping)
            if 'weights' in data:
                w = data['weights']
                
                # 1. Base Signal (Signal Strength)
                # pattern_learner: audio_rms, audio_slope
                if 'audio_rms' in w or 'audio_slope' in w:
                    cls.WEIGHTS['base'] = w.get('audio_rms', 0.1) + w.get('audio_slope', 0.1)
                
                # 2. Emotion/Excitement (Visual & Audio Complexity)
                # pattern_learner: audio_zcr, visual_clip
                if 'audio_zcr' in w or 'visual_clip' in w:
                    cls.WEIGHTS['emotion'] = w.get('audio_zcr', 0.2) + w.get('visual_clip', 0.2)
                
                # 3. Direct Overrides (if present)
                for key in ['base', 'emotion', 'info', 'context_break', 'payoff']:
                    if key in w:
                        cls.WEIGHTS[key] = w[key]
                        
                # Ensure values aren't zero if they shouldn't be
                cls.WEIGHTS['base'] = max(0.1, cls.WEIGHTS['base'])
                cls.WEIGHTS['emotion'] = max(0.1, cls.WEIGHTS['emotion'])

            # Update Thresholds
            if 'thresholds' in data:
                # pattern_learner uses 'clamped_max_score'? No direct map to Golden Score
                # But we can look for specific override
                if 'golden_threshold' in data['thresholds']:
                     cls.GOLDEN_SCORE_THRESHOLD = data['thresholds']['golden_threshold']
            
            # Update Parameters
            if 'parameters' in data:
                 if 'merge_gap_seconds' in data['parameters']:
                     cls.CONTINUITY_GAP = max(3.0, data['parameters']['merge_gap_seconds'] / 10.0)
                 
                 # [NEW] Editorial Agent Configs
                 if 'narrative_priority' in data['parameters']:
                     cls.NARRATIVE_PRIORITY_THRESHOLD = data['parameters']['narrative_priority']
                 if 'smart_merge_gap' in data['parameters']:
                     cls.SMART_MERGE_GAP = data['parameters']['smart_merge_gap']
                 if 'debounce_seconds' in data['parameters']:
                     cls.DEBOUNCE_SECONDS = data['parameters']['debounce_seconds']
                 if 'auto_approve' in data['parameters']:
                     cls.AUTO_APPROVE = data['parameters']['auto_approve']
                 if 'preroll' in data['parameters']:
                     cls.PREROLL = data['parameters']['preroll']
                 if 'postroll' in data['parameters']:
                     cls.POSTROLL = data['parameters']['postroll']
                 if 'video_codec' in data['parameters']:
                     cls.VIDEO_CODEC = data['parameters']['video_codec']
                 if 'whisper_model' in data['parameters']:
                     cls.WHISPER_MODEL = data['parameters']['whisper_model']

            print(f"   -> Weights Updated: {cls.WEIGHTS}")
            print(f"   -> Threshold: {cls.GOLDEN_SCORE_THRESHOLD}")
            
        except Exception as e:
            print(f"❌ Failed to load preset: {e}")
