
class FactoryConfig:
    # --- Scoring Thresholds ---
    # 최종 점수 커트라인 (이 점수 이상만 영상으로 제작)
    GOLDEN_SCORE_THRESHOLD = 0.55 
    
    # --- Weights (황금 스코어 가중치) ---
    WEIGHTS = {
        'base': 0.4,          # 기본 오디오 신호 강도
        'emotion': 0.4,       # 감정/재미 (LLM 평가 0~1)
        'info': 0.2,          # 정보 밀도 (LLM 평가 0~1)
        'context_break': 0.2  # 맥락 단절 (감점 요소)
    }

    # --- Editorial Rules ---
    # 연속성 판단 기준 (이 시간보다 짧은 간격이면 하나로 묶거나 패널티 완화)
    CONTINUITY_GAP = 3.0  
    
    # LLM 평가 시 살펴볼 앞뒤 맥락 범위 (초 단위)
    # 클립 앞뒤로 이만큼 더 읽어서 상황을 파악함
    CONTEXT_WINDOW_SEC = 120 

    # --- System ---
    # LLM 평가 중간 저장 파일명
    CHECKPOINT_FILE = "temp_evals.json"

    @classmethod
    def load_preset(cls, json_path):
        import json
        import os
        
        if not os.path.exists(json_path):
            print(f"⚠️ Preset file not found: {json_path}")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📂 Loading Preset: {data.get('description', 'Custom Style')}")
            
            # Update Weights
            if 'weights' in data:
                # Map keys if necessary, pattern_learner uses different keys?
                # pattern_learner: audio_rms, audio_slope, audio_zcr, visual_clip
                # factory_config: base, emotion, info, context_break
                
                # Mapping Strategy:
                # base <- audio_rms + audio_slope (Signal strength)
                # emotion <- audio_zcr (Excitement) + visual_clip
                # info <- fixed default or from json if new key exists
                
                w = data['weights']
                cls.WEIGHTS['base'] = w.get('audio_rms', 0.2) + w.get('audio_slope', 0.2)
                cls.WEIGHTS['emotion'] = w.get('audio_zcr', 0.2) + w.get('visual_clip', 0.2)
                # info/context_break might not be in learner yet, keep defaults or check
                if 'info' in w: cls.WEIGHTS['info'] = w['info']
                if 'context_break' in w: cls.WEIGHTS['context_break'] = w['context_break']

            # Update Thresholds
            if 'thresholds' in data:
                # pattern_learner uses 'clamped_max_score'? No direct map to Golden Score
                # But we can look for specific override
                if 'golden_threshold' in data['thresholds']:
                     cls.GOLDEN_SCORE_THRESHOLD = data['thresholds']['golden_threshold']
            
            # Update Parameters
            if 'parameters' in data:
                 if 'merge_gap_seconds' in data['parameters']:
                     cls.CONTINUITY_GAP = data['parameters']['merge_gap_seconds'] / 20.0 # Scale down? Learner uses large values (~60s)
                     # Wait, Learner merge_gap is for merging candidates. Factory GAP is for Penalty.
                     # Maybe we should interpret merge_gap as tolerance.
                     cls.CONTINUITY_GAP = max(3.0, data['parameters']['merge_gap_seconds'] / 10.0)

            print(f"   -> Weights Updated: {cls.WEIGHTS}")
            print(f"   -> Threshold: {cls.GOLDEN_SCORE_THRESHOLD}")
            
        except Exception as e:
            print(f"❌ Failed to load preset: {e}")
