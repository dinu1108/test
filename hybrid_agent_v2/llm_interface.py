import os
import json
import re
import time
from google import genai
from google.genai import types
from config import load_api_key, get_all_api_keys

class LLMInterface:
    def __init__(self):
        load_api_key()
        self.api_keys = get_all_api_keys()
        self.current_key_idx = 0
        self.model_name = "models/gemini-2.5-flash"
        self.client = genai.Client(api_key=self.api_keys[0])

    def evaluate_candidates(self, kb, video_id, candidates):
        """
        [2단계: 평가자 역할] 
        각 후보 컷의 전후 맥락을 KB에서 조회하여 정밀 평가합니다.
        중간 저장(Checkpoint) 기능을 포함합니다.
        """
        import os
        from presets.factory_config import FactoryConfig
        
        print(f"[LLM] ⚖️ {len(candidates)}개의 후보 컷에 대해 정밀 심사를 시작합니다 (Contextual Evaluation)...")
        
        # Checkpoint Load
        ckpt_file = FactoryConfig.CHECKPOINT_FILE
        evaluations = []
        start_idx = 0
        
        if os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    evaluations = json.load(f)
                start_idx = len(evaluations)
                print(f"   🔄 Resuming from checkpoint: {start_idx}/{len(candidates)} completed.")
            except Exception as e:
                print(f"   ⚠️ Checkpoint load failed, starting fresh: {e}")

        for i in range(start_idx, len(candidates)):
            cand = candidates[i]
            
            # 1. Context Fetching (±2분)
            # get_context args: video_id, start, end. We expand range here
            ctx_start = max(0, cand['start'] - FactoryConfig.CONTEXT_WINDOW_SEC)
            ctx_end = cand['end'] + FactoryConfig.CONTEXT_WINDOW_SEC
            
            context_docs = kb.get_context(video_id, ctx_start, ctx_end)
            context_text = " ".join([d['text'] for d in context_docs])
            
            # 2. Single Evaluation Prompt
            eval_result = self._evaluate_single(cand, context_text)
            
            # Map ID correctly
            eval_result['id'] = cand['id'] # Ensure ID matches
            evaluations.append(eval_result)
            
            # 3. Checkpoint Save (Every 1 cut or 5 cuts? 1 is safer for expensive LLM)
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump(evaluations, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ Evaluated #{i} (Score E:{eval_result.get('emotion_intensity',0)})")
            
        return {"evaluations": evaluations}

    def _evaluate_single(self, candidate, context_text):
        prompt = f"""
# ROLE: 냉혹한 영상 분석가
# TASK: 아래 후보 컷(CANDIDATE)이 시청자에게 즐거움을 줄 수 있는지 평가하라.
# CONTEXT: 후보 컷의 전후 2분 대본을 참고하여 문맥을 파악하라.

# CANDIDATE INFO:
- Time: {candidate['start']} ~ {candidate['end']}
- Transcript: {candidate['text']}

# SURROUNDING CONTEXT (±2 min):
{context_text}

# EVALUATION CRITERIA (0~1.0):
1. emotion_intensity: (중요) 웃음, 분노, 감탄 등 감정이 폭발하는가?
2. info_density: 유용한 정보나 통찰이 있는가?
3. context_break: 앞뒤 맥락 없이 갑자기 튀어나와서 이해하기 어려운가? (높을수록 나쁨)
4. is_unnecessary: 로딩 화면, 무의미한 잡담 등 버려야 할 구간인가?

# OUTPUT FORMAT (JSON Only):
{{
  "emotion_intensity": 0.8,
  "info_density": 0.5,
  "context_break": 0.1,
  "is_unnecessary": false,
  "reason": "한국어로 짤막한 평가 (이유)"
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return self._safe_parse_json(response.text)
        except Exception as e:
            print(f"🚨 평가 실패 (ID: {candidate.get('id')}): {e}")
            return {} # Return empty dict, will be handled as missing score

    def _safe_parse_json(self, raw_text):
        try:
            return json.loads(raw_text)
        except:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            return json.loads(match.group()) if match else {"evaluations": []}