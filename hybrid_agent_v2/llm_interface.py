import os
import json
import re
import time
from pathlib import Path
from google import genai
from google.genai import types
from config import load_api_key, get_all_api_keys
from presets.factory_config import FactoryConfig

from .rejection_analyst import RejectionAnalyst

class LLMInterface:
    def __init__(self):
        load_api_key()
        self.api_keys = get_all_api_keys()
        self.current_key_idx = 0
        self.model_name = "gemini-1.5-flash-latest" # ✅ V5: 404 해결을 위한 Fallback ID
        self._initialize_client()
        self.ra = RejectionAnalyst()

    def _initialize_client(self):
        """✅ FIX: 클라이언트 초기화 분리"""
        if not self.api_keys:
            raise ValueError("No API keys available!")
        self.client = genai.Client(api_key=self.api_keys[self.current_key_idx])

    def _rotate_api_key(self):
        """✅ FIX: API 키 로테이션 구현"""
        if len(self.api_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            self._initialize_client()
            print(f"[LLM] 🔄 Switched to API key #{self.current_key_idx + 1}")
            return True
        return False

    def evaluate_candidates(self, kb, video_id, candidates, force_refresh=False):
        """
        [2단계: 평가자 역할]
        후보 컷의 전후 맥락을 조회하여 정밀 평가 (V5: 체크포인트 로직 복구 + 리프레시 옵션).
        """
        ckpt_file = Path("./hybrid_agent_v2/chroma_db") / f"{video_id}_ckpt.json"
        evaluations = []
        start_idx = 0

        if not force_refresh and ckpt_file.exists():
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    evaluations = json.load(f)
                start_idx = len(evaluations)
                print(f"   🔄 Resuming from checkpoint: {start_idx}/{len(candidates)} completed.")
                if start_idx >= len(candidates): return {"evaluations": evaluations}
            except Exception as e:
                print(f"   ⚠️ Checkpoint load failed: {e}")

        print(f"[LLM] ⚖️ {len(candidates)}개의 후보 컷에 대해 정밀 심사를 시작합니다...")
        
        for batch_start in range(start_idx, len(candidates), 10):
            batch = candidates[batch_start:batch_start + 10]
            # ... (Context gathering code stays similar)
            
            # Gather contexts for the batch
            batch_data = []
            for i, cand in enumerate(batch, start=batch_start):
                c_start = cand.get('start', cand.get('peak_time', 0) - 15)
                c_end = cand.get('end', cand.get('peak_time', 0) + 15)
                window = getattr(FactoryConfig, 'CONTEXT_WINDOW_SEC', 120)
                ctx_start = max(0, c_start - window)
                ctx_end = c_end + window
                context_docs = kb.get_context(video_id, ctx_start, ctx_end)
                context_text = " ".join([d['text'] for d in context_docs])
                
                batch_data.append({
                    "id": cand.get('id', i),
                    "start": c_start,
                    "end": c_end,
                    "text": cand.get('text', "No transcript"),
                    "context": context_text,
                    "speech_density": cand.get('speech_density', 0.5)
                })

            # ✅ FIX: 호출 간격 2초 (V5 가이드)
            import time
            time.sleep(2)
            
            print(f"DEBUG: LLMInterface Current Model ID -> {self.model_name}")
            # Perform batch evaluation via LLM
            batch_results = self._evaluate_batch(batch_data)
            
            # ✅ FIX: 실패 시 None을 반환받아 체크포인트 오염 방지
            if batch_results is None:
                print(f"   ⚠️ Batch processing failed. Stopping to preserve checkpoint integrity.")
                break

            evaluations.extend(batch_results)

            # ✅ FIX: V5 체크포인트 상시 저장
            try:
                ckpt_file = Path("./hybrid_agent_v2/chroma_db") / f"{video_id}_ckpt.json" # Re-define for scope
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_file, "w", encoding="utf-8") as f:
                    json.dump(evaluations, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"   ⚠️ Failed to write checkpoint: {e}")

        return {"evaluations": evaluations}

    def _evaluate_batch(self, batch_data):
        """
        여러 후보를 한 번에 LLM에 보내 평가받음 (비용 및 시간 절감)
        """
        from presets.factory_config import FactoryConfig
        style_desc = FactoryConfig.DESCRIPTION
        
        candidates_info = ""
        for item in batch_data:
            candidates_info += f"- ID {item['id']}: {item['start']:.1f}~{item['end']:.1f} / Text: {item['text']}\n"
            candidates_info += f"  Context: {item['context'][:1000]}...\n\n"

        prompt = f"""
# ROLE: 서사 중심의 영상 다큐멘터리 편집자
# TASK: 아래 제공된 여러 후보 컷들을 분석하여 점수를 매겨라.

# STYLE GUIDE: "{style_desc}"

# 🧠 LESSONS FROM THE PAST:
{self.ra.get_editing_feedback(limit=5)}

# CANDIDATES TO EVALUATE:
{candidates_info}

# EVALUATION CRITERIA (0~1.0):
1. emotion_intensity: 감정 폭발 정도
2. info_density: 서사적 가치 (무슨 일이 일어났는지 이해 가능 여부)
3. narrative_payoff: 빌드업에 대한 보상 (성공/실패/반전 등)
4. context_break: 맥락 단절 (높을수록 감점)
5. is_unnecessary: 버려야 할 구간 여부 (0 or 1)

# OUTPUT FORMAT (JSON Array of Objects):
[
  {{
    "id": 0,
    "emotion_intensity": 0.8,
    "info_density": 0.9,
    "narrative_payoff": 0.8,
    "context_break": 0.1,
    "is_unnecessary": 0,
    "reason": "한국어 한 줄 평가"
  }},
  ...
]
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # ✅ FIX: 강제 호출 간격 벌리기 (2초)
                time.sleep(2)
                
                print(f"DEBUG: LLMInterface Batch Current Model ID -> {self.model_name}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2 # ✅ JSON 준수율을 위해 조금 더 낮춤
                    )
                )
                results = json.loads(response.text)
                
                # 성공 시 즉시 반환 처리
                results_dict = {res['id']: res for res in results if 'id' in res}
                final_results = []
                for item in batch_data:
                    res = results_dict.get(item['id'], {
                        "emotion_intensity": 0, "info_density": 0, 
                        "narrative_payoff": 0, "context_break": 1.0, "is_unnecessary": 1,
                        "reason": "평가 누락"
                    })
                    res['id'] = item['id']
                    res['speech_density'] = item.get('speech_density', 0.5)
                    if 'reason' not in res: res['reason'] = "평가 내용 없음"
                    final_results.append(res)
                return final_results
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ⚠️ Batch evaluation error (Attempt {attempt+1}/{max_retries}): {error_msg}")
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    if self._rotate_api_key():
                        continue
                
                if attempt < max_retries - 1:
                    time.sleep(5) # 재시도 시에는 조금 더 길게 대기
        
        # ✅ FIX: 모든 재시도 실패 시 None 리턴 (이유를 섞지 않음)
        return None

    def _evaluate_single(self, candidate, context_text):
        """
        단일 후보 평가 (안전성 강화)
        """
        from presets.factory_config import FactoryConfig
        
        # ⚠️ [FIX] 다중 키 체인 방어 (start -> peak_time -> 기본값 0)
        c_start = candidate.get('start') or candidate.get('peak_time', 0) - 15
        c_end = candidate.get('end') or candidate.get('peak_time', 0) + 15
        c_text = candidate.get('text', "No transcript")
        style_desc = FactoryConfig.DESCRIPTION
        
        # 시간 값이 음수가 되지 않도록 보호
        c_start = max(0, float(c_start))
        c_end = max(c_start + 1, float(c_end))  # 최소 1초 길이 보장
        
        prompt = f"""
# ROLE: 서사 중심의 영상 다큐멘터리 편집자
# TASK: 단순히 시끄러운 구간이 아니라, '이야기의 결실'이 있는 구간을 찾아라.

# 시청자가 원하는 '정보(Information)'의 정의:
1. 빌드업의 끝: 고생하던 미션을 마침내 성공하거나 허무하게 실패하는 '결과'가 있는가?
2. 반전의 순간: 평온하다가 갑자기 예상치 못한 사건(갑툭튀, 버그, 배신)이 터지는가?
3. 감정의 근거: 비명을 지른다면 그 이유가 대본 상에 명확히 드러나는가? (이유 없는 비명은 감점)

# STYLE GUIDE: "{style_desc}"

# 🧠 LESSONS FROM THE PAST (Self-Correction):
{self.ra.get_editing_feedback(limit=5)}
(위 피드백을 참고하여 이번 평가에서는 더 정교한 안목을 적용하라)

# CANDIDATE INFO:
- Time: {c_start:.1f} ~ {c_end:.1f}
- Transcript: {c_text}

# SURROUNDING CONTEXT (±2 min):
{context_text}

# EVALUATION CRITERIA (0~1.0):
1. emotion_intensity: 웃음, 분노, 감탄 등 감정이 폭발하는가?
2. info_density: (중요) 이 컷만 봐도 '무슨 일이 일어났는지' 이해할 수 있는가? 서사적 가치가 있는가?
3. narrative_payoff: 앞선 상황에 대한 보상(성공/실패/웃음 포인트)이 확실한가?
4. context_break: 앞뒤 맥락 없이 갑자기 튀어나와서 이해하기 어려운가? (높을수록 나쁨)
5. is_unnecessary: 로딩 화면, 무의미한 잡담 등 버려야 할 구간인가?

# OUTPUT FORMAT (JSON Only):
{{
  "emotion_intensity": 0.8,
  "info_density": 0.9,
  "narrative_payoff": 0.8,
  "context_break": 0.1,
  "is_unnecessary": false,
  "reason": "한국어로 짤막한 평가 (정보/서사성 위주로 기술)"
}}
"""
        try:
            # ✅ FIX: 호출 간격 2초 (V5 가이드)
            time.sleep(2)
            
            print(f"DEBUG: LLMInterface Single Current Model ID -> {self.model_name}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3  # 평가 일관성 향상
                )
            )
            result = self._safe_parse_json(response.text)
            
            # 필수 키 보장 (Default Key Merging)
            defaults = {
                "emotion_intensity": 0.0,
                "info_density": 0.0,
                "narrative_payoff": 0.0,
                "context_break": 1.0,
                "is_unnecessary": False,
                "reason": "평가 실패"
            }
            return {**defaults, **result}  # 누락된 키는 기본값으로 채움
            
        except Exception as e:
            print(f"🚨 평가 실패 (candidate ID: {candidate.get('id', '?')}): {e}")
            return {
                "emotion_intensity": 0,
                "info_density": 0,
                "narrative_payoff": 0,
                "context_break": 1.0,
                "is_unnecessary": False,
                "reason": f"API Error: {str(e)[:50]}"
            }

    def _safe_parse_json(self, raw_text):
        if not raw_text: return {"emotion_intensity": 0}
        try:
            return json.loads(raw_text)
        except:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            return json.loads(match.group()) if match else {"emotion_intensity": 0}