import os
import json
import re
import time
from google import genai
from google.genai import types
from config import load_api_key

class LLMInterface:
    def __init__(self):
        load_api_key()

        self.api_keys = self._get_ordered_keys()
        if not self.api_keys:
            raise RuntimeError("❌ 사용 가능한 GOOGLE_API_KEY가 없습니다")

        self.current_key_idx = 0
        self.disabled_keys = set()

        # ✅ [업그레이드] 확인된 2.5 모델을 최우선으로, 1.5를 백업으로 설정
        self.model_candidates = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
        ]

        self.client = None
        self._configure_genai()

    def _get_ordered_keys(self):
        keys = []
        primary = os.environ.get("GOOGLE_API_KEY")
        if primary:
            keys.append(primary)

        numbered = []
        for k, v in os.environ.items():
            if k.startswith("GOOGLE_API_KEY_"):
                try:
                    idx = int(k.split("_")[-1])
                    numbered.append((idx, v))
                except ValueError:
                    continue

        numbered.sort()
        keys.extend([v for _, v in numbered if v])
        return keys

    def _configure_genai(self):
        key = self.api_keys[self.current_key_idx]
        self.client = genai.Client(api_key=key)
        print(f"[LLM] 🔑 Key #{self.current_key_idx + 1} 활성화 (남은 키: {len(self.api_keys) - len(self.disabled_keys)})")

    def _rotate_key(self):
        for _ in range(len(self.api_keys)):
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            key = self.api_keys[self.current_key_idx]

            if key not in self.disabled_keys:
                self._configure_genai()
                return

        raise RuntimeError("❌ 모든 API 키가 소진되었습니다. 잠시 후 다시 시도하세요.")

    def analyze_full_session(self, full_transcript):
        # 7만 자 분석 시 2만 자 청크는 매우 적절합니다.
        chunk_size = 20000
        chunks = [
            full_transcript[i:i + chunk_size]
            for i in range(0, len(full_transcript), chunk_size)
        ]

        print(f"[LLM] 📦 {len(full_transcript):,}자 → {len(chunks)}개 파트 분석 시작")
        all_highlights = []

        for i, chunk in enumerate(chunks):
            print(f"\n[LLM] 🔄 파트 {i + 1}/{len(chunks)} 분석 중...")
            
            result = self._run_with_retry(chunk)
            highlights = result.get("highlights", [])

            if highlights:
                all_highlights.extend(highlights)
                print(f"   ✅ {len(highlights)}개 발견 (누적: {len(all_highlights)}개)")
                for h in highlights:
                    print(f"      - {h.get('start')} | {h.get('reason', '')[:40]}")
            else:
                print("   ⚠️ 하이라이트 없음")

            # 무료 티어 안정성을 위해 10초 휴식
            if i < len(chunks) - 1:
                time.sleep(10)

        return {"highlights": all_highlights}

    def _run_with_retry(self, text, max_attempts=3):
        # [업그레이드] 한국어 인식이 더 강력한 프롬프트 사용
        prompt = self._get_highlight_prompt(text)

        for attempt in range(max_attempts):
            for model_name in self.model_candidates:
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.8, # 창의적인 구간 선정을 위해 약간 상향
                        ),
                    )

                    return self._safe_parse_json(response.text)

                except Exception as e:
                    err = str(e).lower()
                    print(f"   ❌ {model_name} 에러: {err[:80]}")

                    if "quota" in err or "429" in err:
                        self.disabled_keys.add(self.api_keys[self.current_key_idx])
                        self._rotate_key()
                        time.sleep(15)
                        break # 다음 키로 재시도

                    if "404" in err:
                        continue # 다음 모델 후보로 시도

                    time.sleep(5)

            if attempt < max_attempts - 1:
                time.sleep(10)

        return {"highlights": []}

    def _get_highlight_prompt(self, transcript):
        """[업그레이드] 2.5의 지능을 활용하는 고성능 프롬프트"""
        return f"""
# ROLE: 전설적인 영상 편집자
당신의 목표는 다음 대본에서 시청자들의 눈을 사로잡을 '최고의 순간' 3~5개를 찾는 것입니다.

# MISSION:
- 타임스탬프를 기반으로 하이라이트 구간(시작~종료)을 선정하세요.
- 선정 이유(reason)는 반드시 한국어로 작성하세요.

# SELECTION CRITERIA:
1. 감정 폭발 (크게 웃거나, 놀라거나, 당황하는 순간)
2. 핵심 정보 (청중이 알아야 할 중요한 인사이트)
3. 스토리 반전 (사건의 흐름이 급변하는 지점)

# TRANSCRIPT:
{transcript}

# OUTPUT FORMAT (JSON ONLY):
{{
  "highlights": [
    {{
      "start": "HH:MM:SS",
      "end": "HH:MM:SS",
      "category": "웃음/정보/충격/교훈",
      "reason": "선정 이유를 짧고 강렬하게 설명",
      "confidence_score": 0.0-1.0
    }}
  ]
}}
"""

    def _safe_parse_json(self, raw_text):
        if not raw_text: return {"highlights": []}
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # 마크다운 및 불필요한 텍스트 제거 강화
            cleaned = re.sub(r"```json\s*|```\s*", "", raw_text.strip())
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try: return json.loads(match.group())
                except: pass
        return {"highlights": []}

# --------------------------------------------------
# 테스트 실행부
# --------------------------------------------------
if __name__ == "__main__":
    llm = LLMInterface()
    sample = "[00:00:10] 와! 진짜 대박이다! [00:01:20] 여기서부터가 진짜 중요한 내용이에요."
    print(llm.analyze_full_session(sample))
