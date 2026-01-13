import os
import json
import time
import re
import subprocess
from pathlib import Path
from google import genai
from google.genai import types
from config import load_api_key, get_all_api_keys
from presets.factory_config import FactoryConfig
from .llm_interface import LLMInterface
from .knowledge_base import VideoKnowledgeBase
from .visual_scout import VisualScout
from .rejection_analyst import RejectionAnalyst

class StoryArchitect:
    def __init__(self):
        load_api_key()
        self.api_keys = get_all_api_keys()
        self.model_name = "gemini-1.5-flash-latest" # ✅ V5: 404 해결을 위한 Fallback ID
        self.client = genai.Client(api_key=self.api_keys[0])
        self.rejection_log = Path("rejected_stories.json")
        self.scout = VisualScout()
        self.ra = RejectionAnalyst()

    def _safe_parse_json(self, raw_text):
        """
        ✅ FIX: 강화된 JSON 파싱 (마크다운 코드블록, 주석, 불완전한 JSON 처리)
        """
        if not raw_text:
            return None
        
        # 1. 마크다운 코드블록 제거 (```json ... ``` 또는 ``` ... ```)
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```\s*', '', text)
        
        # 2. JSON 객체/배열 추출 시도
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            text = json_match.group(1)
        
        # 3. 주석 제거 (// ... 또는 /* ... */)
        text = re.sub(r'//.*?\n', '\n', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # 4. 파싱 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[StoryArchitect] ⚠️ JSON Parse Error at position {e.pos}: {e.msg}")
            print(f"[StoryArchitect] 📄 Raw response preview: {raw_text[:500]}...")
            
            # 5. 마지막 시도: 불완전한 JSON 수정 (마지막 쉼표, 닫히지 않은 괄호)
            try:
                # 마지막 쉼표 제거
                text = re.sub(r',\s*([\]}])', r'\1', text)
                # 닫히지 않은 배열/객체 닫기
                if text.count('{') > text.count('}'):
                    text += '}' * (text.count('{') - text.count('}'))
                if text.count('[') > text.count(']'):
                    text += ']' * (text.count('[') - text.count(']'))
                return json.loads(text)
            except:
                return None

    def _summarize_transcript_in_chunks(self, transcript, video_id, max_chunk_chars=20000):
        """
        긴 대본을 청크로 나눠 각각 요약 후 병합 (12시간+ 장편 영상 대응)
        """
        from google.genai import types
        import time
        
        # 1. 청크 분할 (줄바꿈 기준)
        chunks = []
        lines = transcript.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_len = len(line)
            if current_length + line_len > max_chunk_chars and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_len
            else:
                current_chunk.append(line)
                current_length += line_len
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        print(f"[StoryArchitect] 📚 Split into {len(chunks)} chunks for summarization")
        
        # 2. 각 청크에서 이벤트 추출
        all_events = []
        for i, chunk in enumerate(chunks):
            try:
                prompt = f"""
영상 대본에서 중요한 '상태 변화'만 로그 형태로 추출하라.
대본을 요약하거나 설명하지 마십시오.

# 추출 대상 (Event Types):
1. Game Start/End/Reset: 게임의 시작, 종료, 재시도 지점
2. Topic Change: 대화 주제나 방송 분위기가 급격히 변하는 지점
3. High Reaction: 비명, 큰 폭소, 분노 등 감정이 폭발하는 지점

# 대본 청크 #{i+1}/{len(chunks)}:
{chunk[:8000]}

# OUTPUT FORMAT (JSON Array):
[
  {{"time": float, "event": "이벤트 종류", "importance": 1~10}}
]
반드시 유효한 JSON 배열만 출력하십시오.
"""
                # ✅ FIX: 호출 간격 2초 (V5 가이드)
                time.sleep(2)
                
                print(f"DEBUG: StoryArchitect Current Model ID -> {self.model_name}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
                )
                
                chunk_events = self._safe_parse_json(response.text)
                if isinstance(chunk_events, list):
                    all_events.extend(chunk_events)
                    print(f"   ✅ Chunk {i+1}/{len(chunks)} extracted {len(chunk_events)} events")
                else:
                    print(f"   ⚠️ Chunk {i+1} returned invalid format, skipping")
            except Exception as e:
                print(f"   ⚠️ Chunk {i+1} failed: {e}")
                # ✅ V5: 하나라도 실패해도 중단하지 않고 다음 청크로 진행
                continue
        
        if not all_events:
            print("[StoryArchitect] ❌ All chunks failed to extract events. Returning empty list.")
        else:
            print(f"[StoryArchitect] ✅ Extracted {len(all_events)} total events from successfully processed chunks")
        
        return all_events

    def segment_and_rank(self, video_path, transcript, top_n=3, video_id="temp", visual_data=None):
        """
        [Macro Architect] 전체 대본 + 시각적 에너지를 분석하여 '판'을 나누고 랭킹을 매깁니다.
        """
        print(f"\n🏰 [Macro Architect] Starting Narrative Search for '{video_id}'...")
        
        # 1. Visual Analysis
        if visual_data is None and not getattr(FactoryConfig, 'SKIP_VISUAL', False):
            visual_data = self.scout.analyze_video(video_path)
            
        visual_peaks = self._get_high_energy_points(visual_data) if visual_data else []
        peaks_str = ", ".join([f"{p['time']:.1f}s({p['type']})" for p in visual_peaks[:15]])

        # ✅ FIX: 대본 길이 제한 (청크 요약 방식)
        max_transcript_len = 30000  # ~10k tokens
        # ✅ FIX: 대본 길이 제한 (V5: 이벤트 로그 방식)
        if len(transcript) > 20000:
            print(f"[StoryArchitect] 📚 Transcript too long ({len(transcript)} chars)")
            print(f"[StoryArchitect] 🔄 Extracting Event Logs from chunks...")
            event_logs_str = self._summarize_transcript_in_chunks(transcript, video_id)
        else:
            event_logs_str = transcript # 짧으면 그대로 (거의 없음)

        prompt = f"""
# ROLE: 고도로 숙련된 영상 스토리 편집자
# TASK: 추출된 '이벤트 로그'와 '시각적 피크' 정보를 바탕으로, 영상의 거시적 구조를 '판(Match)' 또는 '챕터'로 분할하라.

# 추출된 이벤트 로그 (Event Logs - V5):
{event_logs_str}

# 시각적 강조 지점 (Visual High-Energy Points):
{peaks_str if peaks_str else "No major visual peaks detected."}

# 🧠 LESSONS FROM THE PAST:
{self.ra.get_editing_feedback(limit=5)}

# CRITICAL: 이벤트 로그의 발생 시간을 기준으로 챕터의 시작과 끝을 정하십시오. 
# 반드시 유효한 JSON만 출력하고, 주석이나 마크다운은 금지합니다.

# OUTPUT FORMAT (Pure JSON Only):
{{
  "chapters": [
    {{
      "id": 1,
      "title": "챕터 제목",
      "start_time": 0.0,
      "end_time": 600.0,
      "narrative_score": 9.5,
      "summary": "서사적 요약 및 이벤트 근거",
      "is_boring": false
    }}
  ]
}}
"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # ✅ FIX: 호출 간격 2초 (V5 가이드)
                time.sleep(2)
                
                print(f"DEBUG: StoryArchitect Segment Current Model ID -> {self.model_name}")
                # ✅ FIX: temperature 낮춰서 JSON 형식 준수율 향상
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2  # 0.4 -> 0.2로 낮춤
                    )
                )
                
                # ✅ FIX: 안전한 파싱
                result = self._safe_parse_json(response.text)
                
                if result is None:
                    raise ValueError("Failed to parse JSON response")
                
                all_chapters = result.get('chapters', [])
                
                if not all_chapters:
                    print(f"[StoryArchitect] ⚠️ No chapters returned (Attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return []  # Fallback으로 넘김
                
                # 1. Logging Rejections
                self._log_rejections(all_chapters)
                
                # 2. Ranking & Filtering
                valid_chapters = [c for c in all_chapters if not c.get('is_boring', False)]
                
                if not valid_chapters:
                    print(f"[StoryArchitect] ⚠️ All chapters marked as boring!")
                    # 그래도 상위 N개는 가져가기
                    valid_chapters = sorted(all_chapters, key=lambda x: x.get('narrative_score', 0), reverse=True)[:top_n]
                
                ranked = sorted(valid_chapters, key=lambda x: x.get('narrative_score', 0), reverse=True)
                selection = ranked[:top_n]
                
                # 각 챕터에 시각적 데이터 붙이기
                for s in selection: 
                    s['visual_context'] = visual_data
                
                print(f"✅ Selected TOP {len(selection)} Macro-Chapters based on Narrative & Visual Ranking.")
                return selection
                
            except Exception as e:
                print(f"🚨 Macro-Segmentation 오류 (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"   🔄 Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    print(f"   ❌ Max retries reached. Falling back to signal analysis.")
                    return []

    def _get_high_energy_points(self, visual_data, threshold=0.7):
        """시각적 에너지가 높은 지점들을 추출"""
        if not visual_data: return []
        peaks = []
        energy = visual_data['visual_energy']
        times = visual_data['times']
        
        for i in range(1, len(energy)-1):
            if energy[i] > threshold and energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                p_type = "Action" if visual_data['motion'][i] > visual_data['entropy'][i] else "Detail"
                peaks.append({"time": float(times[i]), "score": float(energy[i]), "type": p_type})
        
        return sorted(peaks, key=lambda x: x['score'], reverse=True)

    def identify_micro_points(self, chapter_data, chapter_transcript):
        """
        [Micro Editor]
        선정된 챕터 내부에서 핵심 포인트(시작/위기/결과)를 찍습니다.
        """
        print(f"   🎬 Micro-Editing Match: '{chapter_data['title']}'...")
        
        # ✅ FIX: 챕터 시간 범위 명시
        c_start = float(chapter_data.get('start_time', 0))
        c_end = float(chapter_data.get('end_time', c_start + 600))
        
        prompt = f"""
# ROLE: 영상 컷 편집 전문가
# TASK: 아래 챕터의 대본을 분석하여 서사를 구성하는 3대 핵심 지점과 브릿지 자막을 생성하라.

# CHAPTER INFO:
- Title: {chapter_data['title']}
- Time Range: {c_start:.1f}s ~ {c_end:.1f}s
- Context: {chapter_data['summary']}

# CHAPTER TRANSCRIPT:
{chapter_transcript[:5000]}

# CRITICAL RULES:
1. **시간(time)은 반드시 챕터 범위 {c_start:.1f}~{c_end:.1f} 내의 초 단위 숫자여야 함**
2. **반드시 유효한 JSON만 출력**
3. **세 개의 포인트(start, crisis, result)를 모두 포함할 것**

# OUTPUT FORMAT (Pure JSON Only):
{{
  "points": {{
    "start": {{"time": {c_start + 10}, "reason": "상황의 시작"}},
    "crisis": {{"time": {(c_start + c_end) / 2}, "reason": "갈등의 절정"}},
    "result": {{"time": {c_end - 10}, "reason": "최종 결과"}}
  }},
  "bridge_text": "연결 메시지"
}}
"""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # ✅ FIX: 호출 간격 2초 (V5 가이드)
                time.sleep(2)
                
                print(f"DEBUG: StoryArchitect Micro Current Model ID -> {self.model_name}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                
                result = self._safe_parse_json(response.text)
                
                # ✅ FIX: 결과 검증
                if result is None:
                    print(f"   ⚠️ Failed to parse JSON (Attempt {attempt+1}/{max_retries})")
                    continue
                
                if 'points' not in result:
                    print(f"   ⚠️ No 'points' key in response (Attempt {attempt+1}/{max_retries})")
                    continue
                
                points = result['points']
                if not all(k in points for k in ['start', 'crisis', 'result']):
                    print(f"   ⚠️ Missing required points (Attempt {attempt+1}/{max_retries})")
                    continue
                
                # 성공!
                return result
                
            except Exception as e:
                print(f"   🚨 Micro-Editing 오류 (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5) # 재시도 시 충분한 대기 시간 확보
        
        # ✅ FIX: 모든 재시도 실패 시 None 리턴 (오염 방지 및 factory_main에서 안전하게 스킵 유도)
        return None

    def _log_rejections(self, chapters):
        """버려진 판들을 기록"""
        rejections = [c for c in chapters if c.get('is_boring')]
        if not rejections: return

        existing_data = []
        if self.rejection_log.exists():
            try:
                with open(self.rejection_log, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except: 
                existing_data = []
            
        for r in rejections:
            r['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            existing_data.append(r)
            
        try:
            with open(self.rejection_log, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            print(f"[StoryArchitect] 🗑️ Logged {len(rejections)} rejected chapters for future learning.")
        except Exception as e:
            print(f"[StoryArchitect] ⚠️ Failed to save rejection log: {e}")
