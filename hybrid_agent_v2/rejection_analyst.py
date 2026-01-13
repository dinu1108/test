
import json
from pathlib import Path
from collections import Counter, deque

class RejectionAnalyst:
    def __init__(self, macro_log="rejected_stories.json", micro_log="hybrid_agent_v2/chroma_db/rejection_logs.jsonl"):
        self.macro_log = Path(macro_log)
        self.micro_log = Path(micro_log)

    def get_editing_feedback(self, limit=10):
        """
        최근 거절된 사례들을 분석하여 편집 안목에 대한 피드백을 생성합니다.
        
        개선사항:
        1. 강화된 예외 처리
        2. 데이터 검증 로직 추가
        3. 메모리 효율적인 로그 읽기
        4. 상세한 에러 로깅
        """
        insights = []
        
        # 1. Macro Analysis (Chapters)
        macro_insights = self._analyze_macro_rejections(limit)
        if macro_insights:
            insights.append(macro_insights)

        # 2. Micro Analysis (Candidates)
        micro_insights = self._analyze_micro_rejections(limit)
        if micro_insights:
            insights.append(micro_insights)

        if not insights:
            return "과거 분석 이력이 충분하지 않습니다. 현재 설정된 스타일 가이드에 충실하세요."
            
        feedback = "\n".join(insights)
        feedback += "\n\n[지침] 위 사례들을 참고하여 이번 영상에서는 서사적 가치가 더 높거나 시각적으로 임팩트 있는 구간을 더 공격적으로 발굴하세요."
        return feedback

    def _analyze_macro_rejections(self, limit):
        """매크로 거절 로그 분석 (개선된 에러 처리)"""
        if not self.macro_log.exists():
            return None
        
        # 파일 크기 체크 (0바이트 방어)
        if self.macro_log.stat().st_size == 0:
            print(f"[RejectionAnalyst] ⚠️ Macro log is empty: {self.macro_log}")
            return None

        try:
            with open(self.macro_log, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 데이터 타입 검증
            if not isinstance(data, list):
                print(f"[RejectionAnalyst] ⚠️ Unexpected macro log format (expected list, got {type(data).__name__})")
                return None
                
            if len(data) == 0:
                return None
                
            recent = data[-limit:] if len(data) >= limit else data
            
            # 안전한 데이터 추출 (기본값 처리)
            reasons = []
            for item in recent:
                if isinstance(item, dict):
                    reasons.append(item.get('rejection_reason', '지루함'))
            
            if not reasons:
                return None
                
            common = Counter(reasons).most_common(1)[0][0]
            return f"최근 매크로 단계에서 '{common}' 등의 이유로 많은 구간이 제외되었습니다."
            
        except json.JSONDecodeError as e:
            print(f"[RejectionAnalyst] ❌ Macro log JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"[RejectionAnalyst] ❌ Unexpected error reading macro log: {e}")
            return None

    def _analyze_micro_rejections(self, limit):
        """마이크로 거절 로그 분석 (메모리 효율 개선)"""
        if not self.micro_log.exists():
            return None
            
        if self.micro_log.stat().st_size == 0:
            print(f"[RejectionAnalyst] ⚠️ Micro log is empty: {self.micro_log}")
            return None

        try:
            # 메모리 효율적인 tail 읽기 (deque 사용)
            recent_lines = deque(maxlen=limit)
            
            with open(self.micro_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # 빈 줄 무시
                        recent_lines.append(line)
            
            if not recent_lines:
                return None
            
            reasons = []
            for line in recent_lines:
                try:
                    log = json.loads(line)
                    if isinstance(log, dict):
                        reasons.append(log.get('reason', '점수 미달'))
                except json.JSONDecodeError:
                    # 손상된 라인은 건너뛰기
                    continue
            
            if not reasons:
                return None
                
            common = Counter(reasons).most_common(1)[0][0]
            return f"최근 개별 클립 선정 시 '{common}' 판단이 많았습니다."
            
        except Exception as e:
            print(f"[RejectionAnalyst] ❌ Error reading micro log: {e}")
            return None

    def get_detailed_stats(self):
        """
        [추가 기능] 거절 통계 상세 분석
        """
        stats = {
            "macro_total": 0,
            "micro_total": 0,
            "top_macro_reasons": [],
            "top_micro_reasons": []
        }
        
        # Macro 통계
        if self.macro_log.exists() and self.macro_log.stat().st_size > 0:
            try:
                with open(self.macro_log, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        stats["macro_total"] = len(data)
                        reasons = [item.get('rejection_reason', 'Unknown') for item in data if isinstance(item, dict)]
                        stats["top_macro_reasons"] = Counter(reasons).most_common(5)
            except: pass

        # Micro 통계 (스트리밍 방식)
        if self.micro_log.exists() and self.micro_log.stat().st_size > 0:
            try:
                reasons = []
                with open(self.micro_log, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            log = json.loads(line.strip())
                            stats["micro_total"] += 1
                            if isinstance(log, dict):
                                reasons.append(log.get('reason', 'Unknown'))
                        except: continue
                stats["top_micro_reasons"] = Counter(reasons).most_common(5)
            except: pass
        
        return stats
