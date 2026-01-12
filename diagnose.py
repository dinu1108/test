import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd()))

from hybrid_agent_v2.knowledge_base import VideoKnowledgeBase
from hybrid_agent_v2.llm_interface import LLMInterface
import chromadb

def diagnose_transcript(video_id="test"):
    """대본 품질을 진단하고 샘플을 출력합니다"""
    print("="*60)
    print("📊 TRANSCRIPT QUALITY DIAGNOSIS")
    print("="*60)
    
    kb = VideoKnowledgeBase()
    
    # 1. 전체 대본 가져오기
    full_text = kb.get_full_transcript(video_id)
    
    print(f"\n[1] 대본 기본 정보:")
    print(f"   - 총 길이: {len(full_text):,} 글자")
    print(f"   - 단어 수: {len(full_text.split()):,} 단어")
    print(f"   - 예상 분석 청크: {(len(full_text) // 15000) + 1}개")
    
    # 2. 타임스탬프 포함 여부 확인
    has_timestamps = "[" in full_text and "]" in full_text
    print(f"   - 타임스탬프 포함: {'✅ Yes' if has_timestamps else '❌ No'}")
    
    # 3. 첫 1000자 샘플 출력
    print(f"\n[2] 대본 앞부분 샘플 (처음 1000자):")
    print("-"*60)
    print(full_text[:1000])
    print("-"*60)
    
    # 4. ChromaDB에 저장된 세그먼트 확인
    try:
        # KnowledgeBase의 collection을 직접 사용하거나 client로 접근
        # 여기서는 KB 내부의 client 사용 (이미 persistent path 설정됨)
        # collection 이름 규칙은 knowledge_base.py에 따름 (기본: "video_memory")
        # 하지만 get_full_transcript는 video_id로 필터링하므로 전체 collection에서 count해야 함
        
        # 특정 비디오 ID에 대한 세그먼트 수 카운트
        results = kb.collection.get(where={"video_id": video_id})
        count = len(results['ids'])
        
        print(f"\n[3] ChromaDB 저장 상태:")
        print(f"   - 저장된 세그먼트 수: {count}개")
        
        # 샘플 세그먼트 확인
        if count > 0:
            print(f"\n[4] 샘플 세그먼트 (최초 3개):")
            for i in range(min(3, count)):
                doc = results['documents'][i]
                meta = results['metadatas'][i]
                print(f"\n   세그먼트 #{i+1}:")
                print(f"   시작: {meta.get('start', 'N/A')}s")
                print(f"   내용: {doc[:200]}...")
    except Exception as e:
        print(f"\n[3] ChromaDB 확인 실패: {e}")
    
    # 5. 타임스탬프 형식 분석
    print(f"\n[5] 타임스탬프 형식 분석:")
    import re
    
    # 다양한 타임스탬프 패턴 찾기
    patterns = {
        "HH:MM:SS": r'\d{2}:\d{2}:\d{2}',
        "MM:SS": r'\d{1,2}:\d{2}',
        "[HH:MM:SS]": r'\[\d{2}:\d{2}:\d{2}\]',
        "숫자만": r'\d+\.\d+초'
    }
    
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, full_text[:5000])
        if matches:
            print(f"   ✅ {pattern_name} 형식 발견: {matches[:3]}")
    
    # 6. AI가 분석할 수 있는 형태인지 확인
    print(f"\n[6] AI 분석 가능성 평가:")
    
    checks = {
        "충분한 길이 (1000자 이상)": len(full_text) >= 1000,
        "타임스탬프 존재": has_timestamps,
        "대화 내용 존재": any(word in full_text for word in ['말', '얘기', '이야기', 'said', 'talk', '어', '아', '네', '요']),
        "빈 대본 아님": full_text.strip() != ""
    }
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
    
    # 7. 권장사항
    print(f"\n[7] 권장 조치:")
    if not has_timestamps:
        print("   ⚠️ 타임스탬프가 없습니다. Whisper 받아쓰기 재실행 필요")
    if len(full_text) < 1000:
        print("   ⚠️ 대본이 너무 짧습니다. 영상 길이 확인 필요")
    if all(checks.values()):
        print("   ✅ 대본 품질은 정상입니다. LLM 프롬프트 개선 필요")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    diagnosis_target = "test"
    if len(sys.argv) > 1:
        diagnosis_target = sys.argv[1]
    diagnose_transcript(diagnosis_target)
