# Auto Highlight Extractor (Hybrid AI Factory)

방송 영상학습 기반 자동 하이라이트 추출기입니다.
**Hybrid AI (Gemini 2.5) + Factory Automation** 아키텍처를 도입하여 18시간 분량의 초대형 영상을 "단 한 번의 호출"로 완벽하게 요약합니다.

---

## 🏭 Factory Mode (V3) - **Current Standard**

**"Analyst(분석) + LLM(평가) + Golden Score(확정)"**로 이어지는 정밀 공정 시스템입니다.

### 1️⃣ 작동 원리 (The Pipeline)

#### **Step 1: Signal Analysis (Analyst V1)**
- **역할**: 1차 후보군 신속 추출
- **방식**: 오디오 파형(RMS, Slope, ZCR)을 분석하여 에너지 피크 상위 15개를 선정합니다.
- **목적**: LLM 토큰 절약 및 물리적으로 확실한(소리지르는) 구간 확보

#### **Step 2: LLM Precision Evaluation (Gemini 2.5)**
- **역할**: 정밀 심사 (판사 역할)
- **방식**: 선정된 15개 후보의 대본을 읽고 다음 4가지 지표(0.0~1.0)를 평가합니다.
    1.  **emotion_intensity**: 감정 폭발력 (재미)
    2.  **info_density**: 정보 밀도
    3.  **context_break**: 맥락 단절 여부 (감점 요인)
    4.  **is_unnecessary**: 불필요 구간 여부
- **Constraint**: **새로운 타임라인 생성 금지** (오직 평가만 수행)

#### **Step 3: Golden Score Logic (공장장 공식)**
- **역할**: 최종 확정 및 튜닝
- **공식**: `(Base*0.4) + (Emotion*0.4) + (Info*0.2) - (Break*0.2)`
- **Rule**:
    - **Threshold**: 최종 점수 **0.55** 이상 통과 (재미 위주 완화)
    - **Penalty**: 이전 컷과 3초 이상 끊기면 **-0.1점** 감점 (연속성 유도)
    - **Logging**: 탈락한 컷은 사유와 점수를 `rejection_logs.jsonl`에 기록 (피드백 루프)

#### **Step 4: Fast Production (FFmpeg)**
- **역할**: 물리적 파일 생성
- **기능**: Audio Fade In/Out 적용, GPU 가속 렌더링

### 2️⃣ 실행 방법
```bash
# 1. API 키 확인 (.env 파일 자동 로드)
# 2. 공장 가동
python factory_main.py "raw_data/파일이름.mp4"
```

---

## 📂 Folder Structure (폴더 구조)

```text
auto_highlight_extractor/
├── factory_main.py        # [V3] 공장 모드 실행 (★Main)
├── main_v2.py             # [V2] 대화형 에이전트 실행
├── main.py                # [V1] 로컬 모드 실행
├── .env                   # [보안] API 키 저장소
│
├── hybrid_agent_v2/       # [Core] 하이브리드 엔진
│   ├── llm_interface.py   # Gemini 2.5 평가자 (Score & Reason)
│   ├── fast_cutter.py     # FFmpeg 렌더링
│   ├── knowledge_base.py  # ChromaDB & Rejection Logging
│   └── chroma_db/         # [Log] rejection_logs.jsonl (실패 기록)
│
├── modules/               # [Module] 기능 모듈
│   ├── analyst.py         # [Signal] V1 오디오 분석기
│   ├── dinu_test/         # [Ext] 외부 테스트 모듈 (git clone)
│   └── collector.py       # 다운로더
│
└── clips/                 # [Result] 결과물이 저장되는 곳
```
