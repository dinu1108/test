import sys
import os
from pathlib import Path
import argparse

# 현재 실행 파일의 위치를 기준으로 경로 설정 (모듈 로딩 최적화)
base_dir = Path(__file__).parent.absolute()
sys.path.append(str(base_dir))

# 하이브리드 에이전트 모듈 불러오기
try:
    from hybrid_agent_v2.knowledge_base import VideoKnowledgeBase
    from hybrid_agent_v2.llm_interface import LLMInterface
    from hybrid_agent_v2.fast_cutter import FastCutter
    print("[System] ✅ All modules loaded successfully.")
except ImportError as e:
    print(f"[System] ❌ Module Import Error: {e}")
    sys.exit(1)

from presets.factory_config import FactoryConfig

def finalize_clips(base_candidates, llm_evals, kb, video_id):
    final_selections = []
    
    # 평가 데이터를 ID 매핑
    eval_map = {e['id']: e for e in llm_evals.get('evaluations', [])}

    print(f"\n📊 [Scoring] Calculating Golden Scores for {len(base_candidates)} candidates...")
    
    last_end_time = -999 # 연속성 체크용
    
    for i, candidate in enumerate(base_candidates):
        evaluation = eval_map.get(i)
        if not evaluation:
            print(f"   Skip #{i}: No evaluation data.")
            kb.log_rejection(video_id, candidate, "No Evaluation Data", 0.0)
            continue

        # 🎯 공장장님표 황금 스코어 공식 적용 (Config Based)
        base_score = candidate.get('score', 0.7)
        
        e_intensity = evaluation.get('emotion_intensity', 0.5)
        i_density = evaluation.get('info_density', 0.5)
        c_break = evaluation.get('context_break', 0.5) 
        
        w = FactoryConfig.WEIGHTS
        final_score = (
            base_score * w['base'] +
            e_intensity * w['emotion'] +
            i_density * w['info'] -
            c_break * w['context_break']
        )

        # 🔗 [Rule] 연속성 패널티
        gap = candidate['start'] - last_end_time
        if last_end_time > 0 and gap > FactoryConfig.CONTINUITY_GAP:
            final_score -= 0.1
            print(f"     -> 📉 Continuity Penalty (-0.1) applied. Gap: {gap:.1f}s")

        # 감점 요인 적용
        if evaluation.get('is_unnecessary', False):
            final_score -= 0.3

        reason = evaluation.get('reason', 'N/A')
        print(f"   🎬 컷 #{i} | 최종: {final_score:.2f} (기본:{base_score:.1f} 감정:{e_intensity}) | {reason}")

        # Config Threshold
        if final_score > FactoryConfig.GOLDEN_SCORE_THRESHOLD:
            final_selections.append({
                **candidate,
                "score": final_score,
                "reason": reason
            })
            last_end_time = candidate['end']
        else:
            kb.log_rejection(video_id, candidate, f"Score Low: {final_score:.2f} (E:{e_intensity}) - {reason}", final_score)

    return final_selections

def run_factory(video_path):
    print(f"🏭 [Factory Mode] Starting Line for: {video_path}")
    
    video_file = Path(video_path)
    if not video_file.exists():
        potential_path = base_dir / "raw_data" / video_file.name
        if potential_path.exists(): video_file = potential_path
        else:
            print(f"❌ [Error] Video file not found: {video_path}")
            return

    video_id = video_file.stem
    
    # 1. Analyst V1 (Signal Processing) - Get Candidates
    print("\n🔍 [Stage 1] Signal Analysis (Analyst V1)...")
    from modules.analyst import Analyst
    analyst = Analyst()
    
    # Analyze Audio/Video Signals
    audio_data = analyst.analyze_audio_advanced(str(video_file))
    if not audio_data:
        print("❌ Audio analysis failed.")
        return

    scores, times = analyst.calculate_scores(audio_data)
    
    # Simple Peak Picking for Base Candidates
    import numpy as np
    indices = np.argsort(scores)[::-1][:15] # Top 15 Peaks
    indices.sort()
    
    base_candidates = []
    # KB Load for Transcripts
    kb = VideoKnowledgeBase()
    # Ensure Ingest
    kb.ingest(str(video_file))
    
    print(f"   -> Extracting Transcript Contexts for {len(indices)} peaks...")
    
    for idx_i, idx in enumerate(indices):
        t = times[idx]
        # Get context 30s window (Initial snippet)
        ctx = kb.get_context(video_id, max(0, t-15), t+15)
        txt = " ".join([c['text'] for c in ctx])
        base_candidates.append({
            "id": idx_i, # ID for mapping
            "start": max(0, t-15),
            "end": t+15,
            "score": float(scores[idx]),
            "text": txt
        })

    # 2. LLM Evaluation (Golden Score Metrics) with Context
    print(f"\n🧠 [Stage 2] LLM Precision Evaluation (Gemini 2.5)...")
    llm = LLMInterface()
    # Removed explicit full_text fetch, handled inside evaluate_candidates contextually
    
    eval_results = llm.evaluate_candidates(kb, video_id, base_candidates)
    
    # 3. Final Selection (Golden Score Logic)
    valid_clips = finalize_clips(base_candidates, eval_results, kb, video_id)

    
    print(f"\n📋 [Plan] Final Approved Cuts: {len(valid_clips)}")

    # 4. Rendering
    if valid_clips:
        print(f"\n⚙️ [Production] Rendering {len(valid_clips)} Clips...")
        cutter = FastCutter()
        cutter.cut_clips(str(video_file), valid_clips)
        print("\n✨ [Factory] Complete!")
    else:
        print("\n❌ No clips survived the Golden Score threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Video Highlight Factory V3")
    parser.add_argument("video_path", help="Path to the source video file")
    parser.add_argument("--preset", help="Path to style preset JSON (e.g., presets/isyse.json)", default=None)
    args = parser.parse_args()
    
    if args.preset:
        FactoryConfig.load_preset(args.preset)

    try:
        run_factory(args.video_path)
    except KeyboardInterrupt:
        print("\n🛑 [System] Factory stopped by user.")
    except Exception as e:
        print(f"\n🚨 [Critical Error] {e}")
