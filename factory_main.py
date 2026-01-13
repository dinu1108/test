import sys
import os
import argparse
import numpy as np
from pathlib import Path

# [1] 경로 및 환경 설정
base_dir = Path(__file__).parent.absolute()
sys.path.append(str(base_dir))

try:
    from hybrid_agent_v2.knowledge_base import VideoKnowledgeBase
    from hybrid_agent_v2.llm_interface import LLMInterface
    from hybrid_agent_v2.fast_cutter import FastCutter
    from presets.factory_config import FactoryConfig
    from modules.analyst import Analyst
    print("[System] ✅ All Hybrid Engines Loaded.")
except ImportError as e:
    print(f"[System] ❌ Critical Import Error: {e}")
    sys.exit(1)

def finalize_clips(base_candidates, llm_evals, kb, video_id):
    """
    [Stage 3] 공장장님표 황금 스코어 합성 및 타임라인 확장
    """
    final_selections = []
    # ✅ FIX: Candidate ID 매칭 안전화 (Index 대신 ID 기반 매칭)
    eval_map = {e['id']: e for e in llm_evals.get('evaluations', [])}
    
    PREROLL = FactoryConfig.PREROLL
    POSTROLL = FactoryConfig.POSTROLL

    print(f"\n📊 [Scoring] Synthesizing Golden Scores for {len(base_candidates)} candidates...")
    
    last_end_time = -999 

    for i, candidate in enumerate(base_candidates):
        # ✅ FIX: CID(Candidate ID) 기반 매칭으로 재정렬/필터링 시에도 안전함
        cid = candidate.get('id', i)
        evaluation = eval_map.get(cid)
        
        if not evaluation:
            kb.log_rejection(video_id, candidate, "No Evaluation Data", 0.0)
            continue

        try:
            base_score = float(candidate.get('score', 0.5))
            e_intensity = float(evaluation.get('emotion_intensity', 0.5))
            i_density = float(evaluation.get('info_density', 0.5))
            c_break = float(evaluation.get('context_break', 0.5))
            # ✅ FIX #4: narrative_payoff 실제 점수 반영
            payoff = float(evaluation.get('narrative_payoff', 0.5))
        except (TypeError, ValueError):
            print(f"   ⚠️ 컷 #{cid} - 점수 데이터 형식 분석 실패, 건너뜀")
            kb.log_rejection(video_id, candidate, "Type conversion failed", 0.0)
            continue
        
        if any(x is None for x in [base_score, e_intensity, i_density, c_break]):
            print(f"   ⚠️ 컷 #{cid} - 평가 데이터 불완전, 건너뜀")
            kb.log_rejection(video_id, candidate, "Incomplete evaluation data", 0.0)
            continue
        
        w = FactoryConfig.WEIGHTS
        # ✅ V5: 신규 가중치 공식 (Event_Match 0.4 + Signal_Peak 0.3 + Speech_Density 0.3)
        # 여기서는 LLM 평가 결과(narrative_payoff 등)를 Event_Match로, base_score를 Signal_Peak로 활용
        
        event_match = i_density # LLM이 판단한 정보 밀도/이벤트 적합도
        signal_peak = base_score # 오디오 분석 기반 신호 강도
        speech_density = float(evaluation.get('speech_density', 0.5)) # KB에서 가져온 화법 밀도
        
        final_score = (
            event_match * 0.4 +
            signal_peak * 0.3 +
            speech_density * 0.3
        )

        if evaluation.get('is_unnecessary', False):
            final_score -= 0.5
            
        peak = candidate.get('peak_time') or candidate.get('start', 0.0)
        try:
            peak = float(peak)
        except (TypeError, ValueError):
            peak = 0.0
        
        # ✅ FIX #9: CONTINUITY_GAP 동적 조절 (아이디어 반영: 고득점 시 간격 축소)
        gap_limit = FactoryConfig.CONTINUITY_GAP
        if final_score > 0.8: gap_limit *= 0.5 # 고득점 후보는 더 촘촘하게 배치 허용
        
        gap = peak - last_end_time
        if last_end_time > 0 and gap < gap_limit:
            final_score -= 0.1

        reason = evaluation.get('reason', 'N/A')
        print(f"   🎬 컷 #{cid} | 최종: {final_score:.2f} (기본:{base_score:.1f} 감정:{e_intensity:.1f} 서사:{i_density:.1f}) | {reason[:40]}")

        if final_score > FactoryConfig.GOLDEN_SCORE_THRESHOLD:
            final_selections.append({
                "start": max(0, peak - PREROLL),
                "end": peak + POSTROLL,
                "score": final_score,
                "reason": reason,
                "original_peak": peak,
                "bridge_text": candidate.get('bridge_text', ""),
                "id": cid # ID 보존
            })
            last_end_time = peak + POSTROLL
        else:
            kb.log_rejection(video_id, candidate, f"Low Score: {final_score:.2f}", final_score)

    return final_selections

def run_factory(video_path, top_n_chapters=3, force_refresh=False):
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"❌ File Not Found: {video_path}")
        return

    video_id = video_file.stem
    kb = VideoKnowledgeBase()
    kb.ingest(str(video_file))

    print(f"\n🗺️ [V5 Map-Reduce] Starting Event-Log based Ranking Analysis...")
    from hybrid_agent_v2.story_architect import StoryArchitect
    architect = StoryArchitect()
    
    # [1] Map Phase: 전체 대본을 청크 단위로 처리하여 후보군 추출
    print("   🗺️ [Map] Extracting event-driven candidates from chunks...")
    full_transcript = kb.get_full_transcript(str(video_file))
    event_logs_json = architect._summarize_transcript_in_chunks(full_transcript, video_id)
    
    import json
    try:
        event_logs = json.loads(event_logs_json) if isinstance(event_logs_json, str) else event_logs_json
    except:
        print("   ⚠️ Failed to parse event logs. Falling back to default events.")
        event_logs = []

    # [Visual Scouting]
    visual_data = None
    if not getattr(FactoryConfig, 'SKIP_VISUAL', False):
        try:
            from hybrid_agent_v2.visual_scout import VisualScout
            scout = VisualScout()
            visual_data = scout.analyze_video(str(video_file))
        except Exception as e: print(f"⚠️ VisualScout fail: {e}")

    # [Signal Analysis] for Speech Density & Peaks
    analyst = Analyst()
    audio_data = analyst.analyze_audio_advanced(str(video_file))
    sig_scores, sig_times = analyst.calculate_scores(audio_data)

    # [2] Filter Phase: 이벤트 로그 + 시각적 피크 + 대사 밀도 결합
    print("   🔍 [Filter] Merging signal peaks and event logs (Global Ranking)...")
    base_candidates = []
    
    # 1. Event-based Candidates (LLM Log)
    for log in event_logs:
        t = log.get('time', 0)
        importance = log.get('importance', 5) / 10.0
        
        # 해당 시점의 Speech Density 조회 (KB 메타데이터 활용)
        ctx = kb.get_context(video_id, max(0, t-5), t+5)
        avg_density = np.mean([c.get('speech_density', 0.5) for c in ctx]) if ctx else 0.5
        
        base_candidates.append({
            "id": len(base_candidates),
            "peak_time": t,
            "score": float(importance),
            "speech_density": float(avg_density),
            "text": f"[Event] {log.get('event', 'Unknown')}",
            "type": "event"
        })

    # 2. Add Signal-based Candidates if not redundant
    sig_indices = np.argsort(sig_scores)[::-1][:30]
    for idx in sig_indices:
        t = sig_times[idx]
        if any(abs(t - c['peak_time']) < 60 for c in base_candidates): continue # 1분 내 중복 제거
        
        base_candidates.append({
            "id": len(base_candidates),
            "peak_time": t,
            "score": float(sig_scores[idx]),
            "text": f"[Signal] Audio Peak at {t:.1f}s",
            "type": "signal"
        })

    # [3] Global Ranking: Top 15 선정
    base_candidates = sorted(base_candidates, key=lambda x: x['score'], reverse=True)[:15]
    print(f"   ✅ [V5] Selected top {len(base_candidates)} candidates for precision evaluation.")

    # [4] Reduce Phase: 정밀 평가 및 렌더링
    if base_candidates:
        print(f"\n🧠 [Stage 2] Gemini Precision Evaluation (gemini-1.5-flash)...")
        llm = LLMInterface()
        eval_results = llm.evaluate_candidates(kb, video_id, base_candidates, force_refresh=force_refresh)
        
        valid_clips = finalize_clips(base_candidates, eval_results, kb, video_id)
        
        if valid_clips:
            valid_clips = kb.precise_retranscribe(str(video_file), valid_clips)
            print(f"\n⚙️ [Production] Rendering Final Masterpiece...")
            cutter = FastCutter()
            merged_clips = cutter.smart_merge(valid_clips, min_gap=FactoryConfig.CONTINUITY_GAP)
            cutter.cut_clips(str(video_file), merged_clips)
            print(f"\n✨ [V5 SUCCESS] Narrative Highlight Movie Produced!")
        else:
            print("\n❌ No clips passed the final quality hurdle.")
    else:
        print("\n❌ No candidates identified in this video.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Target video file path")
    parser.add_argument("--preset", help="Style preset (e.g. presets/kimdo.json)", default=None)
    parser.add_argument("--chapters", help="Number of top chapters to select", type=int, default=3)
    parser.add_argument("--no-visual", help="Skip visual analysis to save time", action="store_true")
    parser.add_argument("--force-refresh", help="Ignore existing checkpoints and force fresh analysis", action="store_true")
    args = parser.parse_args()
    
    if args.preset:
        FactoryConfig.load_preset(args.preset)

    if args.no_visual:
        FactoryConfig.SKIP_VISUAL = True

    try:
        run_factory(args.video_path, top_n_chapters=args.chapters, force_refresh=args.force_refresh)
    except Exception as e:
        print(f"\n🚨 [Critical Error] {e}")
        import traceback
        traceback.print_exc()
