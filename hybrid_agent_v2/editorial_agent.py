from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from .knowledge_base import VideoKnowledgeBase
from .llm_interface import LLMInterface
import math

class AgentState(TypedDict):
    video_path: str
    highlights: List[dict]
    current_idx: int
    final_cuts: List[dict]

class EditorialAgent:
    def __init__(self):
        self.kb = VideoKnowledgeBase()
        self.llm = LLMInterface()
        self.workflow = self._build_graph()
        
    def _build_graph(self):
        builder = StateGraph(AgentState)
        
        builder.add_node("scan", self.scan_candidates)
        builder.add_node("search_narrative_start", self.search_narrative_start)
        builder.add_node("human_review", self.human_review)
        
        builder.set_entry_point("scan")
        builder.add_edge("scan", "search_narrative_start")
        builder.add_edge("search_narrative_start", "human_review")
        builder.add_edge("human_review", END)
        
        return builder.compile()

    # ... (scan_candidates logic is above) ...

    def human_review(self, state: AgentState):
        """
        Step 3: Human-in-the-Loop.
        Show the plan to the user and ask for approval.
        """
        cuts = state['final_cuts']
        print("\n" + "="*60)
        print("🎬  HYBRID AGENT EDITORIAL PLAN  🎬")
        print("="*60)
        print(f"{'ID':<4} | {'Start':<8} | {'End':<8} | {'Duration':<6} | {'Summary'}")
        print("-" * 60)
        
        for i, cut in enumerate(cuts):
            dur = cut['end'] - cut['start']
            # Limit summary length
            summ = (cut['summary'][:40] + '..') if len(cut['summary']) > 40 else cut['summary']
            print(f"{i+1:<4} | {cut['start']:<8.1f} | {cut['end']:<8.1f} | {dur:<6.1f} | {summ}")
            
        print("="*60)
        
        while True:
            choice = input("\n[User Check] Proceed with this plan? (y/n): ").strip().lower()
            if choice == 'y':
                print("[Agent] ✅ Plan Approved. Proceeding to Render.")
                return {"final_cuts": cuts}
            elif choice == 'n':
                print("[Agent] 🛑 Plan Aborted by User.")
                # We could implement editing logic here, but for now abort.
                return {"final_cuts": []} # Empty list stops rendering
            else:
                print("Please type 'y' or 'n'.")

    def scan_candidates(self, state: AgentState):
        """
        Step 1: Strategic Filtering (Local V1).
        Use CPU-based signal analysis (Audio/Chat) to find candidates,
        saving Gemini tokens for the deep reasoning step.
        """
        print("[Agent] 🧠 Strategic Filtering: Running Local Signal Analysis...")
        video_path = state['video_path']
        
        # Import V1 Analyst dynamically to avoid circular deps at top level if any
        from modules.analyst import Analyst
        
        # Initialize Analyst (Mock params or load from preset?)
        # For V2, we just want the robust signal detection.
        analyst = Analyst(preset_name="Khan") # Default to a good preset
        
        # 1. Audio Analysis (Local) - Fast
        audio_data = analyst.analyze_audio_advanced(video_path)
        if not audio_data:
            print("[Agent] ❌ No audio data found.")
            return {"highlights": []}
            
        # 2. Score Calculation (Local)
        scores, times = analyst.calculate_scores(audio_data)
        
        # 3. Top-K Candidate Selection
        import numpy as np
        indices = np.argsort(scores)[::-1]
        top_k_indices = indices[:20] # Get Top 20 peaks
        
        candidates = []
        for i in top_k_indices:
            t = times[i]
            s = scores[i]
            # Use Whisper transcript from KB to get text at this time?
            # We can query KB for text at 't'.
            video_id = Path(video_path).stem
            # Fetch small context around peak to identify 'What is this?'
            # Just get 1 segment at t
            context = self.kb.get_context(video_id, t, t+5)
            text = context[0]['text'] if context else "High Energy Moment"
            
            candidates.append({"time": t, "score": float(s), "text": text})
            
        # Sort by time for linear processing
        candidates.sort(key=lambda x: x['time'])
        
        # Debounce/Merge close candidates (Local logic)
        unique = []
        if candidates:
            last = candidates[0]
            unique.append(last)
            for c in candidates[1:]:
                if c['time'] - last['time'] > 60: # 1 minute debounce
                    unique.append(c)
                    last = c
        
        print(f"[Agent] ✅ Found {len(unique)} High-Probability Candidates (Local V1 filtered).")
        return {"highlights": unique}

    def search_narrative_start(self, state: AgentState):
        """
        Step 2: Narrative Lookback Node.
        Logic: RAG Query -> Gemini Analysis (JSON) -> Filtering & Smart Merge
        """
        highlights = state['highlights']
        video_path = state['video_path']
        video_id = Path(video_path).stem
        final_cuts = []
        
        print(f"[Agent] 🕵️ Searching Narrative Start for {len(highlights)} candidates...")
        
        for h in highlights:
            peak_time = h['time']
            # 1. RAG Query (Lookback 5m)
            context_start = max(0, peak_time - 300)
            context = self.kb.get_context(video_id, context_start, peak_time + 30)
            
            if not context:
                print(f"    [Skip] No context found for {peak_time}s")
                continue
                
            # 2. Gemini Analysis (Returns JSON dict)
            result = self.llm.analyze_story_start(
                event_description=h['text'],
                transcript_segment=context
            )
            
            # 3. Parse & Filter
            start_time = result.get('start', -1)
            priority = result.get('priority', 0)
            reason = result.get('reason', 'Unknown')
            
            if priority < 3: # "Context-less cuts are discarded" (User Rule)
                print(f"    🗑️ Discarded (Priority {priority}): {reason}")
                continue
            
            if start_time > 0 and start_time < peak_time:
                print(f"    ✅ Narrative Found (P{priority}): {start_time:.1f}s -> {peak_time:.1f}s | {reason}")
                start = start_time
            else:
                start = max(0, peak_time - 30)
                
            end = peak_time + 20
            
            # 4. Smart Merge (Local check against previous cut)
            if final_cuts:
                last_cut = final_cuts[-1]
                # If gap is small (< 120s), merge
                if start - last_cut['end'] < 120:
                    print(f"    🔗 Smart Merging with previous clip (Gap < 2m)")
                    last_cut['end'] = max(last_cut['end'], end)
                    # Append summary/reason?
                    last_cut['summary'] += f" + {h['text']}"
                    continue

            final_cuts.append({
                "start": start, 
                "end": end, 
                "summary": h['text'], 
                "reason": reason,
                "priority": priority
            })
            
        return {"final_cuts": final_cuts}

    def run(self, video_path):
        # 1. Ingest if needed
        self.kb.ingest(video_path)
        
        # 2. Run Flow
        input_state = {"video_path": video_path, "highlights": [], "current_idx": 0, "final_cuts": []}
        result = self.workflow.invoke(input_state)
        return result['final_cuts']
