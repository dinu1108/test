import argparse
import sys
from pathlib import Path

# Add current dir to path to find modules if needed
sys.path.append(str(Path.cwd()))

from hybrid_agent_v2.editorial_agent import EditorialAgent

def main():
    parser = argparse.ArgumentParser(description="Hybrid Agent V2: RAG + Gemini")
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()

    print(f"🚀 Starting Hybrid Agent V2 on {args.video_path}...")
    
    agent = EditorialAgent()
    cuts = agent.run(args.video_path)
    
    print(f"\n✅ Editor Agent Finished. Generated {len(cuts)} cuts.")
    for i, cut in enumerate(cuts):
        print(f"  [{i+1}] {cut['start']:.1f}s -> {cut['end']:.1f}s | {cut['summary']}")
        
    # Optional: Call Editor V1 to render?
    # For now, just output timestamps as proof of intelligence.

if __name__ == "__main__":
    main()
