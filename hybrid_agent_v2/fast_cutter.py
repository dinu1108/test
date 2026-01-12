import subprocess
from pathlib import Path

class FastCutter:
    def __init__(self):
        pass

    def timestamp_to_seconds(self, ts):
        """HH:MM:SS -> Seconds"""
        try:
            parts = ts.split(':')
            if len(parts) == 3:
                return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0])*60 + float(parts[1])
        except: return 0
        return 0

    def cut_clips(self, video_path, clips, output_dir="clips", max_workers=4):
        """
        Render clips in parallel using ThreadPoolExecutor.
        """
        video_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        
        print(f"\n🚀 [Parallel Production] Rendering with {max_workers} workers...")
        
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._render_single, video_path, clip, i, out_dir)
                for i, clip in enumerate(clips)
            ]
            
            # Wait for all futures
            for future in tqdm(futures, desc="Rendering", unit="clip"):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Render Error: {e}")

        print("[Factory] All operations complete.")

    def _render_single(self, video_path, clip, idx, out_dir):
        start, end = clip['start'], clip['end']
        summary = clip.get('reason', f"clip_{idx}")
        safe_name = "".join([c for c in summary if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")[:30]
        out_file = out_dir / f"ep{idx+1}_{safe_name}.mp4"

        # Timestamp conversion
        s_sec = start if isinstance(start, (int, float)) else self.timestamp_to_seconds(start)
        e_sec = end if isinstance(end, (int, float)) else self.timestamp_to_seconds(end)
        duration = e_sec - s_sec

        if duration <= 0:
            return

        # Audio Filter Logic (Fade Fix for short clips)
        if duration < 1.5:
            # Too short for fades, just pure cut
            af_filter = None 
        else:
            fade_out_st = max(0, duration - 0.5)
            af_filter = f"afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_st:.2f}:d=0.5"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(video_path),
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-c:a", "aac", "-b:a", "192k"
        ]
        
        if af_filter:
            cmd.extend(["-af", af_filter])
            
        cmd.append(str(out_file))

        # Run FFmpeg
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
