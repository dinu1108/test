import subprocess
import json
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class FastCutter:
    def __init__(self):
        # 폰트 경로 설정 (윈도우 기준)
        self.font_path = "C\\\\:/Windows/Fonts/malgunbd.ttf"
        if not Path("C:/Windows/Fonts/malgunbd.ttf").exists():
            self.font_path = "Arial" # Fallback

    def smart_merge(self, clips, min_gap=0):
        """
        Merge overlapping or adjacent clips.
        """
        if not clips:
            return []
        
        sorted_clips = sorted(clips, key=lambda x: x['start'])
        merged = []
        
        if not sorted_clips:
            return []
            
        curr = sorted_clips[0].copy()
        
        for next_clip in sorted_clips[1:]:
            if next_clip['start'] <= curr['end'] + min_gap:
                curr['end'] = max(curr['end'], next_clip['end'])
                if 'reason' in next_clip and 'reason' in curr:
                    if next_clip['reason'] != curr['reason']:
                        curr['reason'] += f" + {next_clip['reason']}"
                # Merge precise segments if they exist
                if 'precise_segments' in next_clip:
                    if 'precise_segments' not in curr: curr['precise_segments'] = []
                    curr['precise_segments'].extend(next_clip['precise_segments'])
            else:
                merged.append(curr)
                curr = next_clip.copy()
        
        merged.append(curr)
        return merged

    def normalize_time(self, value):
        if isinstance(value, (int, float)): return float(value)
        if isinstance(value, str):
            try:
                parts = value.split(':')
                if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                elif len(parts) == 2: return int(parts[0])*60 + float(parts[1])
                else: return float(value)
            except: return 0.0
        return 0.0

    def _generate_srt(self, segments, clip_start, duration):
        """클립 내부 상대 시간을 기준으로 SRT 파일 생성"""
        srt_content = ""
        for i, seg in enumerate(segments):
            s = seg['start'] - clip_start
            e = seg['end'] - clip_start
            
            # 클립 범위 밖의 가드
            if e <= 0 or s >= duration: continue
            s = max(0, s)
            e = min(duration, e)
            
            def format_time(ts):
                h = int(ts // 3600)
                m = int((ts % 3600) // 60)
                s = int(ts % 60)
                ms = int((ts % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            srt_content += f"{i+1}\n{format_time(s)} --> {format_time(e)}\n{seg['text']}\n\n"
        return srt_content

    def cut_clips(self, video_path, clips, output_dir="clips", max_workers=4):
        video_path = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(exist_ok=True)
        
        print(f"\n🚀 [V4 Narrative Production] Rendering {len(clips)} Story Units...")
        
        rendered_files = []
        
        # 1. 개별 클립 렌더링 (자막/페이드 포함)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._render_single, video_path, clip, i, out_dir): i
                for i, clip in enumerate(clips)
            }
            
            for future in tqdm(futures, desc="Rendering Story Units", unit="unit"):
                idx = futures[future]
                try:
                    res_files = future.result()
                    rendered_files.extend(res_files)
                except Exception as e:
                    print(f"\n❌ Story Unit #{idx} Failed: {e}")

        if not rendered_files:
            print("❌ No files were rendered.")
            return

        # 2. 최종 무비 병합 (Concatenation)
        self.merge_to_final_movie(rendered_files, out_dir / "final_narrative_highlight.mp4")

    def _render_single(self, video_path, clip, idx, out_dir):
        """자막이 입혀진 클립 + 타이틀 카드/브릿지 렌더링"""
        start = self.normalize_time(clip.get('start', 0))
        end = self.normalize_time(clip.get('end', 0))
        duration = end - start
        if duration <= 0: return []
        
        res_files = []
        
        # (A) 타이틀 카드 (챕터 제목이 있고 첫 번째 포인트일 때 등 특정 조건)
        # 여기서는 단순히 bridge_text가 있을 때 브릿지 카드 생성
        bridge_text = clip.get('bridge_text')
        if bridge_text:
            bridge_file = self._render_bridge(video_path, bridge_text, idx, start, out_dir)
            if bridge_file: res_files.append(bridge_file)

        # (B) 메인 클립 (자막 포함)
        summary = clip.get('reason', f"clip_{idx}")
        safe_name = "".join([c for c in summary if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")[:30]
        out_file = out_dir / f"unit_{idx+1:03d}_{safe_name}.mp4"

        # 자막 제작
        srt_file = None
        vf_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" # 기본 스케일링
        
        if 'precise_segments' in clip:
            srt_content = self._generate_srt(clip['precise_segments'], start, duration)
            if srt_content.strip():
                srt_path = out_dir / f"sub_{idx+1:03d}.srt"
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                srt_file = str(srt_path).replace("\\", "/") # FFmpeg는 슬래시 선호
                # 자막 오버레이 필터 추가
                vf_filter += f",subtitles='{srt_file}':force_style='FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=1,Alignment=2'"

        # 오디오 페이드
        af_filter = f"afade=t=in:st=0:d=0.5,afade=t=out:st={max(0, duration-0.5):.2f}:d=0.5"

        from presets.factory_config import FactoryConfig
        codec = FactoryConfig.VIDEO_CODEC
        
        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(video_path),
            "-vf", vf_filter,
            "-af", af_filter,
            "-c:v", codec, 
            "-preset", "p4" if "nvenc" in codec else "medium",
            "-cq" if "nvenc" in codec else "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            str(out_file)
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        res_files.append(out_file)
        return res_files

    def _render_bridge(self, video_path, text, idx, timestamp, out_dir):
        out_file = out_dir / f"unit_{idx+1:03d}_A_bridge.mp4"
        if len(text) > 40: text = text[:38] + ".."
        
        drawtext_filter = f"drawtext=fontfile='{self.font_path}':text='{text}':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=(h-text_h)/2:shadowcolor=black:shadowx=2:shadowy=2"
        
        from presets.factory_config import FactoryConfig
        codec = FactoryConfig.VIDEO_CODEC
        
        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-ss", f"{timestamp:.3f}", "-i", str(video_path),
            "-t", "2.5",
            "-vf", f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,gblur=sigma=20,format=yuv420p,{drawtext_filter}",
            "-c:v", codec, "-preset", "p4" if "nvenc" in codec else "medium",
            "-c:a", "aac", str(out_file)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            return out_file
        except: return None

    def merge_to_final_movie(self, file_list, output_path):
        """Concat demuxer를 사용하여 무손실 병합"""
        print(f"\n🎬 [Fusion] Merging {len(file_list)} units into final movie...")
        
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ✅ FIX: 임시 목록 파일을 출력 디렉토리에 생성하여 상대 경로 사용 (Windows 경로 이슈 방지)
        list_file_path = output_path.parent / "concat_list.txt"
        
        with open(list_file_path, "w", encoding="utf-8") as f:
            for file_path in file_list:
                # 출력 파일 기준으로 상대 경로 계산
                try:
                    p = os.path.relpath(file_path, output_path.parent).replace("\\", "/")
                except ValueError:
                    # 서로 다른 드라이브일 경우 절대 경로 사용
                    p = str(Path(file_path).absolute()).replace("\\", "/")
                
                # 홑따옴표 이스케이프 처리
                p = p.replace("'", "'\\''")
                f.write(f"file '{p}'\n")
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file_path),
            "-c", "copy",
            str(output_path)
        ]
        
        try:
            # shell=True는 지양하되, Windows에서 복잡한 인자 처리가 필요할 경우 고려
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"✨ [V4 SUCCESS] Final Masterpiece created at: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"   🚨 Concatenation failed with exit code {e.returncode}. Attempting recovery...")
            # Fallback: re-encode if copy fails (rare but possible due to stream mismatch)
            cmd_fallback = cmd.copy()
            cmd_fallback[cmd_fallback.index("-c") + 1] = "libx264"
            cmd_fallback.insert(cmd_fallback.index("libx264") + 1, "-c:a")
            cmd_fallback.insert(cmd_fallback.index("-c:a") + 1, "aac")
            subprocess.run(cmd_fallback, check=True)
            print(f"✨ [V4 SUCCESS] Final Masterpiece created (via re-encode) at: {output_path}")
        finally:
            if list_file_path.exists():
                list_file_path.unlink()
