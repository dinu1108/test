from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import whisper
import warnings
import traceback
import subprocess

# Suppress warnings
warnings.filterwarnings("ignore")

class Editor:
    def __init__(self, clips_dir="clips", model_size="base"):
        self.clips_dir = clips_dir
        os.makedirs(self.clips_dir, exist_ok=True)
        
        target_magick = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
        if os.path.exists(target_magick):
            change_settings({"IMAGEMAGICK_BINARY": target_magick})
        else:
            print(f"[Editor] Warning: ImageMagick not found at {target_magick}")
        
        self.font_face = 'Malgun Gothic' 
        # Font path for Pillow (Thumbnail)
        self.font_path = "C:/Windows/Fonts/malgunbd.ttf" # Bold for Thumbnail
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/malgun.ttf"
            
        print(f"[Editor] Loading Whisper model ('{model_size}')...")
        self.model = whisper.load_model(model_size)

    def smart_merge(self, intervals, gap_threshold=300):
        if not intervals: return []
        sorted_ints = sorted(intervals, key=lambda x: x['start'])
        merged = []
        curr = sorted_ints[0].copy()
        for next_item in sorted_ints[1:]:
            if next_item['start'] <= curr['end'] + gap_threshold: 
                curr['end'] = max(curr['end'], next_item['end'])
                if len(next_item.get('summary', '')) > len(curr.get('summary', '')):
                    curr['summary'] = next_item['summary']
                # Keep max score
                curr['score'] = max(curr.get('score', 0), next_item.get('score', 0))
            else:
                merged.append(curr)
                curr = next_item.copy()
        merged.append(curr)
        return merged

    def generate_thumbnail(self, video, time_sec, text, output_path):
        print(f"[Editor] Generating Thumbnail at {time_sec:.1f}s...")
        try:
            # Extract frame using MoviePy
            frame = video.get_frame(time_sec)
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            
            # Dynamic font size
            width, height = img.size
            font_size = int(height * 0.15)
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except:
                font = ImageFont.load_default()

            # Text: "AI BEST MOMENT" + Summary
            # Simplification: Top Line "AI HIGHLIGHT", Bottom Line summary
            top_text = "🔥 AI BEST MOMENT 🔥"
            
            def draw_text_with_stroke(x, y, txt, f, color='white', stroke='black'):
                w = 4 # Stroke width
                draw.text((x-w, y), txt, font=f, fill=stroke)
                draw.text((x+w, y), txt, font=f, fill=stroke)
                draw.text((x, y-w), txt, font=f, fill=stroke)
                draw.text((x, y+w), txt, font=f, fill=stroke)
                draw.text((x, y), txt, font=f, fill=color)

            # Draw Top
            # text bbox
            # bbox = draw.textbbox((0,0), top_text, font=font)
            # text_w = bbox[2] - bbox[0]
            # Centered
            draw_text_with_stroke(width/2 - (font_size*len(top_text)*0.3), 50, top_text, font)
            
            # Draw Bottom (Summary) if short
            if text and len(text) < 20:
                draw_text_with_stroke(width/2 - (font_size*len(text)*0.3), height-150, text, font)
            
            img.save(output_path, quality=95)
            print(f"[Editor] Thumbnail saved: {output_path}")
        except Exception as e:
            print(f"[Editor] Thumbnail generation failed: {e}")

    def create_hook_clip(self, video_clip, start_time, duration=5):
        print("[Editor] Creating Intro Hook...")
        try:
            # Safe boundary
            s = max(0, start_time)
            e = min(video_clip.duration, start_time + duration)
            hook = video_clip.subclip(s, e)
            
            # Overlay "Coming Up"
            txt = TextClip("🔥 잠시 후 하이라이트! 🔥", fontsize=50, color='yellow', font=self.font_face, stroke_color='black', stroke_width=2).set_position(('center', 'bottom')).set_duration(hook.duration)
            
            return CompositeVideoClip([hook, txt])
        except:
            return None

    def create_intro(self, style_name, width, height, duration=3):
        bg = ColorClip(size=(width, height), color=(0,0,0)).set_duration(duration)
        try:
            txt = TextClip(f"AI Highlight Extraction\nStyle: {style_name.upper()}", color='cyan', font=self.font_face, fontsize=70, stroke_color='white', stroke_width=2, method='caption', align='center', size=(width, None)).set_position('center').set_duration(duration)
            return CompositeVideoClip([bg, txt])
        except: return bg

    def add_subtitles(self, clip, duration):
        temp_audio = "temp_sub_audio.wav"
        try:
            clip.audio.write_audiofile(temp_audio, logger=None)
            res = self.model.transcribe(temp_audio, language='ko')
            subs = []
            for seg in res['segments']:
                t, s, e = seg['text'].strip(), seg['start'], seg['end']
                if not t or e-s < 0.3: continue
                try: 
                    tc = TextClip(t, fontsize=40, color='white', font=self.font_face, stroke_color='black', stroke_width=2, method='caption', size=(int(clip.w*0.8),None)).set_position(('center',0.85), relative=True).set_start(s).set_duration(e-s)
                    subs.append(tc)
                except: pass
            if os.path.exists(temp_audio): os.remove(temp_audio)
            return CompositeVideoClip([clip]+subs) if subs else clip
        except: 
            if os.path.exists(temp_audio): os.remove(temp_audio)
            return clip

    def add_context_overlay(self, clip, summary_text):
        if summary_text and len(summary_text) > 5:
            try:
                bar = ColorClip(size=(clip.w, 60), color=(0,0,0,180)).set_position(('center','top')).set_duration(10)
                txt = TextClip(f"AI 분석: {summary_text}", fontsize=30, color='cyan', font=self.font_face, stroke_color='black', stroke_width=1, method='caption', align='West', size=(int(clip.w*0.9), 60)).set_position((20,'top')).set_start(0).set_duration(10)
                return CompositeVideoClip([clip, bar, txt])
            except: pass
        return clip

    def create_full_recap(self, video_path, hotspots, style_name="General"):
        print(f"\n[{self.__class__.__name__}] Production (Style: {style_name})...")
        if not hotspots: return

        # 1. Select Best Moment for Hook & Thumbnail
        # Sort by score descending
        # Ensure hotspots have 'score', if not default 0
        best_moment = max(hotspots, key=lambda x: x.get('score', 0))
        # Best time is approx mid of the interval? 
        # Analyst returns start/end. Let's pick start + 30s as "peak" approximation or just start if short.
        # Assuming build up is 60s, peak is likely at start + 60s is original.
        # hotspots -> 'start' = peak - 60. So peak = start + 60.
        peak_time = best_moment['start'] + 60 
        
        merged = self.smart_merge(hotspots)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_files = [] 
        
        try:
            video = VideoFileClip(video_path)
            
            # A. Create Intro Hook
            hook_clip = self.create_hook_clip(video, peak_time, duration=5)
            if hook_clip:
                p = os.path.join(self.clips_dir, "hook.mp4")
                hook_clip.write_videofile(p, codec='libx264', audio_codec='aac', preset='ultrafast', logger=None)
                temp_files.append(p)

            # B. Create Intro Title
            intro = self.create_intro(style_name, video.w, video.h)
            p_intro = os.path.join(self.clips_dir, "intro_title.mp4")
            intro.write_videofile(p_intro, codec='libx264', audio_codec='aac', preset='ultrafast', logger=None)
            temp_files.append(p_intro)
            
            # C. Render Clips
            for i, item in enumerate(merged):
                s, e = max(0, item['start']), min(item['end'], video.duration)
                if s >= e: continue
                
                cl = video.subclip(s, e)
                cl = self.add_context_overlay(cl, item.get('summary', ''))
                cl = self.add_subtitles(cl, e-s)
                
                out_p = os.path.join(self.clips_dir, f"seq_{i:03d}.mp4")
                cl.write_videofile(out_p, codec='libx264', audio_codec='aac', preset='ultrafast', threads=8, logger='bar')
                temp_files.append(out_p)
                del cl

            # D. Create Thumbnail
            thumb_path = os.path.join(self.clips_dir, f"{video_name}_thumbnail.jpg")
            self.generate_thumbnail(video, peak_time, best_moment.get('summary', ''), thumb_path)

            video.close()
            
            # E. Concatenate
            if temp_files:
                list_p = os.path.join(self.clips_dir, "concat.txt")
                with open(list_p, 'w', encoding='utf-8') as f:
                    for t in temp_files: f.write(f"file '{os.path.abspath(t).replace(os.sep,'/')}'\n")
                
                final_p = os.path.join(self.clips_dir, f"{video_name}_FINAL.mp4")
                subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_p, "-c", "copy", final_p], check=True)
                print(f"\n✅ Final Video: {final_p}")
                print(f"✅ Thumbnail: {thumb_path}")
                
                # Cleanup
                os.remove(list_p)
                for t in temp_files:
                    try: os.remove(t)
                    except: pass
                    
        except Exception as e:
            traceback.print_exc()
