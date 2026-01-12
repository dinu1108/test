import numpy as np
import scipy.io.wavfile as wavfile
import os
import shutil
from pathlib import Path
from sync_matcher import SyncMatcher

def create_dummy_audio(filename, duration, sr=4000, tone=440):
    t = np.linspace(0, duration, int(sr * duration))
    # Generate a chirp signal so it's unique at different times
    y = np.sin(2 * np.pi * tone * t + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    # Add some noise
    y += np.random.normal(0, 0.1, y.shape)
    
    # Needs to be saved as mp4 for the matcher (it expects mp4 usually, but ffmpeg handles wav too if renamed or just use wav and tweak matcher?)
    # Matcher uses ffmpeg to extract. ffmpeg handles wav.
    # But matcher looks for *.mp4 glob.
    # Let's simple create a .mp4 container with silent video or just rename .wav to .mp4 for audio-only testing if ffmpeg accepts it (it might complain about stream).
    # Safer: produce a real mp4 with audio using ffmpeg.
    
    # Save as temporary wav
    wavfile.write('temp_gen.wav', sr, y.astype(np.float32))
    
    # Convert to mp4
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={duration}", "-i", "temp_gen.wav", "-c:v", "libx264", "-c:a", "aac", filename]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove('temp_gen.wav')

import subprocess

def test():
    Path("test_data/raw").mkdir(parents=True, exist_ok=True)
    Path("test_data/edit").mkdir(parents=True, exist_ok=True)
    
    print("Generating Raw (10 min)...")
    create_dummy_audio("test_data/raw/raw_test.mp4", 600)  # 10 mins
    
    print("Generating Edit (1 min cut from 05:00)...")
    # Cut from 300s to 360s
    cmd = ["ffmpeg", "-y", "-ss", "300", "-i", "test_data/raw/raw_test.mp4", "-t", "60", "-c", "copy", "test_data/edit/edit_test.mp4"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("Running Matcher...")
    matcher = SyncMatcher()
    # Mocking arguments
    results = matcher.match("test_data/raw", "test_data/edit/edit_test.mp4")
    
    print("\nResults:")
    print(results)
    
    # Validation
    found = False
    for k, v in results.items():
        # Expect raw_test.mp4
        # Expect start ~00:05:00
        if "raw_test" in v['source']:
            print("Source Match: OK")
            # Parse start "00:05:00"
            h, m, s = map(int, v['start'].split(':'))
            sec = h*3600 + m*60 + s
            if 295 <= sec <= 305:
                print("Timestamp Match: OK (Found within 5s margin)")
                found = True
            else:
                print(f"Timestamp Mismatch: Found {v['start']}, Expected 00:05:00")
    
    if not found:
        print("Test FAILED: Segment not found.")
    else:
        print("Test PASSED.")

    # Cleanup
    shutil.rmtree("test_data")

if __name__ == "__main__":
    test()
