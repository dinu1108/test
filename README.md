# Auto Highlight Extractor (Preset-Based Hybrid AI)

This system is an automated video editing pipeline that extracts high-impact moments from long-form broadcast videos. It features a **Preset-Based Architecture** allowing users to switch between editing styles (Energetic, Talkative, Skillful) and uses a **Hybrid AI Funnel** (Signal Processing -> CLIP -> LLaVA).

## 1. System Pipeline Architecture (Tree View)

```text
[Main Entry: main.py] (CLI: --style energetic)
|
|-- ⚙️ Configuration (Preset Injection)
|   |-- presets/energetic.json (Focus: Scream, Rapid Reaction)
|   |-- presets/talkative.json (Focus: Chat, Dialogue Emotion)
|   +-- presets/skillful.json  (Focus: Visual Clarity, Game Events)
|
|-- 📂 Ingestion (Data Collection)
|   |-- yt-dlp: Download Video & Live Chat Data
|   +-- FFmpeg: Extract Wav & Frames (Parallel)
|
|-- 📂 Analysis Stage 1: Advanced Signal Processing (Heuristic)
|   |-- [Audio] RMS Slope (Onset Speed) + Pitch + ZCR (Screams)
|   |-- [Chat] Velocity Peaks + Keywords
|   |-- [Scoring] Clamped Temporal Accumulation (Prevents Score Explosion)
|   +-- [Filter] Hybrid Threshold (Top-K + Super Highlights)
|
|-- 📂 Analysis Stage 2: Visual Screening (CLIP)
|   |-- Input: Top Candidates from Stage 1
|   |-- Logic: "Is this a black screen / loading screen?"
|   +-- Action: Fast Reject (VRAM Optimized)
|
|-- 📂 Analysis Stage 3: Deep Verification (LLaVA)
|   |-- Input: Top 80 Survivors
|   |-- Prompt: "Is this a Viral Highlight? (Persona-driven)"
|   +-- Output: Final Verified Highlights with Descriptions
|
|-- 📂 Production (Editing & Rendering)
|   |-- Whisper: Generate Subtitles & Emotion Check
|   |-- MoviePy: Individual Sequence Rendering (Temp Files)
|   +-- FFmpeg: Stream Copy Concat (No Re-encoding)
|
+-- 🎬 Output: [VideoName]_FINAL_RECAP.mp4
```

## 2. Usage & Styles

Run the extractor with a specific style preset:

```bash
# 1. Energetic (Default) - Best for Horror Games, Action
python main.py https://youtu.be/VideoURL --style energetic

# 2. Talkative - Best for Just Chatting, Talk Shows
python main.py https://youtu.be/VideoURL --style talkative

# 3. Skillful - Best for competitive Gameplay
python main.py https://youtu.be/VideoURL --style skillful
```

## 3. Core Technologies

| Feature | Tech | Description |
| :--- | :--- | :--- |
| **Onset Detection** | `Librosa` | Uses **RMS Slope** (Derivative) to detect the exact moment a reaction starts, not just when it is loud. |
| **Clamping** | `Numpy` | **Temporal Accumulation** adds up scores over time but is clamped to a Max Score to prevent long loud noise from dominating. |
| **Persona AI** | `LLaVA` | The AI is prompted with a specific persona ("You are a YouTube Shorts Editor") to judge entertainment value. |
| **Hybrid Filter** | `Logic` | Combines **Top-N** (Relative) and **Soft Threshold** (Absolute) to ensure "super highlights" are never missed even in busy streams. |
| **Fast Render** | `FFmpeg` | **Divide & Concat** strategy renders clips individually to avoid memory leaks, then merges instantly. |

## 4. Requirements
*   Python 3.10+
*   NVIDIA GPU (CUDA)
*   Ollama (running `llava`) at `localhost:11434`
*   FFmpeg & ImageMagick installed and on shared paths.
