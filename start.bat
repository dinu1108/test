@echo off
echo [Start] Activating Python 3.10 Environment...
call venv_310\Scripts\activate

echo [Start] Checking CUDA Status...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo [Start] Running Auto Highlight Extractor...
python main.py
pause
