import os
from dotenv import load_dotenv
from pathlib import Path

def load_api_key():
    # 1. config.py가 있는 폴더 찾기
    current_dir = Path(__file__).parent.absolute()
    
    # 2. 후보 경로 설정 (현재 폴더 및 상위 폴더)
    env_candidates = [
        current_dir / ".env",
        current_dir.parent / ".env",  # hybrid_agent_v2 폴더 밖에 있는 경우 대비
        Path(os.getcwd()) / ".env"     # 현재 작업 디렉토리
    ]
    
    found = False
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"[Config] ✅ .env loaded from: {env_path}")
            found = True
            break
            
    if not found:
        print("[Config] ⚠️ Warning: .env file not found in any candidate paths.")

    # API 키 확인 (Google SDK는 GOOGLE_API_KEY를 자동으로 인식함)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Config] ❌ GOOGLE_API_KEY not found in environment variables.")
    else:
        # 키의 일부만 출력해서 확인 (보안 유지)
        print(f"[Config] Primary API Key detected: {api_key[:5]}**********")

def get_all_api_keys():
    """Available Google API Keys list return (Strictly from .env if possible)"""
    # 1. 파일에서 직접 로드 (시스템 환경변수 오염 방지)
    from dotenv import dotenv_values
    
    current_dir = Path(__file__).parent.absolute()
    env_candidates = [
        current_dir / ".env",
        current_dir.parent / ".env",
        Path(os.getcwd()) / ".env"
    ]
    
    env_config = {}
    found_path = None
    for env_path in env_candidates:
        if env_path.exists():
            env_config = dotenv_values(env_path)
            found_path = env_path
            # print(f"[ConfigDebug] Loaded keys directly from file: {env_path}")
            break
            
    keys = []
    
    # 2. .env 파일 내용 우선 사용
    if env_config:
        # Primary Key
        if "GOOGLE_API_KEY" in env_config:
            keys.append(env_config["GOOGLE_API_KEY"])
        
        # Secondary Keys (Key order preservation)
        # Check specifically for numeric sequence 2..10 to keep order
        for i in range(2, 10):
            key_name = f"GOOGLE_API_KEY_{i}"
            if key_name in env_config and env_config[key_name].strip():
                keys.append(env_config[key_name])
                
        # If user has weird named keys not in sequence, scan them too?
        # For now, let's trust the sequence or explicit names in file.
    else:
        # Fallback to os.environ if no file found (e.g. Docker/Cloud)
        load_api_key() # Load into env first
        if os.environ.get("GOOGLE_API_KEY"):
            keys.append(os.environ.get("GOOGLE_API_KEY"))
        for k, v in sorted(os.environ.items()):
            if k.startswith("GOOGLE_API_KEY_") and v not in keys:
                 keys.append(v)

    # Clean duplicates while preserving order
    unique_keys = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            unique_keys.append(k)
            seen.add(k)

    print(f"[Config] 🔑 Found {len(unique_keys)} API Keys (Strict Mode).")
    return unique_keys
