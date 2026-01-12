import os
import time
from google import genai

from config import load_api_key, get_all_api_keys

TEST_MODEL = "gemini-2.0-flash"

def list_available_models(client):
    print("  📋 사용 가능한 모델 목록 조회 중...")
    try:
        models = client.models.list()
        print("  ✅ [API 제공 모델 목록]")
        for m in models:
            # name usually comes as "models/gemini-1.5-flash", etc.
            print(f"     - {m.name}")
        print("")
        return True
    except Exception as e:
        print(f"  ❌ 모델 목록 조회 실패: {e}\n")
        return False

def check_all_keys():
    # Use strict loader from config
    keys = get_all_api_keys()

    if not keys:
        print("❌ 유효한 API 키를 찾을 수 없습니다. .env 파일을 확인해주세요.")
        return

    print(f"\n🔍 총 {len(keys)}개의 API 키를 정밀 점검합니다... (소스: .env 파일 우선)\n")

    for i, key in enumerate(keys):
        masked_key = f"{key[:5]}...{key[-5:]}" if len(key) > 10 else "***"
        print(f"--- [{i+1}번 키 점검] {masked_key} ---")

        try:
            client = genai.Client(api_key=key)

            print(f"  🚀 테스트 모델 [{TEST_MODEL}] 호출 중...")
            
            # 1차 시도
            try:
                response = client.models.generate_content(
                    model=TEST_MODEL,
                    contents="Respond with OK",
                )
            except Exception as e_inner:
                msg_inner = str(e_inner).lower()
                # 404 Error Check
                if "404" in msg_inner or "not found" in msg_inner:
                     print(f"  ⚠️ 1차 시도 실패 (404 Not Found). 모델명을 찾을 수 없습니다.")
                     # Try listing models to help user fix it
                     list_available_models(client)
                     
                     # 2차 시도: models/ 접두사 붙여보기 (Last Ditch Effort)
                     print(f"  🔄 'models/' 접두사로 재시도...")
                     response = client.models.generate_content(
                        model=f"models/{TEST_MODEL}",
                        contents="Respond with OK",
                     )
                else:
                    raise e_inner

            if response.text:
                print("  ✅ 정상 응답 (이 키는 살아있습니다!)\n")
            else:
                print("  ⚠️ 응답은 왔지만 텍스트 없음\n")

        except Exception as e:
            msg = str(e).lower()
            print(f"  🚨 오류 상세: {e}")

            if "quota" in msg or "429" in msg:
                print("  ⚠️ 쿼터 초과 (429)\n")
            elif "permission" in msg or "403" in msg:
                print("  🚫 권한 없음 / 키 비활성 (403)\n")
            elif "not found" in msg or "404" in msg:
                print("  ❌ 모델을 절대 찾을 수 없음 (404) - 위 목록을 참고하여 모델명을 수정하세요.\n")
            else:
                print("  ❌ 접근 불가 / 기타 오류\n")
        
        time.sleep(1)

if __name__ == "__main__":
    check_all_keys()