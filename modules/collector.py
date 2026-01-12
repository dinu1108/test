import os
import yt_dlp
import json

class Collector:
    """
    영상과 채팅 데이터를 다운로드하는 클래스
    """
    def __init__(self, raw_data_dir="raw_data"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def download_video_and_chat(self, url):
        """
        URL에서 영상과 메타데이터(채팅 포함 가능 여부는 플랫폼 의존적)를 다운로드합니다.
        유튜브의 경우 yt-dlp가 채팅을 직접적으로 완벽하게 json으로 내려주지 않을 수 있어,
        별도의 처리가 필요할 수 있으나 기본적으로 bestvideo+bestaudio를 받습니다.
        
        참고: 실시간 채팅 다시보기 다운로드는 yt-dlp의 --write-sub --write-auto-sub 또는
        후처리 툴이 필요할 수 있습니다. 여기서는 기본 yt-dlp 기능을 사용합니다.
        """
        print(f"[Collector] Downloading from {url}...")
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{self.raw_data_dir}/%(id)s.%(ext)s',
            
            # 429 Error & Bot Detection Avoidance (Headers)
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            },
            
            'ignoreerrors': True,
            'nocheckcertificate': True,
            'quiet': False,
            
            # Chat/Subtitle settings
            'writesubtitles': True, 
            'writeautomaticsub': False,
            'writeinfojson': True,
            
            # Accelerated Download
            'external_downloader': 'aria2c',
            'external_downloader_args': ['-x', '8', '-k', '1M', '--split', '8'],
            
            # Additional reliability
            'retries': 10,
            
            # Use local cookies.txt file
            'cookiefile': 'cookies.txt',
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                
                # 에러 무시 옵션으로 인해 info_dict가 None일 수 있음
                if not info_dict:
                    print(f"[Collector] Failed to extract info for {url}")
                    return None, None

                video_id = info_dict.get('id', 'unknown')
                video_path = os.path.join(self.raw_data_dir, f"{video_id}.mp4")
                
                # 채팅 파일 경로 추정 (yt-dlp 버전에 따라 다를 수 있음)
                # info.json, live_chat.json 등
                json_path = os.path.join(self.raw_data_dir, f"{video_id}.info.json") 
                
                # .live_chat.json이 생성되었을 경우 우선순위
                live_chat_path = os.path.join(self.raw_data_dir, f"{video_id}.live_chat.json")
                if os.path.exists(live_chat_path):
                    json_path = live_chat_path
                
                print(f"[Collector] Download complete: {video_path}")
                return video_path, json_path
                
        except Exception as e:
            print(f"[Collector] Critical Error during download: {e}")
            return None, None

if __name__ == "__main__":
    # Test Code
    url = input("Enter video URL: ")
    collector = Collector()
    collector.download_video_and_chat(url)
