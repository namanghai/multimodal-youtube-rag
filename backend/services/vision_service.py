import cv2
import yt_dlp
import pytesseract
from PIL import Image
import os
import numpy as np
from backend.config import settings

class VisionService:
    def extract_visual_data(self, youtube_url: str, video_id: str) -> list[dict]:
        # 1. Get direct stream URL using yt-dlp
        ydl_opts = {'format': 'best[height<=480]'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info['url']

        cap = cv2.VideoCapture(stream_url)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_interval_frames = int(fps * settings.FRAME_SAMPLE_INTERVAL)
        
        prev_frame = None
        extracted_visuals = []
        frame_count = 0
        total_extracted = 0

        while cap.isOpened() and total_extracted < settings.MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval_frames == 0:
                timestamp_sec = frame_count / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check for significant scene change
                is_unique = True
                if prev_frame is not None:
                    diff = np.mean(cv2.absdiff(gray, prev_frame))
                    if diff < settings.SCENE_CHANGE_THRESHOLD:
                        is_unique = False
                
                if is_unique:
                    prev_frame = gray
                    text = pytesseract.image_to_string(gray).strip()
                    if len(text) > 15:  # filter out OCR noise
                        extracted_visuals.append({
                            "text": f"[On-Screen Text]: {text}",
                            "start_time": timestamp_sec,
                            "end_time": timestamp_sec + settings.FRAME_SAMPLE_INTERVAL,
                            "type": "visual"
                        })
                        total_extracted += 1
                        
            frame_count += 1

        cap.release()
        return extracted_visuals