import cv2
import yt_dlp
import pytesseract
import numpy as np
from config import settings


class VisionService:
    """
    Samples frames from a YouTube stream URL, detects scene changes,
    and runs OCR (Tesseract) to extract on-screen text as visual chunks.

    Requires:
      - Tesseract OCR installed and on PATH (or configure pytesseract.tesseract_cmd)
      - OpenCV (opencv-python-headless) and yt-dlp installed
    """

    def extract_visual_data(self, youtube_url: str, video_id: str) -> list[dict]:
        # 1. Resolve the direct stream URL (no download needed)
        ydl_opts = {"format": "best[height<=480]", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info.get("url", "")
            if not stream_url:
                raise RuntimeError("yt-dlp could not resolve a stream URL for this video.")

        # 2. Open stream with OpenCV
        cap = cv2.VideoCapture(stream_url)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_interval_frames = max(1, int(fps * settings.FRAME_SAMPLE_INTERVAL))

        prev_gray = None
        extracted_visuals: list[dict] = []
        frame_count = 0

        while cap.isOpened() and len(extracted_visuals) < settings.MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval_frames == 0:
                timestamp_sec = frame_count / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Scene-change detection
                is_new_scene = True
                if prev_gray is not None:
                    diff = np.mean(cv2.absdiff(gray, prev_gray))
                    if diff < settings.SCENE_CHANGE_THRESHOLD:
                        is_new_scene = False

                if is_new_scene:
                    prev_gray = gray
                    ocr_text = pytesseract.image_to_string(gray).strip()
                    if len(ocr_text) > 15:   # Filter OCR noise
                        extracted_visuals.append({
                            "text":       f"[On-Screen Text at {timestamp_sec:.1f}s]: {ocr_text}",
                            "start_time": timestamp_sec,
                            "end_time":   timestamp_sec + settings.FRAME_SAMPLE_INTERVAL,
                            "type":       "visual",
                        })

            frame_count += 1

        cap.release()
        return extracted_visuals