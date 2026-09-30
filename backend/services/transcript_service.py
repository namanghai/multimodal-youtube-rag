from youtube_transcript_api import YouTubeTranscriptApi
from config import settings


class TranscriptService:
    """Fetches and chunks a YouTube transcript via youtube-transcript-api."""

    def get_transcript(self, video_id: str) -> list[dict]:
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched_data = ytt_api.fetch(video_id)

            raw_entries = []
            for entry in fetched_data:
                if isinstance(entry, dict):
                    raw_entries.append(entry)
                else:
                    raw_entries.append({
                        "text":     getattr(entry, "text",     str(entry)),
                        "start":    getattr(entry, "start",    0.0),
                        "duration": getattr(entry, "duration", 0.0),
                    })

            return self._chunk_transcript(raw_entries)
        except Exception as exc:
            print(f"[TranscriptService] Could not fetch transcript: {exc}")
            return []

    def _chunk_transcript(self, raw_entries: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        current_text: list[str] = []
        chunk_start = 0.0

        for entry in raw_entries:
            if not current_text:
                chunk_start = entry["start"]

            current_text.append(entry["text"])
            elapsed = (entry["start"] + entry["duration"]) - chunk_start

            if elapsed >= settings.CHUNK_DURATION_SECONDS:
                chunks.append({
                    "text":       " ".join(current_text),
                    "start_time": chunk_start,
                    "end_time":   entry["start"] + entry["duration"],
                    "type":       "audio",
                })
                current_text = []

        # Flush the last partial chunk
        if current_text and raw_entries:
            last = raw_entries[-1]
            chunks.append({
                "text":       " ".join(current_text),
                "start_time": chunk_start,
                "end_time":   last["start"] + last["duration"],
                "type":       "audio",
            })

        return chunks