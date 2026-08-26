from youtube_transcript_api import YouTubeTranscriptApi
from backend.config import settings

class TranscriptService:
    def get_transcript(self, video_id: str) -> list[dict]:
        try:
            # Instantiate the API object for newer package versions
            ytt_api = YouTubeTranscriptApi()
            fetched_data = ytt_api.fetch(video_id)
            
            raw_entries = []
            for entry in fetched_data:
                if isinstance(entry, dict):
                    raw_entries.append(entry)
                else:
                    raw_entries.append({
                        "text": getattr(entry, "text", str(entry)),
                        "start": getattr(entry, "start", 0.0),
                        "duration": getattr(entry, "duration", 0.0)
                    })
                    
            return self._chunk_transcript(raw_entries)
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return []

    def _chunk_transcript(self, raw_entries: list[dict]) -> list[dict]:
        chunks = []
        current_chunk_text = []
        chunk_start_time = 0.0
        
        for entry in raw_entries:
            if not current_chunk_text:
                chunk_start_time = entry['start']
                
            current_chunk_text.append(entry['text'])
            current_duration = (entry['start'] + entry['duration']) - chunk_start_time
            
            if current_duration >= settings.CHUNK_DURATION_SECONDS:
                chunks.append({
                    "text": " ".join(current_chunk_text),
                    "start_time": chunk_start_time,
                    "end_time": entry['start'] + entry['duration'],
                    "type": "audio"
                })
                current_chunk_text = []

        if current_chunk_text:
            chunks.append({
                "text": " ".join(current_chunk_text),
                "start_time": chunk_start_time,
                "end_time": raw_entries[-1]['start'] + raw_entries[-1]['duration'],
                "type": "audio"
            })
            
        return chunks