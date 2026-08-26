import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "llama3")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    
    CHUNK_DURATION_SECONDS: int = int(os.getenv("CHUNK_DURATION_SECONDS", "45"))
    CHUNK_OVERLAP_SECONDS: int = int(os.getenv("CHUNK_OVERLAP_SECONDS", "10"))
    
    FRAME_SAMPLE_INTERVAL: int = int(os.getenv("FRAME_SAMPLE_INTERVAL_SECONDS", "5"))
    SCENE_CHANGE_THRESHOLD: float = float(os.getenv("SCENE_CHANGE_THRESHOLD", "30.0"))
    MAX_FRAMES_PER_VIDEO: int = int(os.getenv("MAX_FRAMES_PER_VIDEO", "40"))
    
    TOP_K_TEXT: int = int(os.getenv("TOP_K_TEXT", "5"))
    TOP_K_VISUAL: int = int(os.getenv("TOP_K_VISUAL", "3"))

settings = Settings()