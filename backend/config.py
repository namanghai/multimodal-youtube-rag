import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
INDEX_DIR  = DATA_DIR / "faiss_indices"

# Create directories on import so services can write to them immediately
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    # ── Ollama (free, local) ─────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    CHAT_MODEL:      str = os.getenv("CHAT_MODEL",      "phi3")

    # ── Embeddings (free, local via sentence-transformers) ───────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_DURATION_SECONDS: int   = int(os.getenv("CHUNK_DURATION_SECONDS", "45"))
    CHUNK_OVERLAP_SECONDS:  int   = int(os.getenv("CHUNK_OVERLAP_SECONDS",  "10"))

    # ── Vision / OCR ─────────────────────────────────────────────────────────
    FRAME_SAMPLE_INTERVAL:   int   = int(os.getenv("FRAME_SAMPLE_INTERVAL_SECONDS", "5"))
    SCENE_CHANGE_THRESHOLD:  float = float(os.getenv("SCENE_CHANGE_THRESHOLD", "30.0"))
    MAX_FRAMES_PER_VIDEO:    int   = int(os.getenv("MAX_FRAMES_PER_VIDEO", "40"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K_TEXT:   int = int(os.getenv("TOP_K_TEXT",   "5"))
    TOP_K_VISUAL: int = int(os.getenv("TOP_K_VISUAL", "3"))


settings = Settings()