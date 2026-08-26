# YouTube Multimodal RAG Chatbot

Chat with any YouTube video: transcript + on-screen visual/OCR content are chunked,
embedded, and stored in a per-video FAISS index. GPT-4o answers questions grounded
strictly in that retrieved context, with clickable timestamps back into the video.

## Project structure

```
youtube-rag/
├── backend/
│   ├── main.py                     # FastAPI app + routes
│   ├── config.py                   # all env-driven settings
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response models
│   ├── services/
│   │   ├── transcript_service.py   # yt-dlp + youtube-transcript-api, chunking
│   │   ├── vision_service.py       # frame sampling, scene detection, OCR
│   │   ├── vector_store.py         # FAISS index build/search per video_id
│   │   ├── ingestion_service.py    # orchestrates the /process-video pipeline
│   │   ├── rag_service.py          # retrieval + grounded GPT-4o chat
│   │   ├── summary_service.py      # whole-video summary + chapters
│   │   └── quiz_service.py         # 3-question quiz generation
│   ├── utils/
│   │   └── youtube_utils.py        # video ID parsing, timestamp URL builder
│   └── data/
│       ├── faiss_indices/          # <video_id>.index + <video_id>.meta.json
│       └── frames/                 # extracted frame JPEGs, per video_id
└── frontend/
    ├── app.py                      # Streamlit client
    └── requirements.txt
```

## Prerequisites

- Python 3.10+
- **Tesseract OCR** installed at the system level (pytesseract just calls it):
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), add to PATH
- ffmpeg on PATH (yt-dlp needs it for muxing) — `brew install ffmpeg` / `apt-get install ffmpeg`
- An OpenAI API key with access to `gpt-4o` and `text-embedding-3-small`

## Setup

```bash
# 1. clone/copy the project, then from the repo root:

# --- backend ---
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# --- frontend (separate terminal) ---
cd ../frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

**Terminal 1 — backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
API docs at `http://localhost:8000/docs`.

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run app.py
```
Opens at `http://localhost:8501`. It talks to the backend at `http://localhost:8000`
by default — override with a `BACKEND_URL` value in `.streamlit/secrets.toml` if needed.

## Usage flow

1. Paste a YouTube URL in the sidebar, click **Process video**.
   - Fetches transcript + metadata, samples frames on scene changes, OCRs them,
     embeds everything, builds a FAISS index under `backend/data/faiss_indices/`.
2. Use the chat box to ask questions — answers cite transcript/visual chunks with
   clickable timestamps.
3. Sidebar tabs generate a **Summary + chapters** or a **3-question quiz**, both
   grounded in the indexed content.

## Notes on production-readiness

- **Idempotent ingestion**: re-processing a known `video_id` is a no-op unless
  `force_reprocess=true` is passed — avoids re-downloading/re-embedding on repeat calls.
- **Best-effort visual pipeline**: if frame download/OCR fails (e.g. region lock,
  ffmpeg missing), ingestion still succeeds text-only rather than hard-failing.
- **Cost control**: `USE_VISION_MODEL_FOR_FRAMES` is off by default — OCR alone
  covers most slide/text content; enable the GPT-4o vision captioning only if you
  need descriptions of diagrams/UI with no text.
- **Swap-in for scale**: FAISS here is local flat-index, fine for single-instance
  deployments. For multi-instance/high-volume, swap `vector_store.py` for a hosted
  vector DB (pgvector, Pinecone, Qdrant) behind the same function signatures.
- **CORS** is wide open (`allow_origins=["*"]`) for local dev — restrict this to
  your actual frontend origin before deploying.
