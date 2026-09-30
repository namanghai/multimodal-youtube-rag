# 🎥 YouTube Multimodal RAG Chatbot

Chat with any YouTube video using **100 % free, local AI** — no OpenAI key needed.

- **Transcript** → chunked, embedded with `all-MiniLM-L6-v2` (sentence-transformers), stored in a per-video FAISS index.
- **On-screen text** → frames are sampled on scene changes and OCR'd with Tesseract.
- **Ollama** (`llama3` by default) answers questions grounded strictly in the retrieved chunks, with **clickable timestamps** back into the video.

---

## 📁 Project Structure

```
youtube-rag/
├── backend/
│   ├── main.py                      # FastAPI app + routes
│   ├── config.py                    # All env-driven settings + path constants
│   ├── requirements.txt
│   ├── .env.example                 # Copy → .env and edit
│   ├── models/
│   │   └── schemas.py               # Pydantic request / response models
│   ├── services/
│   │   ├── ingestion_service.py     # Orchestrates the /process-video pipeline
│   │   ├── transcript_service.py    # youtube-transcript-api + chunking
│   │   ├── vision_service.py        # yt-dlp stream → OpenCV frame sampling → Tesseract OCR
│   │   ├── vector_store.py          # FAISS index build / search (sentence-transformers)
│   │   ├── rag_service.py           # Retrieve chunks → call Ollama → return answer + sources
│   │   ├── summary_service.py       # Whole-video summary + chapters via Ollama
│   │   └── quiz_service.py          # 3-question multiple-choice quiz via Ollama
│   ├── utils/
│   │   └── youtube_utils.py         # Video ID parser + timestamp URL builder
│   └── data/
│       ├── faiss_indices/           # Auto-created; gitignored
│       └── frames/                  # Auto-created; gitignored
└── frontend/
    ├── app.py                       # Streamlit UI
    └── requirements.txt
```

---

## ✅ Prerequisites

| Dependency | Install |
|---|---|
| **Python 3.10+** | [python.org](https://www.python.org/downloads/) |
| **Ollama** | [ollama.com](https://ollama.com) — download, install, run `ollama serve` |
| **llama3 model** | `ollama pull llama3` (≈ 4.7 GB, runs on CPU) |
| **Tesseract OCR** | Windows: [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) → add to PATH · macOS: `brew install tesseract` · Ubuntu: `sudo apt install tesseract-ocr` |
| **ffmpeg** | Windows: [ffmpeg.org/download](https://ffmpeg.org/download.html) → add to PATH · macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` |

> [!NOTE]
> Tesseract and ffmpeg are only needed for **visual/OCR chunks**. If they are missing, ingestion still works text-only — no hard failure.

---

## 🚀 Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/youtube-rag.git
cd youtube-rag

# ── Backend ──────────────────────────────────────────────────────────────────
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

pip install -r requirements.txt

# Copy example env and review settings
cp .env.example .env
# No API keys needed — just make sure Ollama is running

# ── Frontend (separate terminal) ─────────────────────────────────────────────
cd ../frontend
python -m venv venv
venv\Scripts\activate        # or source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Running

**Terminal 1 — Ollama:**
```bash
ollama serve          # starts the local LLM server on http://localhost:11434
```

**Terminal 2 — Backend:**
```bash
cd backend
# Windows (activated venv):
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```
Interactive API docs: `http://localhost:8000/docs`

**Terminal 3 — Frontend:**
```bash
cd frontend
venv\Scripts\activate
streamlit run app.py
```
Opens at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Paste a YouTube URL** in the sidebar → click **▶ Process Video**.
   - Downloads transcript, samples frames on scene changes, OCRs them, embeds everything into a FAISS index.
   - Use **Force re-process** to rebuild an already-indexed video.
2. **Ask questions** in the chat box — answers cite transcript/visual chunks with **clickable `[MM:SS]` timestamps**.
3. Sidebar → **📋 Summary & Chapters** to get a bullet-point summary and chapter breakdown.
4. Sidebar → **🧠 Generate Quiz** to get 3 multiple-choice questions grounded in the video content.

---

## ⚙️ Configuration

All settings live in `backend/.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `CHAT_MODEL` | `phi3` | Any model pulled via `ollama pull <model>` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| `CHUNK_DURATION_SECONDS` | `45` | Transcript chunk length in seconds |
| `FRAME_SAMPLE_INTERVAL_SECONDS` | `5` | OCR frame sampling interval |
| `MAX_FRAMES_PER_VIDEO` | `40` | Cap on visual chunks per video |
| `TOP_K_TEXT` | `5` | Transcript chunks retrieved per query |
| `TOP_K_VISUAL` | `3` | Visual chunks retrieved per query |

**Recommended free Ollama models (ordered by CPU speed):**
```bash
ollama pull phi3          # ⭐ Recommended — 3.8B, ~2GB, fast on CPU
ollama pull tinyllama     # Fastest — 1.1B, ~600MB, very basic quality
ollama pull mistral       # 7B, ~4GB, good quality, moderate speed
ollama pull llama3        # 8B, ~4.7GB, best quality, slow on CPU-only
```

> [!TIP]
> If you have an NVIDIA GPU, all models run much faster. Check GPU usage with `ollama ps` after sending a request.

---

## 📝 Notes

- **No API keys required** — everything runs locally via Ollama + sentence-transformers.
- **Idempotent ingestion** — re-processing a known `video_id` is a no-op unless `force_reprocess=true`.
- **Best-effort vision pipeline** — if OCR fails (Tesseract/ffmpeg missing or region-locked video), ingestion succeeds text-only.
- **FAISS is in-memory** — restarting the backend clears all indices. For persistence, save/load the FAISS index to disk or swap in a hosted vector DB (pgvector, Qdrant, Pinecone).
- **CORS** is wide open (`*`) for local development — restrict `allow_origins` before deploying.

---

## 📄 License

MIT
