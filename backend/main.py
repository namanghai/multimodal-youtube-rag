"""
FastAPI application — all routes are thin wrappers that delegate to service
layer. Imports use relative names so the app runs from inside backend/ with:

    uvicorn main:app --reload --port 8000
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings                                    # noqa: F401 (verifies config loads)
from models.schemas import (
    ProcessVideoRequest, ProcessVideoResponse,
    ChatRequest, ChatResponse,
    SummaryResponse, QuizResponse,
)
from services.ingestion_service import process_video
from services.vector_store import VectorStoreService
from services.rag_service import RAGService
from services.summary_service import SummaryService
from services.quiz_service import QuizService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Multimodal RAG API",
    description=(
        "Chat with any YouTube video using 100 % free local models: "
        "Ollama (llama3) for generation and all-MiniLM-L6-v2 for embeddings."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session stores ──────────────────────────────────────────────────
# video_id -> VectorStoreService
_vector_stores: dict[str, VectorStoreService] = {}
# video_id -> list[dict]  (all chunks, for summary/quiz)
_chunks_cache:  dict[str, list[dict]]         = {}

summary_service = SummaryService()
quiz_service    = QuizService()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/process-video", response_model=ProcessVideoResponse)
def process_video_route(request: ProcessVideoRequest):
    """Download transcript + OCR frames, embed everything, build FAISS index."""
    try:
        result, all_chunks = process_video(
            youtube_url=request.youtube_url,
            force_reprocess=request.force_reprocess,
            vector_stores=_vector_stores,
            chunks_cache=_chunks_cache,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during video processing")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):
    """RAG chat: retrieve relevant chunks, answer with Ollama."""
    vs = _vector_stores.get(request.video_id)
    if vs is None:
        raise HTTPException(
            status_code=404,
            detail="Video not processed yet. Call POST /process-video first.",
        )
    rag = RAGService(vs)
    return rag.answer_query(
        video_id=request.video_id,
        query=request.question,
        chat_history=[m.model_dump() for m in request.chat_history],
    )


@app.get("/summary/{video_id}", response_model=SummaryResponse)
def summary_route(video_id: str):
    """Generate a bullet-point summary + chapter breakdown."""
    chunks = _chunks_cache.get(video_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Video not found. Process it first.")
    return summary_service.generate(video_id, chunks)


@app.get("/quiz/{video_id}", response_model=QuizResponse)
def quiz_route(video_id: str):
    """Generate 3 multiple-choice quiz questions from the indexed content."""
    chunks = _chunks_cache.get(video_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Video not found. Process it first.")
    return quiz_service.generate(video_id, chunks)