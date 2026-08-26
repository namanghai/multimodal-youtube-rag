from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.utils.youtube_utils import extract_video_id
from backend.services.transcript_service import TranscriptService
from backend.services.vision_service import VisionService
from backend.services.vector_store import VectorStoreService
from backend.services.rag_service import RAGService
from backend.services.summary_service import SummaryService
from backend.services.quiz_service import QuizService
app = FastAPI(title="YouTube Multimodal RAG API")

# In-memory storage for active session indices and text chunk caches
vector_stores = {}
video_chunks_cache = {}

transcript_service = TranscriptService()
vision_service = VisionService()
summary_service = SummaryService()
quiz_service = QuizService()

class ProcessVideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    video_id: str
    query: str
    chat_history: list = []

@app.post("/process-video")
async def process_video(request: ProcessVideoRequest):
    try:
        video_id = extract_video_id(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # 1. Fetch transcript chunks
    audio_chunks = transcript_service.get_transcript(video_id)
    
    # 2. Fetch OCR visual chunks (optional: handles frame extraction)
    try:
        visual_chunks = vision_service.extract_visual_data(request.url, video_id)
    except Exception:
        visual_chunks = []  # Fallback gracefully if OCR/video stream fails
        
    all_chunks = audio_chunks + visual_chunks
    if not all_chunks:
        raise HTTPException(status_code=400, detail="Could not extract content or transcript from this video.")
        
    # 3. Store locally in FAISS using sentence-transformers
    vs = VectorStoreService()
    vs.add_chunks(all_chunks)
    
    vector_stores[video_id] = vs
    video_chunks_cache[video_id] = all_chunks
    
    return {
        "video_id": video_id,
        "total_chunks": len(all_chunks),
        "audio_chunks": len(audio_chunks),
        "visual_chunks": len(visual_chunks)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.video_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Video has not been processed yet. Please process the URL first.")
        
    vs = vector_stores[request.video_id]
    rag = RAGService(vs)
    response = rag.answer_query(request.query, request.chat_history)
    return response

@app.get("/summary/{video_id}")
async def get_summary(video_id: str):
    if video_id not in video_chunks_cache:
        raise HTTPException(status_code=404, detail="Video data not found.")
    return summary_service.generate_summary_and_chapters(video_chunks_cache[video_id])

@app.get("/quiz/{video_id}")
async def get_quiz(video_id: str):
    if video_id not in video_chunks_cache:
        raise HTTPException(status_code=404, detail="Video data not found.")
    return quiz_service.generate_quiz(video_chunks_cache[video_id])