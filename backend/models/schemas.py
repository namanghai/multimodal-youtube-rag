from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------- /process-video ----------

class ProcessVideoRequest(BaseModel):
    youtube_url: str = Field(..., description="Full YouTube video URL")
    force_reprocess: bool = Field(
        False, description="Re-ingest even if this video_id is already indexed"
    )


class ProcessVideoResponse(BaseModel):
    video_id: str
    title: str
    duration_seconds: int
    num_text_chunks: int
    num_visual_chunks: int
    status: Literal["processed", "already_indexed"]


# ---------- /chat ----------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    video_id: str
    question: str
    chat_history: List[ChatMessage] = Field(default_factory=list)


class SourceChunk(BaseModel):
    type: Literal["text", "visual"]
    content: str
    start_time: float
    end_time: float
    youtube_timestamp_url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    video_id: str


# ---------- /summary ----------

class Chapter(BaseModel):
    title: str
    start_time: float
    youtube_timestamp_url: str


class SummaryResponse(BaseModel):
    video_id: str
    summary: str
    chapters: List[Chapter]


# ---------- /quiz ----------

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_option_index: int
    explanation: str
    source_timestamp: Optional[float] = None


class QuizResponse(BaseModel):
    video_id: str
    questions: List[QuizQuestion]
