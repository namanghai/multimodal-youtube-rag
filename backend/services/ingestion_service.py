"""
Orchestrates the full /process-video pipeline:
transcript extraction -> chunking -> frame sampling/OCR -> embedding -> FAISS index.
Kept separate from main.py so the route handler stays thin.
"""
import logging

from config import FRAMES_DIR
from models.schemas import ProcessVideoResponse
from services import transcript_service, vision_service, vector_store
from services.vector_store import ChunkMetadata
from utils.youtube_utils import extract_video_id

logger = logging.getLogger(__name__)


def process_video(youtube_url: str, force_reprocess: bool = False) -> ProcessVideoResponse:
    video_id = extract_video_id(youtube_url)

    if vector_store.index_exists(video_id) and not force_reprocess:
        metadata = vector_store.load_metadata(video_id)
        meta_info = transcript_service.fetch_video_metadata(video_id)
        return ProcessVideoResponse(
            video_id=video_id,
            title=meta_info.title,
            duration_seconds=meta_info.duration_seconds,
            num_text_chunks=sum(1 for m in metadata if m.type == "text"),
            num_visual_chunks=sum(1 for m in metadata if m.type == "visual"),
            status="already_indexed",
        )

    logger.info("Processing video %s", video_id)
    video_meta = transcript_service.fetch_video_metadata(video_id)

    segments = transcript_service.fetch_transcript(video_id)
    raw_text_chunks = transcript_service.chunk_transcript(segments)
    text_chunks = [
        ChunkMetadata(text=c.text, start_time=c.start_time, end_time=c.end_time, type="text")
        for c in raw_text_chunks
    ]

    try:
        raw_visual_chunks = vision_service.extract_visual_chunks(video_id, FRAMES_DIR)
    except Exception as exc:
        # Visual pipeline is best-effort — a video with a working transcript should
        # still index successfully even if frame download/OCR hits an issue.
        logger.warning("Visual pipeline failed for %s, continuing text-only: %s", video_id, exc)
        raw_visual_chunks = []

    visual_chunks = [
        ChunkMetadata(text=c.text, start_time=c.start_time, end_time=c.end_time, type="visual")
        for c in raw_visual_chunks
    ]

    vector_store.build_and_save_index(video_id, text_chunks, visual_chunks)

    return ProcessVideoResponse(
        video_id=video_id,
        title=video_meta.title,
        duration_seconds=video_meta.duration_seconds,
        num_text_chunks=len(text_chunks),
        num_visual_chunks=len(visual_chunks),
        status="processed",
    )
