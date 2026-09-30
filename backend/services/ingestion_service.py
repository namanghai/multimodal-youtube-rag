"""
Ingestion pipeline: transcript → chunking → frame OCR → embedding → FAISS.

Called by main.py's /process-video route. Takes shared in-memory dicts
(vector_stores, chunks_cache) as arguments so no global state lives here.
"""
from __future__ import annotations

import logging
from typing import Any

from utils.youtube_utils import extract_video_id, build_timestamp_url
from services.transcript_service import TranscriptService
from services.vision_service import VisionService
from services.vector_store import VectorStoreService
from models.schemas import ProcessVideoResponse

logger = logging.getLogger(__name__)

_transcript_svc = TranscriptService()
_vision_svc     = VisionService()


def process_video(
    youtube_url:   str,
    force_reprocess: bool,
    vector_stores: dict[str, Any],
    chunks_cache:  dict[str, list[dict]],
) -> tuple[ProcessVideoResponse, list[dict]]:
    """
    Full ingestion pipeline.

    Returns (ProcessVideoResponse, all_chunks) so the caller can cache chunks
    for the summary / quiz endpoints.
    """
    video_id = extract_video_id(youtube_url)

    # ── Idempotency guard ────────────────────────────────────────────────────
    if video_id in vector_stores and not force_reprocess:
        cached = chunks_cache.get(video_id, [])
        text_n   = sum(1 for c in cached if c.get("type") == "audio")
        visual_n = sum(1 for c in cached if c.get("type") == "visual")
        return (
            ProcessVideoResponse(
                video_id=video_id,
                title="(cached)",
                duration_seconds=0,
                num_text_chunks=text_n,
                num_visual_chunks=visual_n,
                status="already_indexed",
            ),
            cached,
        )

    logger.info("Starting ingestion for video_id=%s", video_id)

    # ── 1. Transcript ────────────────────────────────────────────────────────
    audio_chunks = _transcript_svc.get_transcript(video_id)
    logger.info("Transcript chunks: %d", len(audio_chunks))

    # ── 2. Visual / OCR (best-effort) ────────────────────────────────────────
    try:
        visual_chunks = _vision_svc.extract_visual_data(youtube_url, video_id)
        logger.info("Visual chunks: %d", len(visual_chunks))
    except Exception as exc:
        logger.warning("Visual pipeline failed (%s). Continuing text-only.", exc)
        visual_chunks = []

    all_chunks = audio_chunks + visual_chunks
    if not all_chunks:
        raise ValueError(
            "Could not extract any content from this video. "
            "The video may be private, age-restricted, or have no transcript."
        )

    # ── 3. Embed + index ─────────────────────────────────────────────────────
    vs = VectorStoreService()
    vs.add_chunks(all_chunks)

    vector_stores[video_id] = vs
    chunks_cache[video_id]  = all_chunks

    return (
        ProcessVideoResponse(
            video_id=video_id,
            title="",           # yt-dlp metadata fetch is optional; omit for speed
            duration_seconds=0,
            num_text_chunks=len(audio_chunks),
            num_visual_chunks=len(visual_chunks),
            status="processed",
        ),
        all_chunks,
    )
