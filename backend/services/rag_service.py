"""
RAG service — retrieves relevant chunks from FAISS then calls Ollama via the
OpenAI-compatible endpoint (openai Python SDK, base_url pointed at Ollama).
100 % free, runs completely locally.
"""
from openai import OpenAI
from config import settings
from services.vector_store import VectorStoreService
from utils.youtube_utils import build_timestamp_url


class RAGService:
    def __init__(self, vector_store: VectorStoreService):
        self.vs = vector_store
        # Ollama exposes an OpenAI-compatible /v1 endpoint
        self.client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",   # Required by the SDK; Ollama ignores it
        )

    def answer_query(
        self,
        video_id:     str,
        query:        str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        # ── Retrieve relevant chunks ──────────────────────────────────────────
        chunks = self.vs.search(query, top_k=settings.TOP_K_TEXT + settings.TOP_K_VISUAL)

        context_blocks: list[str] = []
        sources: list[dict] = []

        for chunk in chunks:
            mins = int(chunk["start_time"] // 60)
            secs = int(chunk["start_time"] % 60)
            ts   = f"[{mins:02d}:{secs:02d}]"
            context_blocks.append(f"{ts} ({chunk['type']}): {chunk['text']}")

            sources.append({
                "type":                  chunk["type"],
                "content":               chunk["text"],
                "start_time":            chunk["start_time"],
                "end_time":              chunk["end_time"],
                "youtube_timestamp_url": build_timestamp_url(video_id, chunk["start_time"]),
            })

        context_str = "\n".join(context_blocks) if context_blocks else "(no context retrieved)"

        # ── Build messages ────────────────────────────────────────────────────
        system_prompt = (
            "You are an AI assistant analyzing a YouTube video's transcript and on-screen text. "
            "Answer the user's question using ONLY the context provided below. "
            "Always cite the timestamp [MM:SS] where each piece of information appears."
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({
            "role":    "user",
            "content": f"Context from video:\n{context_str}\n\nQuestion: {query}",
        })

        # ── Call Ollama ───────────────────────────────────────────────────────
        response = self.client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""

        return {
            "answer":   answer,
            "sources":  sources,
            "video_id": video_id,
        }