"""
Vector store using FAISS + sentence-transformers.

The SentenceTransformer model is loaded ONCE as a module-level singleton
so it is ready before the first request arrives.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
from config import settings

logger = logging.getLogger(__name__)

# ── Singleton embedding model — loaded once at import time ────────────────────
logger.info("Loading embedding model '%s' …", settings.EMBEDDING_MODEL)
_embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
logger.info("Embedding model ready.")


class VectorStoreService:
    """
    In-memory FAISS flat-L2 index per video.
    Embeddings are produced by the shared singleton model above.
    """

    def __init__(self):
        # Reuse the already-loaded singleton
        self.model     = _embed_model
        self.dimension = self.model.get_embedding_dimension()
        self.index     = faiss.IndexFlatL2(self.dimension)
        self.metadata: list[dict] = []

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        texts      = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_vec          = self.model.encode([query]).astype("float32")
        k                  = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                chunk = self.metadata[idx].copy()
                chunk["score"] = float(dist)
                results.append(chunk)
        return results