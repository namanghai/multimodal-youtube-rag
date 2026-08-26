from sentence_transformers import SentenceTransformer
import faiss
import numpy as np 
from backend.config import settings

class VectorStoreService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def add_chunks(self, chunks: list[dict]):
        if not chunks:
            return
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).astype("float32")
        
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
            
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results