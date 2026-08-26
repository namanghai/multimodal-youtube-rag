from openai import OpenAI
from backend.config import settings
from backend.services.vector_store import VectorStoreService

class RAGService:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama"
        )

    def answer_query(self, query: str, chat_history: list = None) -> dict:
        retrieved_chunks = self.vector_store.search(query, top_k=settings.TOP_K_TEXT)
        
        context_blocks = []
        for chunk in retrieved_chunks:
            mins = int(chunk['start_time'] // 60)
            secs = int(chunk['start_time'] % 60)
            timestamp_str = f"[{mins:02d}:{secs:02d}]"
            context_blocks.append(f"{timestamp_str} ({chunk['type']}): {chunk['text']}")
            
        context_str = "\n".join(context_blocks)
        
        system_prompt = (
            "You are an assistant analyzing a YouTube video transcript and on-screen visuals. "
            "Answer the question strictly using the provided context. "
            "Always include the timestamp [MM:SS] showing where each piece of information was found."
        )
        
        user_prompt = f"Context from video:\n{context_str}\n\nUser Question: {query}"
        
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=messages,
            temperature=0.2
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": retrieved_chunks
        }