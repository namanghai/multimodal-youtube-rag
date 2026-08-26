from openai import OpenAI
from backend.config import settings

class SummaryService:
    def __init__(self):
        self.client = OpenAI(base_url=settings.OLLAMA_BASE_URL, api_key="ollama")

    def generate_summary_and_chapters(self, chunks: list[dict]) -> dict:
        full_text = " ".join([c["text"] for c in chunks[:30]])  # Sample first blocks
        prompt = (
            "Based on the following video content, provide:\n"
            "1. A concise 3-bullet-point summary.\n"
            "2. A breakdown of key chapters with estimated timestamps.\n\n"
            f"Content:\n{full_text}"
        )
        res = self.client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return {"summary_and_chapters": res.choices[0].message.content}