from openai import OpenAI
from backend.config import settings

class QuizService:
    def __init__(self):
        self.client = OpenAI(base_url=settings.OLLAMA_BASE_URL, api_key="ollama")

    def generate_quiz(self, chunks: list[dict]) -> dict:
        sample_text = " ".join([c["text"] for c in chunks[:20]])
        prompt = (
            "Generate 3 multiple-choice quiz questions with 4 options (A, B, C, D) "
            "and indicate the correct answer based on this video text:\n\n"
            f"{sample_text}"
        )
        res = self.client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return {"quiz": res.choices[0].message.content}