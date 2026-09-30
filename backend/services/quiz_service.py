from openai import OpenAI
from config import settings


class QuizService:
    """Generates 3 multiple-choice quiz questions using Ollama."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",
        )

    def generate(self, video_id: str, chunks: list[dict]) -> dict:
        sample_text = " ".join(c["text"] for c in chunks[:20])

        prompt = (
            "Generate exactly 3 multiple-choice quiz questions based on the following video content.\n"
            "For each question provide:\n"
            "  - The question text\n"
            "  - Four answer options (A, B, C, D)\n"
            "  - The correct answer letter\n"
            "  - A brief explanation\n\n"
            f"Video content:\n{sample_text}"
        )

        response = self.client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return {
            "video_id": video_id,
            "quiz":     response.choices[0].message.content or "",
        }