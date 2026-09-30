from openai import OpenAI
from config import settings

# Max characters fed to the LLM — keeps prompts short so Ollama responds fast on CPU
_MAX_CONTEXT_CHARS = 3000


class SummaryService:
    """Generates a bullet-point summary + chapter breakdown using Ollama."""

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",
        )

    def generate(self, video_id: str, chunks: list[dict]) -> dict:
        # Use first 10 chunks and cap total chars — keeps inference fast on CPU
        sample_text = " ".join(c["text"] for c in chunks[:10])
        sample_text = sample_text[:_MAX_CONTEXT_CHARS]

        prompt = (
            "Based on the following YouTube video content, provide:\n"
            "1. A concise 3-bullet-point summary of the key points.\n"
            "2. A breakdown of 3-5 main chapters/topics with approximate timestamps.\n\n"
            f"Content:\n{sample_text}"
        )

        response = self.client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return {
            "video_id":             video_id,
            "summary_and_chapters": response.choices[0].message.content or "",
        }