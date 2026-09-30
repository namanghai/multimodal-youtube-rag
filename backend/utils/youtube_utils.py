import re


def extract_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from any YouTube URL format."""
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?#]|$)',
        r'(?:embed/)([0-9A-Za-z_-]{11})',
        r'youtu\.be/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(
        f"Could not extract a video ID from URL: {url!r}. "
        "Please provide a valid YouTube URL."
    )


def build_timestamp_url(video_id: str, seconds: float) -> str:
    """Return a YouTube deep-link that starts playback at the given second."""
    t = int(seconds)
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"