import streamlit as st
import httpx
import os

# ── Backend URL (override via environment variable if needed) ─────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="YouTube Multimodal RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)
st.title("🎥 YouTube Multimodal RAG Assistant")
st.caption("100 % free · powered by Ollama (llama3) + sentence-transformers")

# ── Session state initialisation ──────────────────────────────────────────────
if "video_id"     not in st.session_state:
    st.session_state.video_id     = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 1. Process a Video")
    yt_url = st.text_input("Paste YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

    force = st.checkbox("Force re-process (ignore cache)", value=False)

    if st.button("▶ Process Video", use_container_width=True):
        if not yt_url.strip():
            st.warning("Please enter a valid YouTube URL.")
        else:
            with st.spinner("Fetching transcript, running OCR on frames, building FAISS index…"):
                try:
                    res = httpx.post(
                        f"{BACKEND_URL}/process-video",
                        json={"youtube_url": yt_url, "force_reprocess": force},
                        timeout=180.0,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.video_id     = data["video_id"]
                        st.session_state.chat_history = []   # reset chat for new video
                        st.success(
                            f"✅ **{data.get('status', 'processed').replace('_', ' ').title()}**  \n"
                            f"Video ID: `{data['video_id']}`  \n"
                            f"Text chunks: {data['num_text_chunks']} · "
                            f"Visual chunks: {data['num_visual_chunks']}"
                        )
                    else:
                        detail = res.json().get("detail", res.text)
                        st.error(f"❌ {detail}")
                except httpx.ConnectError:
                    st.error(
                        "Cannot connect to the backend. "
                        "Make sure `uvicorn main:app --reload --port 8000` is running."
                    )
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # ── Value-adds (only when a video is loaded) ──────────────────────────────
    if st.session_state.video_id:
        st.divider()
        st.header("📝 2. Video Insights")

        if st.button("📋 Summary & Chapters", use_container_width=True):
            with st.spinner("Generating summary with Ollama…"):
                try:
                    res = httpx.get(
                        f"{BACKEND_URL}/summary/{st.session_state.video_id}",
                        timeout=240.0,
                    )
                    if res.status_code == 200:
                        st.markdown(res.json().get("summary_and_chapters", ""))
                    else:
                        st.error(res.json().get("detail", "Failed to generate summary."))
                except Exception as exc:
                    st.error(f"Error: {exc}")

        if st.button("🧠 Generate Quiz", use_container_width=True):
            with st.spinner("Creating quiz questions with Ollama…"):
                try:
                    res = httpx.get(
                        f"{BACKEND_URL}/quiz/{st.session_state.video_id}",
                        timeout=240.0,
                    )
                    if res.status_code == 200:
                        st.markdown(res.json().get("quiz", ""))
                    else:
                        st.error(res.json().get("detail", "Failed to generate quiz."))
                except Exception as exc:
                    st.error(f"Error: {exc}")

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ── Main area: Chat ───────────────────────────────────────────────────────────
if st.session_state.video_id:
    st.subheader(f"💬 Chat  ·  Video ID: `{st.session_state.video_id}`")

    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user input
    if user_query := st.chat_input("Ask anything about this video…"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    payload = {
                        "video_id":    st.session_state.video_id,
                        "question":    user_query,
                        "chat_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history[:-1]
                        ],
                    }
                    res = httpx.post(
                        f"{BACKEND_URL}/chat",
                        json=payload,
                        timeout=240.0,
                    )
                    if res.status_code == 200:
                        data   = res.json()
                        answer = data.get("answer", "No answer returned.")
                        st.markdown(answer)

                        # Show clickable source timestamps
                        sources = data.get("sources", [])
                        if sources:
                            with st.expander("📌 Sources"):
                                for src in sources:
                                    url   = src.get("youtube_timestamp_url", "")
                                    start = src.get("start_time", 0)
                                    mins, secs = divmod(int(start), 60)
                                    label = f"[{mins:02d}:{secs:02d}] ({src.get('type', '')})"
                                    if url:
                                        st.markdown(f"- [{label}]({url}) — {src.get('content', '')[:120]}…")
                                    else:
                                        st.markdown(f"- {label} — {src.get('content', '')[:120]}…")

                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        err = res.json().get("detail", "Unknown error")
                        st.error(f"❌ {err}")
                except Exception as exc:
                    st.error(f"Connection error: {exc}")
else:
    st.info("👈 Paste a YouTube URL in the sidebar and click **▶ Process Video** to start!")