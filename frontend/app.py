import streamlit as st
import httpx

# Point to your running FastAPI backend
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="YouTube Multimodal RAG Chatbot", layout="wide")
st.title("🎥 YouTube Multimodal RAG Assistant (Free Local AI)")

# Initialize session state variables
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: Ingestion & Insights ---
with st.sidebar:
    st.header("1. Video Ingestion")
    yt_url = st.text_input("Paste YouTube URL:")
    
    if st.button("Process Video"):
        if not yt_url:
            st.warning("Please enter a valid YouTube URL.")
        else:
            with st.spinner("Downloading transcript, extracting frames & OCR, and indexing vectors..."):
                try:
                    res = httpx.post(f"{BACKEND_URL}/process-video", json={"url": yt_url}, timeout=120.0)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.video_id = data["video_id"]
                        st.success(f"Successfully indexed! Total chunks: {data['total_chunks']}")
                    else:
                        err_detail = res.json().get("detail", "Unknown error")
                        st.error(f"Error: {err_detail}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {e}")

    if st.session_state.video_id:
        st.divider()
        st.header("2. Video Value-Adds")
        
        if st.button("Generate Summary & Chapters"):
            with st.spinner("Generating summary via local LLM..."):
                try:
                    # Extended timeout to 180s for local LLM generation
                    res = httpx.get(f"{BACKEND_URL}/summary/{st.session_state.video_id}", timeout=180.0)
                    if res.status_code == 200:
                        st.markdown(res.json().get("summary_and_chapters", ""))
                    else:
                        st.error("Failed to generate summary.")
                except Exception as e:
                    st.error(f"Error: {e}")
                
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz questions..."):
                try:
                    # Extended timeout to 180s for local LLM generation
                    res = httpx.get(f"{BACKEND_URL}/quiz/{st.session_state.video_id}", timeout=180.0)
                    if res.status_code == 200:
                        st.markdown(res.json().get("quiz", ""))
                    else:
                        st.error("Failed to generate quiz.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- MAIN AREA: Chat Interface ---
if st.session_state.video_id:
    st.subheader(f"Chatting with Video ID: `{st.session_state.video_id}`")
    
    # Render past conversation messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # User query input box
    if user_query := st.chat_input("Ask a question about what was said or shown in the video:"):
        # Append user message to state and display
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Call FastAPI backend for RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Searching transcripts/visuals and generating answer..."):
                try:
                    payload = {
                        "video_id": st.session_state.video_id,
                        "query": user_query,
                        "chat_history": st.session_state.chat_history[:-1]
                    }
                    # Extended timeout to 180s for local RAG inference
                    res = httpx.post(f"{BACKEND_URL}/chat", json=payload, timeout=180.0)
                    if res.status_code == 200:
                        answer = res.json().get("answer", "No answer returned.")
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    else:
                        st.error("Error retrieving answer from backend.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
else:
    st.info("👈 Paste a YouTube URL in the sidebar and click **Process Video** to start chatting!")