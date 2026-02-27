import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Chunking Mode", ["sliding", "semantic"])
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    st.markdown("---")
    st.markdown("### About")
    st.write("Compare Sliding Window vs Semantic Chunking for research paper QA.")

# ---------------- SESSION ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- MAIN UI ---------------- #
st.title("📚 AI Research Assistant")
st.caption("Ask questions from your indexed research papers.")

# Chat-style input
query = st.chat_input("Ask a question...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.spinner("Retrieving answer..."):
        try:
            response = requests.post(API_URL, params={"query": query, "mode": mode})
            answer = response.json().get("answer", "No answer returned.") if response.status_code == 200 else f"API Error: {response.status_code}"
        except Exception as e:
            answer = f"Connection Error: {str(e)}"
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ---------------- DISPLAY CHAT ---------------- #
for message in st.session_state.chat_history:
    is_user = message["role"] == "user"
    bg_color = "#3b82f6" if is_user else "#e5e7eb"
    text_color = "white" if is_user else "black"
    align = "flex-end" if is_user else "flex-start"

    st.markdown(
        f"""
        <div style='
            display: flex;
            justify-content: {align};
            margin-bottom: 10px;
        '>
            <div style='
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 12px;
                background: {bg_color};
                color: {text_color};
                line-height: 1.5;
                word-wrap: break-word;
            '>
                {message["content"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )