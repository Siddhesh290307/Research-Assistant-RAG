import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="RAG Research Assistant", layout="wide")

st.title("📚 AI Research Assistant (RAG)")
st.markdown("Ask questions from your uploaded research papers.")

# Session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
query = st.text_input("Enter your question:")

if st.button("Ask") and query.strip():

    with st.spinner("Retrieving answer..."):

        try:
            response = requests.post(API_URL, params={"query": query})
            
            if response.status_code == 200:
                answer = response.json()["answer"]
            else:
                answer = f"API Error: {response.status_code}"

        except Exception as e:
            answer = f"Connection Error: {str(e)}"

    # Save to history
    st.session_state.chat_history.append((query, answer))

# Display chat history
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")