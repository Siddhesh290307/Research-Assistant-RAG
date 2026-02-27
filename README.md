# RAG Research Assistant

This project is a **Retrieval-Augmented Generation (RAG) Research Assistant** built with **Streamlit**. It allows you to ask questions from your indexed research papers and get answers using LLMs.

The papers used in this are all related to geometry and sparse attention in Transformers.

---

## Features

- Chat-style interface for querying research papers.
- **Chunking Modes**:
  - **Sliding Window**: Splits context into overlapping chunks.
  - **Semantic Chunking**: Splits context based on semantic meaning.
- Supports comparison of chunking strategies for question answering.
- Uses **Ollama LLaMA 2 model** (`llama2:latest`) for generation.

---

## Performance Comparison

| Chunking Mode | Avg F1 | Exact Match | Context Recall@k |
|---------------|--------|------------|----------------|
| Sliding       | 0.216  | 0          | 0.333          |
| Semantic      | 0.222  | 0          | 0.167          |

**Analysis**:  
- **Semantic chunking** achieves slightly higher F1 score, indicating better answer quality.  
- **Sliding window** retrieves context more reliably, giving higher context recall.  
- Choice depends on whether **answer quality** or **context coverage** is more important.
- Overall for these documents, sliding window chunking is clearly giving better results.

---

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / Mac
   venv\Scripts\activate     # Windows
   ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Make sure llama2(Ollama) model is installed locally on your system.


