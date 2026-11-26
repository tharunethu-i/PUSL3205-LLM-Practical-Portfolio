# Part A – Basic RAG Implementation

## 1. Project Overview

This project implements a University Knowledge Assistant using Retrieval-Augmented Generation (RAG). It retrieves relevant information from uploaded university PDF documents and generates answers based on the retrieved content.

## 2. Technologies Used

- Python 3.12
- Ollama
- Llama 3.2 (3B)
- Qwen3 Embedding (0.6B)
- LangChain
- ChromaDB
- Streamlit
- Sentence Transformers

## 3. Main Features

- PDF document loading and processing
- Text splitting and embedding generation
- Vector-based document retrieval
- Cross-encoder reranking
- Context-based answer generation
- Multiple PDF support
- User PDF uploads
- Source references

## 4. Project Files

| File | Description |
|---|---|
| `app.py` | Streamlit user interface |
| `ingest.py` | PDF processing and document ingestion |
| `rag_pipeline.py` | Retrieval and answer generation |
| `rebuild_database.py` | Rebuilds the vector database |
| `requirements.txt` | Python dependencies |

## 5. Installation and Setup

1. Install Python 3.12 and Ollama.
2. Create and activate a virtual environment.
3. Install the required Python packages:

   `pip install -r requirements.txt`

4. Download the required Ollama models:

   `ollama pull llama3.2:3b`

   `ollama pull qwen3-embedding:0.6b`

5. Start the application:

   `streamlit run app.py`

## 6. RAG Workflow

PDF Documents → Text Extraction → Text Splitting → Embedding Generation → ChromaDB → Similarity Retrieval → Cross-Encoder Reranking → LLM Answer Generation → User Response

## 7. Part A Status

This section presents the baseline RAG implementation, which will be used for comparison with the advanced RAG architecture developed in Part B.
