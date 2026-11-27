# University Knowledge Assistant – Part B
## Hybrid Retrieval and Cross-Encoder Reranking

### 1. Project Overview

This project extends the University Knowledge Assistant developed in Part A. Part B introduces a hybrid information retrieval approach that combines dense vector retrieval, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.

The system allows users to ask questions about uploaded university documents and retrieves relevant information to generate answers using a Retrieval-Augmented Generation (RAG) pipeline.

The application supports searching across multiple uploaded PDF documents rather than relying on a single fixed handbook.

### 2. Objectives

The main objectives of Part B are:

- Implement dense vector retrieval to identify semantically relevant document chunks.
- Implement BM25 keyword retrieval to identify documents containing relevant terms.
- Combine retrieval results using Reciprocal Rank Fusion (RRF).
- Apply Cross-Encoder reranking to improve the relevance of retrieved documents.
- Generate answers using the retrieved document context.
- Evaluate the performance of the hybrid retrieval system against Part A.

### 3. System Architecture

The Part B retrieval pipeline consists of the following stages:

1. **Document ingestion:** PDF documents are loaded and divided into smaller text chunks.
2. **Dense retrieval:** Semantic search retrieves relevant chunks using vector embeddings.
3. **BM25 retrieval:** Keyword-based search retrieves documents based on matching terms.
4. **Reciprocal Rank Fusion:** Dense and BM25 results are combined into a unified ranked list.
5. **Cross-Encoder reranking:** Candidate documents are scored and reordered according to their relevance to the user's question.
6. **Context selection:** The most relevant documents are selected within the context limits.
7. **Answer generation:** The retrieved context is passed to the language model to generate an answer.

### 4. Hybrid Retrieval and Reranking

The system uses the following retrieval techniques:

**Dense Vector Retrieval**

Uses semantic embeddings to identify relevant content even when the wording of a question differs from the document.

**BM25 Keyword Retrieval**

Uses keyword matching to identify relevant document chunks based on terms appearing in the user's query.

**Reciprocal Rank Fusion (RRF)**

Combines the ranked results from dense retrieval and BM25 into a unified list.

**Weighted Cross-Encoder Reranking**

The Cross-Encoder evaluates the relevance of candidate documents to the user's question.

For comparison and list-based questions, the implementation combines three ranking signals:

- Dense retrieval ranking: 0.55
- Cross-Encoder ranking: 0.35
- BM25 ranking: 0.10

This approach aims to retain relevant documents from different retrieval methods while improving their final ranking.

### 5. Technologies and Tools

- Python 3.12.4
- Streamlit
- LangChain
- ChromaDB
- Ollama
- Llama 3.2 (3B) – language model
- Qwen3 Embedding (0.6B) – embedding model
- BM25 (`rank-bm25`)
- Sentence Transformers Cross-Encoder
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- PyPDFLoader
- Pandas

### 6. Project Structure

```text
University_RAG_Assistant part b/
│
├── documents/
│   ├── COMPUTING AWARD HANDBOOK_MOHE (2).pdf
│   └── Week 1.pdf
│
├── Evaluation/
│   ├── evaluation_dataset.csv
│   ├── part_b_results.csv
│   ├── part_b_answer_results.csv
│   ├── part_b_results_previous.csv
│   └── part_b_answer_results_previous.csv
│
├── chroma_db/
│
├── app.py
├── ingest.py
├── rag_pipeline.py
├── rebuild_database.py
├── evaluate_part_b.py
├── evaluate_answers_part_b.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 7. Installation and Execution

**Step 1 – Open the project folder**

Open the `PUSL3205-LLM-Practical-Portfolio` folder and navigate to the Part B project directory.

**Step 2 – Activate the virtual environment**

Windows PowerShell:

```powershell
.\rag_env\Scripts\Activate.ps1
```

**Step 3 – Install dependencies (if not already installed)**

```bash
pip install -r requirements.txt
```

**Step 4 – Start the application**

Ensure that Ollama is running and the required models are available.

```bash
python -m streamlit run app.py
```

The application will be available at:

`http://localhost:8501`
### 8. Evaluation Methodology

The system was evaluated using a dataset containing 15 test questions.

The retrieval evaluation excluded one unsupported cafeteria-menu question, resulting in 14 supported questions.

The following metrics were used:

- **Recall@5:** Measures the proportion of relevant documents retrieved within the top five results.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant document is ranked.
- **Retrieval Latency:** Measures the time required to retrieve and rerank relevant documents.
- **Answer Evaluation:** Records successful and failed answer-generation responses and average response latency.

### 9. Part B Evaluation Results

**Retrieval Evaluation**

| Metric | Result |
|---|---:|
| Total indexed chunks | 75 |
| Supported questions evaluated | 14 |
| Average Recall@5 | 0.9821 |
| Average MRR | 0.9643 |
| Average retrieval latency | 2.7267 seconds |

**Answer Evaluation**

| Metric | Result |
|---|---:|
| Total questions | 15 |
| Successful responses | 15 |
| Failed responses | 0 |
| Average response latency | 5.23 seconds |

The results demonstrate that the application successfully generated responses for all 15 test questions. Factual correctness was also reviewed against the expected information in the uploaded documents.

### 10. Part A and Part B Comparison

The following table compares the available evaluation results from the two implementations.

| Metric | Part A | Part B |
|---|---:|---:|
| Average Recall@5 | 0.9643 | 0.9821 |
| Average MRR | 1.0000 | 0.9643 |
| Average answer latency | 26.64 s | 5.23 s |

Part B achieved a higher average Recall@5, increasing from 0.9643 to 0.9821. This indicates an improvement in retrieving relevant documents within the top five results.

MRR decreased slightly from 1.0000 to 0.9643, indicating that the first relevant result was not ranked first for every question.

The recorded average answer latency was also lower in Part B. However, these latency figures were collected from separate evaluation runs and should be treated as indicative rather than a strictly controlled performance comparison.

### 11. Limitations

- Retrieval performance depends on the quality and content of uploaded documents.
- Some questions may require information that is not available in the indexed documents.
- The ranking of relevant documents may vary depending on the question.
- Answer correctness requires evaluation against the original source documents.
- The application cannot provide real-time information that is absent from the uploaded documents.

### 12. Conclusion

Part B extends the University Knowledge Assistant by implementing a hybrid retrieval approach that combines semantic and keyword-based search with Reciprocal Rank Fusion and Cross-Encoder reranking.

The evaluation achieved an average Recall@5 of 0.9821 and an average MRR of 0.9643 across 14 supported questions. The answer evaluation produced 15 successful responses out of 15 questions.

Overall, the results demonstrate the potential of combining multiple retrieval techniques to improve document retrieval and question-answering performance in a university knowledge assistant.

---

**Author:** Data Science Student  
**Project:** PUSL 3205 – LLM Practical Portfolio, Part B
- Qwen3 Embedding (0.6B)
- Cross-Encoder Reranking
- Llama 3.2 (3B)

## 3. Identified Limitations

The existing system may experience difficulties with:
- Exact academic terminology and module codes
- Information distributed across multiple documents
- Retrieval of relevant information from similar document sections
- Ambiguous questions

## 4. Proposed Improvement

Implement Hybrid Retrieval by combining:
- Dense Vector Search
- BM25 Keyword Search
- Hybrid Result Fusion
- Cross-Encoder Reranking

## 5. Evaluation

The proposed system will be compared with the baseline RAG system using the same evaluation questions and reference answers.

Evaluation will consider retrieval relevance, answer correctness, source accuracy, response time and reliability.

## 6. Project Status

Planning and architecture design.
