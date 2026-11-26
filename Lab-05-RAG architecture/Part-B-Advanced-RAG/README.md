# Part B – Advanced RAG Implementation

## Project Title
From Naive RAG to Intelligent University Knowledge Retrieval

## 1. Project Overview

This project aims to improve the existing University Knowledge Assistant by implementing an advanced Retrieval-Augmented Generation (RAG) architecture.

The proposed system will investigate Hybrid Retrieval, combining dense vector search and BM25 keyword-based search to improve document retrieval.

## 2. Existing System

The baseline application uses:
- Dense vector retrieval
- ChromaDB
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
