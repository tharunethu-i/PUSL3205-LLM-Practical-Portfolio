
import csv
import re
import time
from pathlib import Path

from ingest import get_vectorstore, get_all_documents
from rag_pipeline import (
    bm25_search,
    reciprocal_rank_fusion,
    rerank_documents
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Evaluation" / "evaluation_dataset.csv"
OUTPUT_PATH = BASE_DIR / "Evaluation" / "part_b_results.csv"

# Retrieval settings - unchanged
DENSE_K = 20
BM25_K = 20
RERANK_K = 40
FINAL_K = 5

# Number of unique pages to display during diagnostics
DIAGNOSTIC_PAGE_LIMIT = 20


def get_expected_pages(expected_source):
    """Extract relevant page numbers from the evaluation dataset."""

    if not expected_source or expected_source.strip().lower() == "none":
        return set()

    return {
        int(page)
        for page in re.findall(r"\d+", expected_source)
    }


def get_page(document):
    """Get the actual PDF page label."""

    metadata = document.metadata

    if metadata.get("page_label") is not None:
        return str(metadata["page_label"])

    if metadata.get("page") is not None:
        return str(int(metadata["page"]) + 1)

    return None


def get_filename(document):
    """Get the source PDF filename."""

    return (
        document.metadata.get("filename")
        or document.metadata.get("source")
        or "Unknown"
    )


def get_handbook_pages(documents, limit=DIAGNOSTIC_PAGE_LIMIT):
    """Return unique handbook pages in their retrieval order."""

    pages = []
    seen_pages = set()

    for document in documents:

        if "COMPUTING AWARD HANDBOOK" not in get_filename(document).upper():
            continue

        page = get_page(document)

        if page is None:
            continue

        page = int(page)

        if page not in seen_pages:
            seen_pages.add(page)
            pages.append(page)

        if len(pages) >= limit:
            break

    return pages


def print_stage_diagnostics(stage_name, documents, expected_pages):
    """Display retrieved pages and positions of expected pages."""

    pages = get_handbook_pages(documents)

    print(f"\n--- {stage_name} ---")
    print("Retrieved pages:", pages)

    for expected_page in sorted(expected_pages):

        if expected_page in pages:
            rank = pages.index(expected_page) + 1
            print(f"Expected Page {expected_page}: Found at position {rank}")
        else:
            print(f"Expected Page {expected_page}: NOT FOUND")


def calculate_metrics(ranked_documents, expected_pages):
    """Calculate page-level Recall@5 and MRR."""

    if not expected_pages:
        return None, None, []

    retrieved_pages = get_handbook_pages(
        ranked_documents,
        limit=FINAL_K
    )

    relevant_retrieved = sum(
        1 for page in retrieved_pages if page in expected_pages
    )

    recall = relevant_retrieved / len(expected_pages)

    reciprocal_rank = 0.0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            reciprocal_rank = 1 / rank
            break

    return recall, reciprocal_rank, retrieved_pages


def main():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    print("Loading evaluation dataset...")

    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("No evaluation questions found.")

    print("Loading vector database and documents...")

    vectorstore = get_vectorstore()
    all_documents = get_all_documents()

    print(f"Total indexed chunks: {len(all_documents)}")

    results = []

    for row in rows:

        question = row["Question"].strip()

        expected_pages = get_expected_pages(
            row.get("Expected_Source", "")
        )

        print("\n" + "=" * 80)
        print("Question:", question)

        # Unsupported questions are excluded from retrieval metrics.
        if not expected_pages:

            print("Unsupported question: skipped for retrieval metrics")

            results.append({
                "Question": question,
                "Recall@5": "N/A",
                "MRR": "N/A",
                "Retrieval_Latency_Seconds": "N/A",
                "Top_5_Pages": "N/A"
            })

            continue

        start_time = time.perf_counter()

        # 1. Dense vector retrieval
        dense_results = vectorstore.similarity_search(
            question,
            k=DENSE_K
        )

        # 2. BM25 keyword retrieval
        bm25_results = bm25_search(
            question,
            all_documents,
            k=BM25_K
        )

        # 3. Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            rrf_k=60
        )

        # 4. Weighted reranking using:
        #    Dense retrieval rank + BM25 rank + CrossEncoder rank
        ranked_documents = rerank_documents(
            question,
            fused_results[:RERANK_K],
            dense_results=dense_results,
            bm25_results=bm25_results
        )

        latency = time.perf_counter() - start_time

        # Detailed diagnostics for multi-document questions
        if len(expected_pages) > 1:

            print("\nMULTI-DOCUMENT RETRIEVAL DIAGNOSTICS")
            print("Expected relevant pages:", sorted(expected_pages))

            print_stage_diagnostics(
                "1. DENSE RETRIEVAL",
                dense_results,
                expected_pages
            )

            print_stage_diagnostics(
                "2. BM25 RETRIEVAL",
                bm25_results,
                expected_pages
            )

            print_stage_diagnostics(
                "3. RRF FUSED RESULTS",
                fused_results[:RERANK_K],
                expected_pages
            )

            print_stage_diagnostics(
                "4. WEIGHTED CROSSENCODER RERANKING",
                ranked_documents,
                expected_pages
            )

        # 5. Calculate evaluation metrics
        recall, mrr, top_pages = calculate_metrics(
            ranked_documents,
            expected_pages
        )

        top_5_text = "; ".join(
            f"Page {page}" for page in top_pages
        )

        print("\nFINAL EVALUATION")
        print("Recall@5:", recall)
        print("MRR:", mrr)
        print("Retrieval latency:", round(latency, 4), "seconds")
        print("Top 5:", top_5_text)

        results.append({
            "Question": question,
            "Recall@5": round(recall, 4),
            "MRR": round(mrr, 4),
            "Retrieval_Latency_Seconds": round(latency, 4),
            "Top_5_Pages": top_5_text
        })

    # Save individual question results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as file:

        fieldnames = [
            "Question",
            "Recall@5",
            "MRR",
            "Retrieval_Latency_Seconds",
            "Top_5_Pages"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    # Calculate averages
    evaluated_results = [
        result for result in results
        if result["Recall@5"] != "N/A"
    ]

    if evaluated_results:

        average_recall = sum(
            result["Recall@5"] for result in evaluated_results
        ) / len(evaluated_results)

        average_mrr = sum(
            result["MRR"] for result in evaluated_results
        ) / len(evaluated_results)

        average_latency = sum(
            result["Retrieval_Latency_Seconds"]
            for result in evaluated_results
        ) / len(evaluated_results)

        print("\n========== PART B SUMMARY ==========")

        print("Questions evaluated:", len(evaluated_results))
        print("Average Recall@5:", round(average_recall, 4))
        print("Average MRR:", round(average_mrr, 4))

        print(
            "Average retrieval latency:",
            round(average_latency, 4),
            "seconds"
        )

        print("Results saved to:", OUTPUT_PATH)

    else:
        print("No supported questions were evaluated.")


if __name__ == "__main__":
    main()
