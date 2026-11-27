
import csv
import time
from pathlib import Path

from rag_pipeline import ask_university

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "Evaluation" / "evaluation_dataset.csv"
OUTPUT_PATH = BASE_DIR / "Evaluation" / "part_b_answer_results.csv"


def main():

    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        questions = list(reader)

    results = []

    print("Starting Part B answer evaluation...")
    print("Total questions:", len(questions))

    for index, row in enumerate(questions, start=1):

        question = row["Question"].strip()

        print(f"\nQuestion {index}/{len(questions)}:")
        print(question)

        start_time = time.perf_counter()

        try:
            response = ask_university(question)

            latency = time.perf_counter() - start_time

            if isinstance(response, dict):
                answer = response.get("answer", "")
                sources = response.get("sources", [])
            else:
                answer = str(response)
                sources = []

            source_text = str(sources)
            status = "Success"

        except Exception as error:
            latency = time.perf_counter() - start_time
            answer = f"ERROR: {error}"
            source_text = ""
            status = "Error"

        print("Answer:", answer)
        print("Response time:", round(latency, 2), "seconds")

        results.append({
            "Question": question,
            "Expected_Answer": row.get("Expected_Answer", ""),
            "Expected_Source": row.get("Expected_Source", ""),
            "Generated_Answer": answer,
            "Retrieved_Sources": source_text,
            "Response_Latency_Seconds": round(latency, 2),
            "Execution_Status": status,
            "Manual_Correctness": "",
            "Grounding_Assessment": ""
        })

    fieldnames = [
        "Question",
        "Expected_Answer",
        "Expected_Source",
        "Generated_Answer",
        "Retrieved_Sources",
        "Response_Latency_Seconds",
        "Execution_Status",
        "Manual_Correctness",
        "Grounding_Assessment"
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    successful = [
        result for result in results
        if result["Execution_Status"] == "Success"
    ]

    if successful:

        average_latency = sum(
            result["Response_Latency_Seconds"]
            for result in successful
        ) / len(successful)

        print("\n========== PART B ANSWER EVALUATION ==========")
        print("Total questions:", len(results))
        print("Successful responses:", len(successful))
        print("Failed responses:", len(results) - len(successful))
        print("Average response latency:", round(average_latency, 2), "seconds")

    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
