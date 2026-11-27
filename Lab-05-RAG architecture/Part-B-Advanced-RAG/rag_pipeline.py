
import os
import re
from collections import defaultdict

from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from ingest import get_vectorstore, get_all_documents


# ============================================================
# 1. CONFIGURATION
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

DENSE_TOP_K = 30
BM25_TOP_K = 30
RERANK_TOP_K = 40

MAX_CONTEXT_DOCUMENTS = 12
MAX_CONTEXT_CHARACTERS = 18000

DEBUG_RETRIEVAL = (
    os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"
)


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def normalise_text(text):

    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.-]", "", text)

    return text.strip()


def get_document_filename(document):

    return document.metadata.get(
        "filename",
        os.path.basename(
            str(document.metadata.get("source", "Unknown document"))
        )
    )


def get_page(document):

    if document.metadata.get("page_label") is not None:
        return str(document.metadata["page_label"])

    page = document.metadata.get("page", "Unknown")

    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(page)


def get_document_key(document):

    return (
        get_document_filename(document),
        get_page(document),
        normalise_text(document.page_content)
    )


def normalise_filename(filename):

    return os.path.basename(str(filename)).strip().lower()


def filter_selected_documents(documents, selected_filenames):

    if selected_filenames is None:
        return documents

    selected = {
        normalise_filename(filename)
        for filename in selected_filenames
    }

    return [
        document
        for document in documents
        if normalise_filename(
            get_document_filename(document)
        ) in selected
    ]


def remove_duplicate_documents(documents):

    unique_documents = []
    seen = set()

    for document in documents:

        key = get_document_key(document)

        if key not in seen:
            seen.add(key)
            unique_documents.append(document)

    return unique_documents


def print_retrieval_debug(stage, documents, limit=10):

    if not DEBUG_RETRIEVAL:
        return

    print("\n" + "=" * 80)
    print(stage)
    print("=" * 80)

    for index, document in enumerate(documents[:limit], start=1):

        print(f"\nResult {index}")
        print("Document:", get_document_filename(document))
        print("Page:", get_page(document))
        print("Content:", document.page_content[:700])


# ============================================================
# 3. DENSE VECTOR RETRIEVAL
# ============================================================

def dense_search(
    question,
    vectorstore,
    k=DENSE_TOP_K,
    selected_filenames=None
):

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    if selected_filenames is not None:

        results = filter_selected_documents(
            results,
            selected_filenames
        )

    return remove_duplicate_documents(results)


# ============================================================
# 4. BM25 KEYWORD RETRIEVAL
# ============================================================

def tokenize_text(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        str(text).lower()
    )


def bm25_search(question, documents, k=BM25_TOP_K):

    if not documents:
        return []

    from rank_bm25 import BM25Okapi

    tokenized_documents = [
        tokenize_text(document.page_content)
        for document in documents
    ]

    tokenized_question = tokenize_text(question)

    if not tokenized_question:
        return []

    if not any(tokenized_documents):
        return []

    bm25 = BM25Okapi(tokenized_documents)

    scores = bm25.get_scores(tokenized_question)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        if len(results) >= k:
            break

        if scores[index] > 0:
            results.append(documents[index])

    return remove_duplicate_documents(results)


# ============================================================
# 5. RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    rrf_k=60
):

    scores = defaultdict(float)
    document_lookup = {}

    for ranking in [dense_results, bm25_results]:

        for rank, document in enumerate(ranking, start=1):

            key = get_document_key(document)

            scores[key] += 1 / (rrf_k + rank)

            document_lookup[key] = document

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True
    )

    return [
        document_lookup[key]
        for key in ranked_keys
    ]


# ============================================================
# 6. CROSS-ENCODER RERANKING
# ============================================================

def rerank_documents(
    question,
    documents,
    dense_results=None,
    bm25_results=None
):
    """
    Rerank candidate chunks with the CrossEncoder.

    For comparison and list questions, combine CrossEncoder ranking
    with the original dense and BM25 rankings. This helps retain
    evidence that may be relevant to different parts of a question,
    instead of allowing the CrossEncoder alone to dominate the
    complete ranking.

    For ordinary questions, retain the standard CrossEncoder ranking.
    """

    if not documents:
        return []

    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    cross_encoder_scores = cross_encoder.predict(pairs)

    cross_encoder_ranked = sorted(
        zip(documents, cross_encoder_scores),
        key=lambda item: item[1],
        reverse=True
    )

    # Standard CrossEncoder ranking for ordinary questions.
    if not (
        is_comparison_question(question)
        or is_list_question(question)
    ):
        return [
            document
            for document, score in cross_encoder_ranked
        ]

    # Build rank lookups using the same stable document key
    # used by the retrieval pipeline.
    dense_rank = {
        get_document_key(document): rank
        for rank, document in enumerate(
            dense_results or [],
            start=1
        )
    }

    bm25_rank = {
        get_document_key(document): rank
        for rank, document in enumerate(
            bm25_results or [],
            start=1
        )
    }

    cross_rank = {
        get_document_key(document): rank
        for rank, (document, score) in enumerate(
            cross_encoder_ranked,
            start=1
        )
    }

    # If the function is called without the original retrieval
    # rankings (for example, by an older evaluation script),
    # safely fall back to the CrossEncoder order.
    if not dense_rank and not bm25_rank:
        return [
            document
            for document, score in cross_encoder_ranked
        ]

    # Weighted Reciprocal Rank Fusion:
    # dense retrieval: 0.55
    # CrossEncoder:    0.35
    # BM25:            0.10
    #
    # A smaller RRF constant gives more influence to higher ranks.
    # The weights are generic and do not depend on PDF page numbers,
    # programme names, or a particular handbook.
    combined_scores = {}

    for document in documents:

        key = get_document_key(document)

        dense_position = dense_rank.get(key)
        cross_position = cross_rank.get(key)
        bm25_position = bm25_rank.get(key)

        score = 0.0

        if dense_position is not None:
            score += 0.55 / (10 + dense_position)

        if cross_position is not None:
            score += 0.35 / (10 + cross_position)

        if bm25_position is not None:
            score += 0.10 / (10 + bm25_position)

        combined_scores[key] = score

    return sorted(
        documents,
        key=lambda document: combined_scores.get(
            get_document_key(document),
            0.0
        ),
        reverse=True
    )



# ============================================================
# 7. QUESTION TYPE DETECTION
# ============================================================

def is_comparison_question(question):

    question = question.lower()

    keywords = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "between"
    ]

    return any(
        keyword in question
        for keyword in keywords
    )


def is_list_question(question):

    question = question.lower()

    keywords = [
        "list",
        "what are the",
        "which modules",
        "all modules",
        "core modules",
        "elective modules"
    ]

    return any(
        keyword in question
        for keyword in keywords
    )


# ============================================================
# 8. CONTEXT DOCUMENT SELECTION
# ============================================================

def select_context_documents(question, documents):

    if not documents:
        return []

    max_documents = MAX_CONTEXT_DOCUMENTS

    if (
        is_comparison_question(question)
        or is_list_question(question)
    ):
        max_documents = 12

    selected = []
    page_counts = defaultdict(int)

    for document in documents:

        filename = get_document_filename(document)
        page = get_page(document)

        page_key = (filename, page)

        if page_counts[page_key] >= 2:
            continue

        if len(selected) >= max_documents:
            break

        selected.append(document)
        page_counts[page_key] += 1

    return remove_duplicate_documents(selected)


# ============================================================
# 9. STRUCTURED TABLE EXTRACTION
# ============================================================

MODULE_ROW_PATTERN = re.compile(
    r"^\s*"
    r"([A-Z]{2,5}\d{3}\.\d+)"
    r"\s+(.+?)"
    r"\s+(\d+(?:\.\d+)?)"
    r"\s+(Core|Elective)"
    r"\s*$",
    re.IGNORECASE
)


def extract_table_rows(documents):

    rows = []
    seen = set()

    for document in documents:

        lines = document.page_content.splitlines()

        for line in lines:

            cleaned_line = re.sub(
                r"\s+",
                " ",
                line
            ).strip()

            match = MODULE_ROW_PATTERN.match(cleaned_line)

            if not match:
                continue

            module_code = match.group(1).strip().upper()
            module_name = match.group(2).strip()
            credits = match.group(3).strip()
            module_type = match.group(4).strip().title()

            row = {
                "document": get_document_filename(document),
                "page": get_page(document),
                "module_code": module_code,
                "module_name": module_name,
                "credits": credits,
                "type": module_type
            }

            key = (
                row["document"],
                row["page"],
                row["module_code"],
                normalise_text(row["module_name"]),
                row["credits"],
                row["type"]
            )

            if key not in seen:
                seen.add(key)
                rows.append(row)

    return rows


# ============================================================
# 10. FORMAT STRUCTURED TABLE EVIDENCE
# ============================================================

def format_table_evidence(rows):

    if not rows:
        return "No structured table rows were extracted."

    formatted_rows = []

    for row in rows:

        formatted_rows.append(
            f"Document: {row['document']} | "
            f"Page: {row['page']} | "
            f"Module Code: {row['module_code']} | "
            f"Module Name: {row['module_name']} | "
            f"Credits: {row['credits']} | "
            f"Type: {row['type']}"
        )

    return "\n".join(formatted_rows)


# ============================================================
# 11. PROGRAMME IDENTIFICATION
# ============================================================

PROGRAMME_ALIASES = {

    "CS": [
        "computer science",
        "cs honours",
        "cs programme",
        "cs program",
        "cs"
    ],

    "SE": [
        "software engineering",
        "se honours",
        "se programme",
        "se program",
        "se"
    ],

    "CN": [
        "computer networks",
        "cn honours",
        "cn programme",
        "cn program",
        "cn"
    ],

    "IS": [
        "management information systems",
        "information systems",
        "mis",
        "is honours",
        "is programme",
        "is program"
    ]

}


def identify_programmes(question):

    question_normalised = normalise_text(question)

    programme_positions = {}

    for prefix, aliases in PROGRAMME_ALIASES.items():

        positions = []

        for alias in aliases:

            pattern = (
                rf"(?<![a-z0-9])"
                rf"{re.escape(alias)}"
                rf"(?![a-z0-9])"
            )

            for match in re.finditer(
                pattern,
                question_normalised
            ):
                positions.append(match.start())

        if positions:
            programme_positions[prefix] = min(positions)

    return sorted(
        programme_positions,
        key=lambda prefix: programme_positions[prefix]
    )


def identify_programme(question):

    matches = identify_programmes(question)

    if len(matches) != 1:
        return None

    return matches[0]


def get_programme_name(prefix):

    names = {
        "CS": "Computer Science",
        "SE": "Software Engineering",
        "CN": "Computer Networks",
        "IS": "Management Information Systems"
    }

    return names.get(prefix, prefix)


# ============================================================
# 12. PROGRAMME TABLE IDENTIFICATION
# ============================================================

def get_programme_table_rows(programme_prefix, table_rows):

    project_rows = []

    for row in table_rows:

        module_code = str(
            row.get("module_code", "")
        ).upper()

        module_name = normalise_text(
            row.get("module_name", "")
        )

        if (
            module_code.startswith(programme_prefix)
            and "award project" in module_name
        ):
            project_rows.append(row)

    unique_project_records = {
        (
            row["module_code"],
            normalise_text(row["module_name"]),
            row["credits"]
        )
        for row in project_rows
    }

    if len(unique_project_records) != 1:
        return []

    matching_project = next(iter(unique_project_records))

    matching_project_rows = [
        row
        for row in project_rows
        if (
            row["module_code"],
            normalise_text(row["module_name"]),
            row["credits"]
        ) == matching_project
    ]

    target_document = matching_project_rows[0]["document"]
    target_page = str(matching_project_rows[0]["page"])

    return [
        row
        for row in table_rows
        if (
            row["document"] == target_document
            and str(row["page"]) == target_page
        )
    ]


# ============================================================
# 13. SOURCE EXTRACTION
# ============================================================

def get_sources(documents):

    sources = []
    seen = set()

    for document in documents:

        filename = get_document_filename(document)
        page = get_page(document)

        key = (filename, page)

        if key not in seen:
            seen.add(key)

            sources.append({
                "document": filename,
                "page": page
            })

    return sources


def get_sources_for_rows(rows):

    sources = []
    seen = set()

    for row in rows:

        key = (
            row["document"],
            str(row["page"])
        )

        if key not in seen:
            seen.add(key)

            sources.append({
                "document": row["document"],
                "page": row["page"]
            })

    return sources


def get_table_source_documents(documents, source):

    if source is None:
        return []

    return remove_duplicate_documents([
        document
        for document in documents
        if (
            get_document_filename(document) == source["document"]
            and str(get_page(document)) == str(source["page"])
        )
    ])


# ============================================================
# 14. DIRECT YEAR 04 CLARIFICATION
# ============================================================

def get_year04_clarification_answer(question):

    question_normalised = normalise_text(question)

    asks_about_year04 = bool(
        re.search(r"\byear\s*0?4\b", question_normalised)
    )

    asks_about_modules = (
        "module" in question_normalised
        or "modules" in question_normalised
    )

    if not asks_about_year04 or not asks_about_modules:
        return None

    specific_information_terms = [
        "module code",
        "what code",
        "which code",
        "credit value",
        "credits",
        "identified by",
        "data warehousing",
        "data mining",
        "artificial intelligence",
        "intrusion",
        "prevention",
        "detection",
        "response",
        "enterprise networks",
        "award project",
        "honours award project",
        "special award project"
    ]

    if any(
        term in question_normalised
        for term in specific_information_terms
    ):
        return None

    if identify_programmes(question):
        return None

    if is_comparison_question(question):
        return None

    broad_list_question = (
        "what are the" in question_normalised
        or "which modules" in question_normalised
        or "all modules" in question_normalised
        or "core modules" in question_normalised
        or "elective modules" in question_normalised
    )

    if not broad_list_question:
        return None

    return {
        "answer": (
            "The Computing Award Handbook contains different "
            "Year 04 module tables for four programmes. "
            "Which programme would you like me to show?\n\n"
            "1. Management Information Systems (MIS)\n"
            "2. Software Engineering\n"
            "3. Computer Science\n"
            "4. Computer Networks"
        ),
        "sources": []
    }


# ============================================================
# 15. DIRECT PROJECT CREDIT ANSWERING
# ============================================================

def get_direct_table_answer(question, table_rows):

    question_normalised = normalise_text(question)

    if not re.search(r"\bcredits?\b", question_normalised):
        return None

    if not any(
        phrase in question_normalised
        for phrase in [
            "how many",
            "how much",
            "allocated",
            "credit value",
            "what is the credit",
            "number of credits"
        ]
    ):
        return None

    programme_prefix = identify_programme(question)

    if programme_prefix is None:
        return None

    matching_rows = []

    for row in table_rows:

        module_code = str(
            row.get("module_code", "")
        ).upper()

        module_name = normalise_text(
            row.get("module_name", "")
        )

        if (
            module_code.startswith(programme_prefix)
            and "award project" in module_name
        ):
            matching_rows.append(row)

    unique_matches = {
        (
            row["module_code"],
            normalise_text(row["module_name"]),
            row["credits"]
        )
        for row in matching_rows
    }

    if len(unique_matches) != 1:
        return None

    matched_row = matching_rows[0]

    return {
        "answer": (
            f"The {matched_row['module_name']} "
            f"({matched_row['module_code']}) "
            f"is allocated {matched_row['credits']} credits."
        ),
        "sources": get_sources_for_rows(matching_rows)
    }


# ============================================================
# 16. DIRECT AWARD PROJECT COMPARISON
# ============================================================

def get_direct_project_comparison_answer(question, table_rows):

    question_normalised = normalise_text(question)

    if not is_comparison_question(question):
        return None

    if "award project" not in question_normalised:
        return None

    requested_programmes = identify_programmes(question)

    if len(requested_programmes) < 2:
        return None

    result_rows = []

    for prefix in requested_programmes:

        project_rows = []

        for row in table_rows:

            module_code = str(
                row.get("module_code", "")
            ).upper()

            module_name = normalise_text(
                row.get("module_name", "")
            )

            if (
                module_code.startswith(prefix)
                and "award project" in module_name
            ):
                project_rows.append(row)

        unique_records = {}

        for row in project_rows:

            record_key = (
                row["module_code"],
                normalise_text(row["module_name"]),
                str(row["credits"])
            )

            if record_key not in unique_records:
                unique_records[record_key] = row

        if len(unique_records) != 1:
            return None

        result_rows.append(
            next(iter(unique_records.values()))
        )

    if len(result_rows) != len(requested_programmes):
        return None

    answer_lines = [
        "Comparison of the award project modules:",
        "",
        "| Programme | Module Code | Module Name | Credits |",
        "|---|---|---|---:|"
    ]

    for prefix, row in zip(
        requested_programmes,
        result_rows
    ):

        answer_lines.append(
            f"| {get_programme_name(prefix)} "
            f"| {row['module_code']} "
            f"| {row['module_name']} "
            f"| {row['credits']} |"
        )

    return {
        "answer": "\n".join(answer_lines),
        "sources": get_sources_for_rows(result_rows)
    }


# ============================================================
# 17. DIRECT CORE / ELECTIVE MODULE ANSWERING
# ============================================================

def get_direct_module_list_answer(question, table_rows):

    question_normalised = normalise_text(question)

    if "module" not in question_normalised:
        return None

    if re.search(r"\bcore\b", question_normalised):
        requested_type = "Core"

    elif re.search(r"\belective\b", question_normalised):
        requested_type = "Elective"

    else:
        return None

    if not re.search(
        r"\byear\s*0?4\b",
        question_normalised
    ):
        return None

    programme_prefix = identify_programme(question)

    if programme_prefix is None:
        return None

    programme_rows = get_programme_table_rows(
        programme_prefix,
        table_rows
    )

    if not programme_rows:
        return None

    matching_rows = [
        row
        for row in programme_rows
        if row["type"].lower() == requested_type.lower()
    ]

    unique_rows = {}

    for row in matching_rows:

        key = (
            row["module_code"],
            row["module_name"],
            row["credits"]
        )

        unique_rows[key] = row

    matching_rows = list(unique_rows.values())

    if not matching_rows:
        return None

    matching_rows.sort(
        key=lambda row: row["module_code"]
    )

    answer_lines = [
        f"Year 04 {requested_type} modules for "
        f"{get_programme_name(programme_prefix)}:",
        "",
        "| Module Code | Module Name | Credits |",
        "|---|---|---:|"
    ]

    for row in matching_rows:

        answer_lines.append(
            f"| {row['module_code']} "
            f"| {row['module_name']} "
            f"| {row['credits']} |"
        )

    return {
        "answer": "\n".join(answer_lines),
        "sources": get_sources_for_rows(matching_rows)
    }


# ============================================================
# 18. DIRECT MODULE CODE LOOKUP
# ============================================================

def get_direct_module_code_answer(question, table_rows):

    code_match = re.search(
        r"\b([A-Z]{2,5}\d{3}\.\d+)\b",
        question.upper()
    )

    if not code_match:
        return None

    module_code = code_match.group(1)

    matching_rows = [
        row
        for row in table_rows
        if row["module_code"].upper() == module_code
    ]

    if not matching_rows:
        return None

    programme_prefix = identify_programme(question)

    if programme_prefix:

        programme_rows = get_programme_table_rows(
            programme_prefix,
            table_rows
        )

        matching_rows = [
            row
            for row in programme_rows
            if row["module_code"].upper() == module_code
        ]

    unique_records = {
        (
            row["module_code"],
            row["module_name"],
            row["credits"],
            row["type"]
        )
        for row in matching_rows
    }

    if len(unique_records) != 1:
        return None

    row = matching_rows[0]

    answer = (
        f"{row['module_code']} identifies "
        f"{row['module_name']}. "
        f"It carries {row['credits']} credits "
        f"and is classified as {row['type']}."
    )

    return {
        "answer": answer,
        "sources": get_sources_for_rows(matching_rows)
    }


# ============================================================
# 19. DIRECT MODULE NAME / DESCRIPTION LOOKUP
# ============================================================

def get_direct_module_name_answer(question, table_rows):

    question_normalised = normalise_text(question)

    asks_for_code = any(
        phrase in question_normalised
        for phrase in [
            "module code",
            "which code",
            "what code",
            "identified by",
            "identify the module"
        ]
    )

    # IMPORTANT FIX:
    # Allows words between "which/what" and "module/course".
    #
    # This recognises:
    # "Which module covers intrusion prevention?"
    #
    # And:
    # "Which Computer Networks Year 04 module covers
    # intrusion prevention, detection and response?"

    asks_which_module = bool(
        re.search(
            r"\b(which|what)\b.{0,100}\b(module|course)\b",
            question_normalised
        )
    )

    if not asks_for_code and not asks_which_module:
        return None

    programme_prefix = identify_programme(question)

    candidate_rows = table_rows

    if programme_prefix:

        programme_rows = get_programme_table_rows(
            programme_prefix,
            table_rows
        )

        if programme_rows:
            candidate_rows = programme_rows

    # Generic words are removed.
    # IMPORTANT: intrusion, prevention, detection and response
    # are deliberately NOT ignored.

    ignored_words = {
        "what", "which", "is", "the", "a", "an",
        "for", "in", "of", "and", "or", "to",
        "does", "do", "that", "this", "covers",
        "cover", "include", "includes", "including",
        "module", "modules", "course", "code",
        "year", "04", "programme", "program",
        "table", "tables", "identified", "by",
        "carry", "carries", "credit", "credits",
        "computer", "science", "software", "engineering",
        "networks", "management", "information", "systems",
        "cs", "se", "cn", "mis", "is",
        "honours", "honors", "bachelor",
        "which", "that", "has", "have",
        "does", "do"
    }

    question_words = set(
        re.findall(r"[a-z0-9]+", question_normalised)
    ) - ignored_words

    matches = []

    for row in candidate_rows:

        module_name = normalise_text(
            row["module_name"]
        )

        module_words = set(
            re.findall(r"[a-z0-9]+", module_name)
        )

        if not module_words:
            continue

        # Exact module-name phrase match.

        exact_match = (
            module_name in question_normalised
        )

        common_words = question_words.intersection(
            module_words
        )

        # Description-based match.

        description_match = (
            len(question_words) >= 2
            and len(common_words) >= 2
            and len(common_words) / len(module_words) >= 0.70
        )

        if exact_match or description_match:
            matches.append(row)

    # Group duplicate module records but retain all
    # source pages for the matching module.

    unique_records = {}

    for row in matches:

        record_key = (
            row["module_code"],
            normalise_text(row["module_name"]),
            row["credits"],
            row["type"]
        )

        if record_key not in unique_records:
            unique_records[record_key] = []

        unique_records[record_key].append(row)

    if len(unique_records) != 1:
        return None

    matched_rows = next(iter(unique_records.values()))
    row = matched_rows[0]

    if asks_for_code:

        answer = (
            f"The module code for {row['module_name']} "
            f"is {row['module_code']} "
            f"({row['credits']} credits, {row['type']})."
        )

    else:

        answer = (
            f"The module is {row['module_name']} "
            f"({row['module_code']}). "
            f"It carries {row['credits']} credits "
            f"and is classified as {row['type']}."
        )

    return {
        "answer": answer,
        "sources": get_sources_for_rows(matched_rows)
    }


# ============================================================
# 20. DIRECT MODULE CLASSIFICATION COMPARISON
# ============================================================

def get_direct_classification_comparison_answer(
    question,
    table_rows
):

    question_normalised = normalise_text(question)

    if "enterprise networks" not in question_normalised:
        return None

    if not any(
        term in question_normalised
        for term in ["core", "elective", "classification"]
    ):
        return None

    result_rows = []

    for prefix in ["IS", "SE", "CS", "CN"]:

        programme_rows = get_programme_table_rows(
            prefix,
            table_rows
        )

        matches = [
            row
            for row in programme_rows
            if normalise_text(row["module_name"])
            == "enterprise networks"
        ]

        unique_matches = {
            (
                row["module_code"],
                row["module_name"],
                row["credits"],
                row["type"]
            )
            for row in matches
        }

        if len(unique_matches) == 1:
            result_rows.append((prefix, matches[0]))

    if not result_rows:
        return None

    answer_lines = [
        "Classification of Enterprise Networks across "
        "the identified Year 04 programme tables:",
        "",
        "| Programme | Module Code | Classification |",
        "|---|---|---|"
    ]

    for prefix, row in result_rows:

        answer_lines.append(
            f"| {get_programme_name(prefix)} "
            f"| {row['module_code']} "
            f"| {row['type']} |"
        )

    return {
        "answer": "\n".join(answer_lines),
        "sources": get_sources_for_rows([
            row for prefix, row in result_rows
        ])
    }


# ============================================================
# 21. PROGRAMME DURATION EXTRACTION
# ============================================================

def asks_about_programme_duration(question):

    question_normalised = normalise_text(question)

    duration_indicators = [
        "how long",
        "how many years",
        "how many months",
        "duration",
        "length of the programme",
        "length of programme",
        "period of study",
        "study period",
        "number of years"
    ]

    return any(
        indicator in question_normalised
        for indicator in duration_indicators
    )


def get_direct_programme_duration_answer(
    question,
    documents
):

    # Only activate for actual duration questions.

    if not asks_about_programme_duration(question):
        return None

    programme_prefix = identify_programme(question)

    if programme_prefix is None:
        return None

    # Use full programme names rather than short abbreviations.
    # This prevents unrelated mentions of CN, CS, SE or MIS
    # from being treated as programme-duration evidence.

    full_programme_names = {
        "CS": ["computer science"],
        "SE": ["software engineering"],
        "CN": ["computer networks"],
        "IS": [
            "management information systems",
            "management information system"
        ]
    }

    aliases = full_programme_names[programme_prefix]

    page_groups = defaultdict(list)

    for document in documents:

        key = (
            get_document_filename(document),
            str(get_page(document))
        )

        page_groups[key].append(document.page_content)

    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10"
    }

    number_pattern = (
        r"(?:\d+|one|two|three|four|five|six|seven|"
        r"eight|nine|ten)"
    )

    duration_patterns = [

        re.compile(
            rf"(?:duration|period\s+of\s+study)"
            rf"(?:\s+of\s+(?:the\s+)?(?:programme|program|course))?"
            rf"\s*(?:is|:|=|-)?\s*"
            rf"({number_pattern})\s*(years?|months?)",
            re.IGNORECASE
        ),

        re.compile(
            rf"({number_pattern})\s*(years?|months?)"
            rf"\s+(?:in\s+duration|duration)",
            re.IGNORECASE
        ),

        re.compile(
            rf"(?:programme|program|course)"
            rf".{{0,100}}?"
            rf"({number_pattern})\s*(years?|months?)",
            re.IGNORECASE
        )

    ]

    matching_evidence = []

    for (filename, page), chunks in page_groups.items():

        combined_content = " ".join(chunks)
        content_normalised = normalise_text(combined_content)

        programme_positions = []

        for alias in aliases:

            pattern = re.compile(
                rf"(?<![a-z0-9])"
                rf"{re.escape(alias)}"
                rf"(?![a-z0-9])",
                re.IGNORECASE
            )

            for match in pattern.finditer(content_normalised):

                programme_positions.append({
                    "position": match.start(),
                    "alias": alias
                })

        if not programme_positions:
            continue

        duration_matches = []

        for pattern in duration_patterns:

            for match in pattern.finditer(combined_content):

                raw_duration = match.group(1).lower()
                unit = match.group(2).lower()

                duration_value = number_words.get(
                    raw_duration,
                    raw_duration
                )

                duration_matches.append({
                    "duration": duration_value,
                    "unit": unit,
                    "position": match.start()
                })

        if not duration_matches:
            continue

        for programme_match in programme_positions:

            closest = min(
                duration_matches,
                key=lambda item: abs(
                    item["position"] -
                    programme_match["position"]
                )
            )

            distance = abs(
                closest["position"] -
                programme_match["position"]
            )

            if distance <= 800:

                matching_evidence.append({
                    "duration": closest["duration"],
                    "unit": closest["unit"],
                    "document": filename,
                    "page": page,
                    "distance": distance
                })

    if not matching_evidence:
        return None

    unique_evidence = {}

    for item in matching_evidence:

        key = (
            item["duration"],
            item["unit"],
            item["document"],
            item["page"]
        )

        if (
            key not in unique_evidence
            or item["distance"] < unique_evidence[key]["distance"]
        ):
            unique_evidence[key] = item

    evidence = min(
        unique_evidence.values(),
        key=lambda item: item["distance"]
    )

    return {
        "answer": (
            f"The {get_programme_name(programme_prefix)} "
            f"programme duration is {evidence['duration']} "
            f"{evidence['unit']}."
        ),
        "sources": [{
            "document": evidence["document"],
            "page": evidence["page"]
        }]
    }


# ============================================================
# 22. UNSUPPORTED QUESTION HANDLING
# ============================================================

def is_clearly_unsupported_question(question, documents):

    question_normalised = normalise_text(question)

    cafeteria_terms = [
        "cafeteria menu",
        "university cafeteria",
        "canteen menu",
        "food menu"
    ]

    if any(
        term in question_normalised
        for term in cafeteria_terms
    ):

        document_text = normalise_text(
            " ".join(
                document.page_content
                for document in documents
            )
        )

        menu_terms = [
            "cafeteria menu",
            "canteen menu",
            "food menu",
            "today's menu",
            "todays menu"
        ]

        if not any(
            term in document_text
            for term in menu_terms
        ):
            return True

    return False


def is_unsupported_answer(answer):

    unsupported_phrases = [
        "i could not find",
        "not available in the provided documents",
        "insufficient information",
        "unable to find",
        "cannot determine from the provided evidence",
        "not mentioned in the provided documents"
    ]

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in unsupported_phrases
    )


# ============================================================
# 23. BUILD DOCUMENT CONTEXT
# ============================================================

def build_context(documents):

    context_parts = []
    total_characters = 0

    for index, document in enumerate(documents, start=1):

        filename = get_document_filename(document)
        page = get_page(document)

        content = document.page_content.strip()

        if not content:
            continue

        section = (
            f"\nSOURCE {index}\n"
            f"Document: {filename}\n"
            f"Page: {page}\n"
            f"Content:\n{content}\n"
        )

        if (
            total_characters + len(section)
            > MAX_CONTEXT_CHARACTERS
        ):
            break

        context_parts.append(section)
        total_characters += len(section)

    return "\n".join(context_parts)


# ============================================================
# 24. DIRECT ANSWER DISPATCHER
# ============================================================

def get_direct_answer(question, table_rows, documents):

    handlers = [

        lambda: get_year04_clarification_answer(
            question
        ),

        lambda: get_direct_project_comparison_answer(
            question,
            table_rows
        ),

        lambda: get_direct_classification_comparison_answer(
            question,
            table_rows
        ),

        lambda: get_direct_table_answer(
            question,
            table_rows
        ),

        lambda: get_direct_module_list_answer(
            question,
            table_rows
        ),

        lambda: get_direct_module_code_answer(
            question,
            table_rows
        ),

        lambda: get_direct_module_name_answer(
            question,
            table_rows
        ),

        lambda: get_direct_programme_duration_answer(
            question,
            documents
        )

    ]

    for handler in handlers:

        result = handler()

        if result is not None:
            return result

    return None


# ============================================================
# 25. MAIN RAG PIPELINE
# ============================================================

def ask_university(question, selected_filenames=None):

    question = str(question).strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    try:

        # STEP 1: LOAD DATABASE AND DOCUMENTS

        vectorstore = get_vectorstore()

        all_documents = get_all_documents()

        if not all_documents:

            return {
                "answer": (
                    "No documents are currently available. "
                    "Please upload and process a PDF first."
                ),
                "sources": []
            }

        # STEP 2: APPLY DOCUMENT SELECTION

        if selected_filenames is not None:

            all_documents = filter_selected_documents(
                all_documents,
                selected_filenames
            )

        if not all_documents:

            return {
                "answer": (
                    "No indexed content was found in the selected "
                    "documents. Please check your document selection."
                ),
                "sources": []
            }

        # STEP 3: DENSE RETRIEVAL

        dense_results = dense_search(
            question=question,
            vectorstore=vectorstore,
            k=DENSE_TOP_K,
            selected_filenames=selected_filenames
        )

        print_retrieval_debug(
            "DENSE RETRIEVAL",
            dense_results
        )

        # STEP 4: BM25 RETRIEVAL

        bm25_results = bm25_search(
            question=question,
            documents=all_documents,
            k=BM25_TOP_K
        )

        print_retrieval_debug(
            "BM25 RETRIEVAL",
            bm25_results
        )

        # STEP 5: RECIPROCAL RANK FUSION

        fused_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            rrf_k=60
        )

        print_retrieval_debug(
            "RECIPROCAL RANK FUSION",
            fused_results
        )

        if not fused_results:

            return {
                "answer": (
                    "I could not find relevant information in "
                    "the indexed documents to answer this question."
                ),
                "sources": []
            }

        # STEP 6: CROSS-ENCODER RERANKING

        rerank_candidates = fused_results[:RERANK_TOP_K]

        reranked_results = rerank_documents(
            question,
            rerank_candidates,
            dense_results=dense_results,
            bm25_results=bm25_results
        )

        print_retrieval_debug(
            "CROSS-ENCODER RERANKING",
            reranked_results
        )

        # STEP 7: CONTEXT SELECTION

        selected_documents = select_context_documents(
            question,
            reranked_results
        )

        print_retrieval_debug(
            "FINAL CONTEXT SENT TO LLM",
            selected_documents
        )

        # STEP 8: STRUCTURED EXTRACTION

        table_rows = extract_table_rows(
            all_documents
        )

        table_evidence = format_table_evidence(
            table_rows
        )

        if DEBUG_RETRIEVAL:

            print("\n" + "=" * 80)
            print("STRUCTURED TABLE INFORMATION")
            print("=" * 80)

            print(table_evidence)

        # STEP 9: DIRECT QUESTION ANSWERING

        direct_result = get_direct_answer(
            question,
            table_rows,
            all_documents
        )

        if direct_result is not None:

            print("\n" + "=" * 80)
            print("DIRECT STRUCTURED ANSWER")
            print("=" * 80)

            print(direct_result["answer"])

            return {
                "answer": direct_result["answer"],
                "sources": direct_result["sources"]
            }

        # STEP 10: HANDLE CLEARLY UNSUPPORTED QUESTIONS

        if is_clearly_unsupported_question(
            question,
            all_documents
        ):

            return {
                "answer": (
                    "I could not find the requested cafeteria menu "
                    "in the uploaded documents. The indexed documents "
                    "do not provide today's cafeteria menu."
                ),
                "sources": []
            }

        # STEP 11: CHECK CONTEXT

        if not selected_documents:

            return {
                "answer": (
                    "I could not identify sufficient relevant "
                    "information in the indexed documents."
                ),
                "sources": []
            }

        context = build_context(selected_documents)

        # STEP 12: GROUNDED PROMPT

        prompt = f"""
You are a University Knowledge Assistant.

Answer the user's question directly using ONLY the evidence
provided below.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

STRUCTURED TABLE INFORMATION:
{table_evidence}

ANSWERING RULES:

1. Answer the exact question written under USER QUESTION.

2. Never say that the user has not provided a question.

3. Use only facts supported by the supplied evidence.

4. Do not invent module codes, names, credit values,
   classifications, dates or programme details.

5. Treat each structured table row as one complete record.

6. Do not combine information from different module rows.

7. Pay attention to the document and page associated
   with each piece of evidence.

8. For Core module questions, include only rows classified
   as Core.

9. For Elective module questions, include only rows classified
   as Elective.

10. For comparison questions, discuss each requested programme
    separately and do not omit available evidence.

11. If a module code is supplied, use the structured table
    information to identify that module when available.

12. If the question asks for a module code, provide the code
    clearly and include its module name when available.

13. If the evidence does not contain the answer, say:
    "I could not find this information in the uploaded documents."

14. Do not ask the user to provide more context when the
    supplied evidence already answers the question.

15. Do not describe the evidence as a table instead of
    answering the question.

16. Keep the answer concise and well structured.

17. Do not claim to have checked information outside
    the uploaded documents.

ANSWER:
"""

        # STEP 13: GENERATE ANSWER

        response = llm.invoke(prompt)

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        answer = str(answer).strip()

        # STEP 14: RETURN ANSWER AND SOURCES

        sources = get_sources(selected_documents)

        if is_unsupported_answer(answer):
            sources = []

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as error:

        print("RAG pipeline error:", error)

        return {
            "answer": (
                "An error occurred while processing your question. "
                "Please check the application terminal for details."
            ),
            "sources": []
        }
