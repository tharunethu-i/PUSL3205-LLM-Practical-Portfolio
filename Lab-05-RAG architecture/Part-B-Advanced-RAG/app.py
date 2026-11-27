
import streamlit as st

from ingest import ingest_pdf, list_documents
from rag_pipeline import ask_university


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="University AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University AI Assistant")

st.caption(
    "Document-based University Knowledge System "
    "powered by Hybrid Retrieval and Reranking"
)


# ==================================================
# 2. SIDEBAR - DOCUMENT MANAGEMENT
# ==================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    # PDF upload
    uploaded_files = st.file_uploader(
        "Upload university PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Process uploaded documents
    if st.button(
        "Process Documents",
        type="primary",
        use_container_width=True
    ):

        if uploaded_files:

            progress = st.progress(0)

            successful_uploads = 0

            for i, uploaded_file in enumerate(uploaded_files):

                try:

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        message = ingest_pdf(uploaded_file)

                    st.success(message)

                    successful_uploads += 1

                except Exception as e:

                    st.error(
                        f"Failed: {uploaded_file.name} - {e}"
                    )

                progress.progress(
                    (i + 1) / len(uploaded_files)
                )

            if successful_uploads > 0:

                st.success(
                    f"{successful_uploads} document(s) processed."
                )

                st.rerun()

        else:

            st.warning(
                "Please select PDF files first."
            )

    st.divider()

    # Display stored documents
    st.subheader("📄 Stored Documents")

    documents = list_documents()

    if documents:

        st.write(
            f"Total PDF files: {len(documents)}"
        )

        for doc in documents:

            st.write(f"📄 {doc}")

    else:

        st.info(
            "No PDFs stored yet. Upload documents to get started."
        )

    st.divider()

    # ==================================================
    # 3. DOCUMENT SEARCH SETTINGS
    # ==================================================

    st.header("🔎 Search Settings")

    search_all_documents = st.checkbox(
        "Search all uploaded documents",
        value=True,
        help=(
            "When enabled, the assistant searches the "
            "entire indexed knowledge base."
        )
    )

    selected_filenames = None

    if search_all_documents:

        st.info(
            "The assistant will search all uploaded documents."
        )

    else:

        if documents:

            selected_filenames = st.multiselect(
                "Select documents to search",
                options=documents,
                default=[],
                help=(
                    "The assistant will search only the "
                    "documents selected here."
                )
            )

            if selected_filenames:

                st.success(
                    f"{len(selected_filenames)} document(s) selected."
                )

            else:

                st.warning(
                    "Please select at least one document."
                )

        else:

            st.warning(
                "Upload and process documents before selecting them."
            )


# ==================================================
# 4. CHAT INTERFACE
# ==================================================

st.header("💬 Ask the University Assistant")

st.write(
    "Ask questions about your uploaded university documents."
)


# Initialise chat history
if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Retrieved Sources"):

                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.write(
                        f"[Source {i}] "
                        f"{source['document']} "
                        f"— Page {source['page']}"
                    )


# ==================================================
# 5. USER QUESTION
# ==================================================

question = st.chat_input(
    "Ask a question about the university..."
)


if question:

    # Check that documents are available
    if not documents:

        st.warning(
            "Please upload and process university PDF documents first."
        )

    # Check selected-document mode
    elif (
        not search_all_documents
        and not selected_filenames
    ):

        st.warning(
            "Please select at least one document to search."
        )

    else:

        # Save user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.write(question)

        # Select search scope
        if search_all_documents:

            search_scope = None

            scope_description = "All uploaded documents"

        else:

            search_scope = selected_filenames

            scope_description = (
                "Selected documents: "
                + ", ".join(selected_filenames)
            )

        # Generate assistant answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching, retrieving and reranking documents..."
            ):

                try:

                    result = ask_university(
                        question,
                        selected_filenames=search_scope
                    )

                    answer = result["answer"]

                    sources = result["sources"]

                    st.markdown(answer)

                    # Display search scope
                    st.caption(
                        f"🔎 Search scope: {scope_description}"
                    )

                    # Display retrieved sources
                    if sources:

                        with st.expander(
                            "📚 View Retrieved Sources"
                        ):

                            for i, source in enumerate(
                                sources,
                                start=1
                            ):

                                st.write(
                                    f"[Source {i}] "
                                    f"{source['document']} "
                                    f"— Page {source['page']}"
                                )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(
                        f"An error occurred: {e}"
                    )
