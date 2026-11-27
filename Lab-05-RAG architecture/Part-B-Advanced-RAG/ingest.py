
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


DOCUMENTS_FOLDER = "documents"
VECTORSTORE_FOLDER = "chroma_db"
COLLECTION_NAME = "university_documents"


def get_embeddings():
    """Load the Ollama embedding model."""

    return OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )


def get_vectorstore():
    """Connect to the persistent ChromaDB collection."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=VECTORSTORE_FOLDER
    )


def split_documents(documents):
    """Split PDF pages into overlapping text chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return text_splitter.split_documents(documents)


def ingest_pdf(uploaded_file):
    """
    Save an uploaded PDF, extract its text, split it
    into chunks, and store the chunks in ChromaDB.
    """

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    filename = os.path.basename(uploaded_file.name)

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files can be uploaded.")

    file_path = os.path.join(
        DOCUMENTS_FOLDER,
        filename
    )

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for document in pages:

        original_page = document.metadata.get("page", 0)

        try:
            page_number = int(original_page) + 1
        except (TypeError, ValueError):
            page_number = original_page

        document.metadata["filename"] = filename
        document.metadata["page_label"] = str(page_number)
        document.metadata["source"] = filename

    chunks = split_documents(pages)

    for chunk in chunks:

        chunk.metadata["filename"] = filename
        chunk.metadata["source"] = filename

        if "page_label" not in chunk.metadata:

            original_page = chunk.metadata.get("page", 0)

            try:
                chunk.metadata["page_label"] = str(
                    int(original_page) + 1
                )
            except (TypeError, ValueError):
                chunk.metadata["page_label"] = str(original_page)

    vectorstore = get_vectorstore()

    # Remove old chunks for this PDF before re-indexing.
    vectorstore.delete(
        where={"filename": filename}
    )

    vectorstore.add_documents(chunks)

    return (
        f"Successfully processed {filename} "
        f"({len(pages)} pages, {len(chunks)} chunks)"
    )


def list_documents():
    """Return the names of PDFs in the documents folder."""

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    return sorted([
        filename
        for filename in os.listdir(DOCUMENTS_FOLDER)
        if filename.lower().endswith(".pdf")
    ])


def get_all_documents():
    """Retrieve all indexed document chunks from ChromaDB."""

    vectorstore = get_vectorstore()

    data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    documents = []

    contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    for content, metadata in zip(contents, metadatas):

        metadata = metadata or {}

        if "filename" not in metadata:
            metadata["filename"] = os.path.basename(
                str(metadata.get("source", "Unknown document"))
            )

        if "page_label" not in metadata:

            original_page = metadata.get("page", 0)

            try:
                metadata["page_label"] = str(
                    int(original_page) + 1
                )
            except (TypeError, ValueError):
                metadata["page_label"] = str(original_page)

        documents.append(
            Document(
                page_content=content or "",
                metadata=metadata
            )
        )

    return documents
