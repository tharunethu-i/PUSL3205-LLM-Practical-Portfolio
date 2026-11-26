
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DOCUMENTS_FOLDER = "documents"
VECTORSTORE_FOLDER = "chroma_db"


def get_embeddings():
    return OllamaEmbeddings(model="qwen3-embedding:0.6b")


def get_vectorstore():
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name="university_documents",
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_FOLDER
    )

    return vectorstore


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


def ingest_pdf(uploaded_file):
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    file_path = os.path.join(DOCUMENTS_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for document in documents:
        document.metadata["filename"] = uploaded_file.name

    chunks = split_documents(documents)

    vectorstore = get_vectorstore()

    # Remove previously stored chunks from this PDF
    vectorstore.delete(where={"filename": uploaded_file.name})

    # Add the latest PDF chunks
    vectorstore.add_documents(chunks)

    return f"Successfully processed {uploaded_file.name} ({len(chunks)} chunks)"


def list_documents():
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    documents = []

    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.lower().endswith(".pdf"):
            documents.append(filename)

    return documents
