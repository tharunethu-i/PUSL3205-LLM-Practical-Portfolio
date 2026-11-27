
import os

from langchain_community.document_loaders import PyPDFLoader
from ingest import get_vectorstore, split_documents


documents_folder = "documents"

vectorstore = get_vectorstore()

pdf_files = [
    filename
    for filename in os.listdir(documents_folder)
    if filename.lower().endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDF document(s).")

for filename in pdf_files:

    file_path = os.path.join(documents_folder, filename)

    print(f"\nProcessing: {filename}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for document in documents:
        document.metadata["filename"] = filename

    chunks = split_documents(documents)

    vectorstore.add_documents(chunks)

    print(f"Successfully indexed {filename}: {len(chunks)} chunks")

print("\nDatabase rebuilding completed!")
