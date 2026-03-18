from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def process_pdfs(pdf_files):
    documents = []

    for pdf in pdf_files:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(pdf.name)
        docs = loader.load()

        # ✅ Check if PDF has content
        if not docs:
            print(f"No content found in {pdf.name}")
            continue

        documents.extend(docs)
        os.remove(pdf.name)

    # ❌ If no documents at all
    if not documents:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    # ❌ If chunks empty → stop
    if not chunks:
        raise ValueError("Text splitting resulted in no chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore
