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
        documents.extend(docs)

        os.remove(pdf.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(documents)


    return vectorstore
