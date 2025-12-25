import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Load all PDFs in the given directory
def load_all_pdfs(pdf_dir: str):
    all_docs = []
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Annotate each page's metadata
        for page in pages:
            page.metadata["source"] = str(pdf_path.name)  # Just the file name, not full path
        all_docs.extend(pages)
    return all_docs

# Load PDF
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages

# Split into chunks
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Create vector store using FAISS
def create_vectorstore(chunks):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

# Create QA chain
def create_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
