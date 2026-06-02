import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def ingest_pdf(pdf_path: str):
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    embedder = get_embedder()
    vectorstore = FAISS.from_texts(chunks, embedder)
    vectorstore.save_local("vectorstore/")
    print(f"Ingested {len(chunks)} chunks")
