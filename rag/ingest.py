import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def extract_text(pdf_path):
    # open the pdf and read all pages
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text = text + page.get_text()
    doc.close()
    return text


def chunk_text(text):
    # split text into smaller pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks


def get_embedder():
    # load the embedding model
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedder


def ingest_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        embedder = get_embedder()
        vectorstore = FAISS.from_texts(chunks, embedder)
        vectorstore.save_local("vectorstore/")
        print("done - " + str(len(chunks)) + " chunks saved")
    except Exception as e:
        print("error during ingest: " + str(e))
        raise e