import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def ingest_pdf(pdf_path):
    """
    Reads each page separately and saves its page number
    as metadata for every created chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    all_chunks = []
    all_metadata = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            if not page_text:
                continue

            page_chunks = splitter.split_text(page_text)

            for chunk in page_chunks:
                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        "page": page_number,
                    }
                )

    if not all_chunks:
        raise ValueError("No readable text found in the PDF.")

    embedder = get_embedder()

    vectorstore = FAISS.from_texts(
        all_chunks,
        embedder,
        metadatas=all_metadata,
    )

    vectorstore.save_local("vectorstore/")

    print(f"Done - {len(all_chunks)} chunks saved.")