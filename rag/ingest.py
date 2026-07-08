import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# this function opens the pdf and reads each page separately
# it remembers which page each piece of text came from
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        page_number = i + 1
        pages.append({"page": page_number, "text": text})
    doc.close()
    return pages


# this function cuts text into smaller pieces
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks


# this function loads the embedding model
def get_embedder():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedder


# this is the main function that runs when a pdf is uploaded
def ingest_pdf(pdf_path):
    try:
        # read each page separately
        pages = extract_pages(pdf_path)

        # collect all chunks and remember which page they came from
        all_chunks = []
        all_metadata = []

        for page in pages:
            chunks = chunk_text(page["text"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({"page": page["page"]})

        # embed all chunks and save to disk with page metadata
        embedder = get_embedder()
        vectorstore = FAISS.from_texts(
            all_chunks,
            embedder,
            metadatas=all_metadata
        )
        vectorstore.save_local("vectorstore/")

        print("done - " + str(len(all_chunks)) + " chunks saved")

    except Exception as e:
        print("error during ingest: " + str(e))
        raise e