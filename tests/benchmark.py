import os
import time
import csv
import shutil
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TEST_QUESTIONS = [
    {"question": "What is Microsoft's carbon negative target year?", "correct_answer": "2030"},
    {"question": "What are the Scope 1 CO2 emissions in FY21?", "correct_answer": "94,292 mtCO2e"},
    {"question": "What are the Scope 2 market-based emissions in FY21?", "correct_answer": "163,935 mtCO2e"},
    {"question": "What renewable energy goal did Microsoft set?", "correct_answer": "100% renewable energy by 2025"},
    {"question": "What is Microsoft's water positive target year?", "correct_answer": "2030"},
    {"question": "How many GW of PPAs did Microsoft sign in FY21?", "correct_answer": "5.8 GW"},
    {"question": "What is Microsoft's zero waste target?", "correct_answer": "Zero waste by 2030"},
    {"question": "What year does Microsoft aim to remove all historical carbon emissions by?", "correct_answer": "2050"},
    {"question": "What are the Scope 1 emissions in FY17?", "correct_answer": "82,448 mtCO2e"},
    {"question": "What is Microsoft's goal for its supply chain emissions?", "correct_answer": "Reduce Scope 3 emissions by more than half from 2020 baseline"},
]

CHUNK_SETTINGS = [
    (300,  30,  "small — most accurate, most embeddings, slowest ingest"),
    (500,  50,  "medium — balanced accuracy and speed"),
    (1000, 50,  "large — fewest embeddings, fastest ingest, least accurate"),
    (1000, 200, "large with high overlap — more context per chunk"),
]

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

VECTOR_STORES = ["faiss", "chromadb"]

LLM_MODELS = [
    "meta-llama-3.1-8b-instruct",
    "qwen3-30b-a3b-instruct-2507",
    "deepseek-r1-distill-llama-70b",
    "apertus-70b-instruct-2509",
    "teuken-7b-instruct-research",
]

PDF_PATH = "uploads/report.pdf"
OUTPUT_CSV = "tests/test_results.csv"

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({"page": i + 1, "text": page.get_text()})
    doc.close()
    return pages

def get_chunks_and_metadata(pdf_path, chunk_size, chunk_overlap):
    pages = extract_pages(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    all_metadata = []
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            all_chunks.append(chunk)
            all_metadata.append({"page": page["page"]})
    return all_chunks, all_metadata

def get_embedder(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def build_faiss(chunks, metadata, embedder):
    from langchain_community.vectorstores import FAISS
    path = "vectorstore_bench/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    vs = FAISS.from_texts(chunks, embedder, metadatas=metadata)
    vs.save_local(path)
    return vs

def build_chromadb(chunks, metadata, embedder):
    from langchain_community.vectorstores import Chroma
    path = "chromadb_bench/"
    if os.path.exists(path):
        shutil.rmtree(path)
    vs = Chroma.from_texts(chunks, embedder, metadatas=metadata,
                           collection_name="bench", persist_directory=path)
    return vs

def search_vectorstore(vs, question, k=6):
    results = vs.similarity_search(question, k=k)
    return [{"text": r.page_content, "page": r.metadata.get("page", 1)} for r in results]

def ask_gwdg(question, chunks, model):
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = ("You are a helpful assistant answering questions about a sustainability report.\n"
              "Use the context below to answer as clearly and completely as possible.\n"
              "If the answer is not in the context say: Not found in document.\n\n"
              f"Context:\n{context}\nQuestion: {question}\nAnswer:")
    client = OpenAI(api_key=os.getenv("GWDG_API_KEY"),
                    base_url="https://chat-ai.academiccloud.de/v1")
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content

def word_count(text):
    return len(str(text).split())

def auto_accuracy(correct, response):
    keywords = [w.lower() for w in correct.replace(",", "").split() if len(w) > 2]
    if not keywords:
        return "unknown"
    rl = response.lower()
    ratio = sum(1 for w in keywords if w in rl) / len(keywords)
    if ratio >= 0.8:
        return "likely_yes"
    elif ratio >= 0.4:
        return "partial"
    return "likely_no"

def run_benchmark():
    os.makedirs("tests", exist_ok=True)
    fieldnames = [
        "chunk_size", "chunk_overlap", "chunk_setting_description",
        "embedding_model", "vector_store", "llm_model",
        "question", "correct_answer", "correct_answer_word_count",
        "total_chunks_in_index", "ingest_time_sec", "query_time_sec",
        "system_answer", "system_answer_word_count",
        "sources_page_numbers", "num_sources_returned",
        "auto_accuracy_estimate", "correct_manual", "notes",
    ]
    total_combos = len(CHUNK_SETTINGS) * len(EMBEDDING_MODELS) * len(VECTOR_STORES) * len(LLM_MODELS)
    total_calls = total_combos * len(TEST_QUESTIONS)
    print(f"Total combinations: {total_combos}")
    print(f"Total API calls: {total_calls}")
    print(f"Estimated runtime: ~{round(total_calls * 8 / 60)} minutes\n")
    combo_num = 0
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk_size, chunk_overlap, chunk_desc in CHUNK_SETTINGS:
            for emb_name in EMBEDDING_MODELS:
                emb_short = emb_name.split("/")[-1]
                print(f"\nLoading embedder: {emb_short}")
                embedder = get_embedder(emb_name)
                chunks, metadata = get_chunks_and_metadata(PDF_PATH, chunk_size, chunk_overlap)
                total_chunks = len(chunks)
                print(f"Chunks: {total_chunks}")
                for vs_name in VECTOR_STORES:
                    combo_num += 1
                    print(f"\n[{combo_num}/{total_combos}] chunk={chunk_size}/{chunk_overlap} | {emb_short} | {vs_name}")
                    ingest_start = time.time()
                    try:
                        vs = build_faiss(chunks, metadata, embedder) if vs_name == "faiss" else build_chromadb(chunks, metadata, embedder)
                        ingest_time = round(time.time() - ingest_start, 2)
                        print(f"Ingest: {ingest_time}s")
                    except Exception as e:
                        print(f"Ingest failed: {e}")
                        continue
                    for llm in LLM_MODELS:
                        print(f"  LLM: {llm}")
                        for q in TEST_QUESTIONS:
                            try:
                                t0 = time.time()
                                retrieved = search_vectorstore(vs, q["question"])
                                answer = ask_gwdg(q["question"], retrieved, llm)
                                query_time = round(time.time() - t0, 2)
                                sources = " | ".join([f"p.{c['page']}" for c in retrieved])
                                num_src = len(retrieved)
                            except Exception as e:
                                answer = f"ERROR: {e}"
                                query_time = 0
                                sources = ""
                                num_src = 0
                            acc = auto_accuracy(q["correct_answer"], answer)
                            writer.writerow({
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "chunk_setting_description": chunk_desc,
                                "embedding_model": emb_short,
                                "vector_store": vs_name,
                                "llm_model": llm,
                                "question": q["question"],
                                "correct_answer": q["correct_answer"],
                                "correct_answer_word_count": word_count(q["correct_answer"]),
                                "total_chunks_in_index": total_chunks,
                                "ingest_time_sec": ingest_time,
                                "query_time_sec": query_time,
                                "system_answer": answer,
                                "system_answer_word_count": word_count(answer),
                                "sources_page_numbers": sources,
                                "num_sources_returned": num_src,
                                "auto_accuracy_estimate": acc,
                                "correct_manual": "",
                                "notes": "",
                            })
                            f.flush()
                            print(f"    [{acc}] {q['question'][:45]}... {query_time}s")
                            time.sleep(2)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")
    print("Open the CSV and fill in 'correct_manual': yes / no / partial")

if __name__ == "__main__":
    run_benchmark()
