import os, time, csv, shutil
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

# only mpnet — MiniLM already done
EMBEDDING_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
]

# only working models — skip teuken and deepseek
LLM_MODELS = [
    "meta-llama-3.1-8b-instruct",
    "qwen3-30b-a3b-instruct-2507",
    "apertus-70b-instruct-2509",
]

# faiss only — chromadb had disk issues
PDF_PATH = "uploads/report.pdf"
OUTPUT_CSV = "tests/test_results.csv"

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [{"page": i+1, "text": p.get_text()} for i, p in enumerate(doc)]
    doc.close()
    return pages

def get_chunks(pdf_path, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, metadata = [], []
    for page in extract_pages(pdf_path):
        for chunk in splitter.split_text(page["text"]):
            chunks.append(chunk)
            metadata.append({"page": page["page"]})
    return chunks, metadata

def build_faiss(chunks, metadata, embedder):
    path = "vectorstore_bench/"
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    vs = FAISS.from_texts(chunks, embedder, metadatas=metadata)
    vs.save_local(path)
    return vs

def search(vs, question):
    return [{"text": r.page_content, "page": r.metadata.get("page",1)} for r in vs.similarity_search(question, k=6)]

def ask_gwdg(question, chunks, model):
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"Answer using only the context below. If not found say: Not found in document.\n\nContext:\n{context}\nQuestion: {question}\nAnswer:"
    client = OpenAI(api_key=os.getenv("GWDG_API_KEY"), base_url="https://chat-ai.academiccloud.de/v1")
    resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}])
    return resp.choices[0].message.content

def auto_acc(correct, response):
    kw = [w.lower() for w in correct.replace(",","").split() if len(w)>2]
    if not kw: return "unknown"
    ratio = sum(1 for w in kw if w in response.lower()) / len(kw)
    return "likely_yes" if ratio>=0.8 else "partial" if ratio>=0.4 else "likely_no"

def run():
    fields = ["chunk_size","chunk_overlap","chunk_setting_description","embedding_model",
              "vector_store","llm_model","question","correct_answer","correct_answer_word_count",
              "total_chunks_in_index","ingest_time_sec","query_time_sec","system_answer",
              "system_answer_word_count","sources_page_numbers","num_sources_returned",
              "auto_accuracy_estimate","correct_manual","notes"]

    total = len(CHUNK_SETTINGS)*len(EMBEDDING_MODELS)*len(LLM_MODELS)*len(TEST_QUESTIONS)
    print(f"Remaining API calls: {total} — estimated {round(total*7/60)} minutes\n")
    print("Appending to existing test_results.csv...\n")

    # append to existing CSV
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        for chunk_size, chunk_overlap, chunk_desc in CHUNK_SETTINGS:
            for emb_name in EMBEDDING_MODELS:
                emb_short = emb_name.split("/")[-1]
                print(f"\nEmbedder: {emb_short}")
                embedder = HuggingFaceEmbeddings(model_name=emb_name)
                chunks, metadata = get_chunks(PDF_PATH, chunk_size, chunk_overlap)
                total_chunks = len(chunks)
                print(f"Chunks: {total_chunks}")

                print("Building FAISS...")
                t0 = time.time()
                try:
                    vs = build_faiss(chunks, metadata, embedder)
                    ingest_time = round(time.time()-t0, 2)
                    print(f"Ingest: {ingest_time}s")
                except Exception as e:
                    print(f"Failed: {e}")
                    continue

                for llm in LLM_MODELS:
                    print(f"  LLM: {llm}")
                    for q in TEST_QUESTIONS:
                        try:
                            t0 = time.time()
                            retrieved = search(vs, q["question"])
                            answer = ask_gwdg(q["question"], retrieved, llm)
                            query_time = round(time.time()-t0, 2)
                            sources = " | ".join([f"p.{c['page']}" for c in retrieved])
                        except Exception as e:
                            answer = f"ERROR: {e}"
                            query_time = 0
                            sources = ""
                            retrieved = []

                        acc = auto_acc(q["correct_answer"], answer)
                        writer.writerow({
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "chunk_setting_description": chunk_desc,
                            "embedding_model": emb_short,
                            "vector_store": "faiss",
                            "llm_model": llm,
                            "question": q["question"],
                            "correct_answer": q["correct_answer"],
                            "correct_answer_word_count": len(q["correct_answer"].split()),
                            "total_chunks_in_index": total_chunks,
                            "ingest_time_sec": ingest_time,
                            "query_time_sec": query_time,
                            "system_answer": answer,
                            "system_answer_word_count": len(str(answer).split()),
                            "sources_page_numbers": sources,
                            "num_sources_returned": len(retrieved),
                            "auto_accuracy_estimate": acc,
                            "correct_manual": "",
                            "notes": ""
                        })
                        f.flush()
                        print(f"    [{acc}] {q['question'][:45]}... {query_time}s")
                        time.sleep(5)

    print(f"\nDone. Appended to {OUTPUT_CSV}")
    print("Open the CSV and fill in 'correct_manual': yes / no / partial")

if __name__ == "__main__":
    run()
