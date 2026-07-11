# Vela — Chat with a PDF
### AI-based Applications · SoSe 2026 · Leuphana University Lüneburg

A RAG-based chatbot that lets users upload environmental sustainability reports and ask questions about the content using a Large Language Model.

---

## Team — Groupe 4

| Name | Role |
|---|---|
| Zahidul Alam | RAG pipeline, PDF processing, Docker |
| Schalum Semenyo | GitHub, project management, documentation |
| TTAmin | Streamlit UI, JSON export, frontend |

**Supervised by:** Dr. Debayan Banerjee · Prof. Dr. Ricardo Usbeck
**Project board:** https://trello.com/b/wOgfu3P3/ai-based-application
**GitHub:** https://github.com/Schalum/Groupe4AIApplication-

---

## Features

- Upload a PDF sustainability report
- Ask questions — answers grounded in the document with source page numbers
- Source text highlighted for each answer
- Document summary with one click
- Structured JSON export (CO2, NOX, electric vehicles, risks, opportunities, strategy, actions, policies, targets)
- Downloadable JSON report
- Model selector — choose from 5 GWDG models
- Feedback logging — thumbs up/down saved to CSV
- Dark and light mode toggle

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF extraction | PyMuPDF | Simple, fast, page-aware text extraction |
| Chunking | LangChain RecursiveCharacterTextSplitter | Splits at natural boundaries, preserves context |
| Embedding model | all-MiniLM-L6-v2 (HuggingFace) | Free, runs locally on CPU, 384-dim vectors |
| Vector store | FAISS | Lightweight, no server needed, saves to disk |
| Language model | GWDG API — Qwen3-30B | University infrastructure, free for students |
| UI | Streamlit | Rapid Python UI with built-in chat interface |
| Packaging | Docker | One command to run on any machine |

---

## Architecture

User uploads PDF
↓
rag/ingest.py
extract_pages()     → reads text + page numbers with PyMuPDF
chunk_text()        → 500 chars, 50 overlap with LangChain
get_embedder()      → loads all-MiniLM-L6-v2
FAISS.from_texts()  → embeds chunks + saves to vectorstore/
User asks a question
↓
rag/query.py
retrieve_chunks()   → embeds question, FAISS similarity search (k=6)
ask_llm()           → builds prompt, calls GWDG API
query_rag()         → returns answer + source chunks with page numbers
streamlit_app.py      → UI layer, connects user to pipeline

---

## Benchmark Results

We tested the pipeline after the final presentation using an automated script across different settings on the Microsoft 2021 Environmental Sustainability Report. 10 questions with known correct answers were used.

**LLM model comparison:**

| Model | Accuracy | Avg query time |
|---|---|---|
| Qwen3-30B | 46% | 1.12s |
| Apertus-70B | 41% | 36.52s |
| Llama 3.1 8B | 28% | 0.69s |

**Chunk size comparison:**

| Chunk size | Overlap | Accuracy | Ingest time |
|---|---|---|---|
| 300 | 30 | 45% | 16.38s |
| 500 | 50 | 43% | 15.79s |
| 1000 | 50 | 41% | 21.95s |

**Conclusion:** Smaller chunks give more accurate answers but require more embeddings. Qwen3-30B gives the best accuracy. Llama 3.1 8B is the fastest option for interactive use. Full results in `tests/test_results.csv` on the `benchmarking` branch.

---

## Known Limitations

- All chunk texts are loaded into memory before passing to FAISS. For very large PDFs this could cause memory issues. Fix: use `FAISS.add_texts()` in batches.
- Chunking breaks on aggregate questions like "how many times is CO2 mentioned" — RAG retrieves top-k chunks only, cannot count across the whole document.

---

## Setup — Run locally

**Requirements:** Python 3.11+ and a GWDG API key

```bash
# 1. Clone the repository
git clone https://github.com/Schalum/Groupe4AIApplication-
cd Groupe4AIApplication-

# 2. Create your environment file
cp .env.example .env
# Open .env and add: GWDG_API_KEY=your_key_here

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

---



## How to get a GWDG API key

1. Go to https://chat-ai.academiccloud.de
2. Log in with your university account
3. Generate an API key
4. Add it to your `.env` file

---

## Project structure

Groupe4AIApplication-/
├── rag/
│   ├── ingest.py          # PDF processing pipeline
│   └── query.py           # RAG query pipeline
├── tests/
│   ├── benchmark.py       # Benchmark script
│   └── test_results.csv   # Benchmark results (benchmarking branch)
├── streamlit_app.py        # Web UI
├── docker-compose.yml      # Container settings
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md

---

## AI Tools Disclosure

Claude (Anthropic) was used for code assistance, architecture planning, and documentation throughout this project.
