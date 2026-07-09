# Vela — Chat with a PDF
### AI-based Applications · SoSe 2026 · Leuphana University Lüneburg

A RAG-based document intelligence app. Upload a sustainability report, ask questions, get grounded answers with source citations and page numbers.

---

## Team — Groupe 4

| Name | Role |
|---|---|
| Zahidul Alam | RAG pipeline, PDF processing, Docker |
| Schalum Semenyo | GitHub, project management, documentation |
| TTAmin | Streamlit UI, JSON export, frontend |

**Supervised by:** Dr. Debayan Banerjee · Prof. Dr. Ricardo Usbeck

**Project board:** https://trello.com/b/wOgfu3P3/ai-based-application

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

```
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
```

---

## Setup — Run locally

### Requirements
- Python 3.11+
- A GWDG API key (obtain at chat-ai.academiccloud.de)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Schalum/Groupe4AIApplication-
cd Groupe4AIApplication-
```

**2. Create your environment file**
```bash
cp .env.example .env
```

Open `.env` and add your GWDG API key:
```
GWDG_API_KEY=your_key_here
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

---

## Setup — Run with Docker

### Requirements
- Docker Desktop installed

### Steps

**1. Clone and set up .env as above**

**2. Build and run**
```bash
docker-compose up --build
```

The app opens at `http://localhost:8501`

**3. Stop**
```bash
docker-compose down
```

---

## How to get a GWDG API key

1. Go to https://chat-ai.academiccloud.de
2. Log in with your university account
3. Generate an API key
4. Add it to your `.env` file

---

## Project structure

```
Groupe4AIApplication-/
├── rag/
│   ├── ingest.py          # PDF processing pipeline
│   └── query.py           # RAG query pipeline
├── streamlit_app.py        # Web UI
├── Dockerfile              # Container definition
├── docker-compose.yml      # Container settings
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md
```

---

## AI Tools Disclosure

Claude (Anthropic) was used for code assistance, architecture planning, and documentation throughout this project.

---

## Questions — for the Q&A session

**Which part of the code does the chunking?**
`rag/ingest.py` — the `chunk_text()` function using `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`

**Which part calls the GWDG API?**
`rag/query.py` — the `ask_llm()` function using `OpenAI(base_url="https://chat-ai.academiccloud.de/v1")`

**Which part compares the embeddings?**
`rag/query.py` — the `retrieve_chunks()` function using `vectorstore.similarity_search(question, k=6)`
