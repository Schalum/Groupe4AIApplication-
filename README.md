# Chat with a PDF — AI-based Applications SoSe 2026

A RAG-based chatbot that lets users upload environmental sustainability 
reports and ask questions about the content using a Large Language Model.

## Team — Groupe 4
- Leuphana University Lüneburg
- Course: AI-based Applications (Prof. Dr. Ricardo Usbeck / Dr. Debayan Banerjee)

## Tech Stack
| Component | Technology |
|---|---|
| Backend | Python, Flask |
| RAG Pipeline | LangChain, FAISS, HuggingFace |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM | GWDG API (Llama 3.1 8B) |
| Frontend | Streamlit |

## Features
- Upload a PDF sustainability report
- Ask questions about the document
- Automatic JSON extraction (CO2, NOX, emissions, targets)
- Downloadable structured report

## Setup
1. Clone the repo
2. Copy `.env.example` to `.env` and add your GWDG API key
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run streamlit_app.py`

## Architecture
- `rag/ingest.py` — PDF extraction, chunking, embedding, FAISS index
- `rag/query.py` — question embedding, chunk retrieval, GWDG LLM call
- `app.py` — Flask routes (/upload, /chat, /download-json)
- `templates/index.html` — Frontend UI

## AI Tools Disclosure
Claude (Anthropic) — used for code assistance and project planning
