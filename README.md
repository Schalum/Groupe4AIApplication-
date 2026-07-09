# Chat with a PDF — AI-based Applications SoSe 2026

A RAG-based chatbot that lets users upload environmental sustainability
reports and ask questions about the content using a Large Language Model.

## Team — Groupe 4
- Schalum Semenyo
- Zahidul Alam
- TTAmin
- Leuphana University Lüneburg
- Course: AI-based Applications (Prof. Dr. Ricardo Usbeck / Dr. Debayan Banerjee)

## Project Management
- Trello board: https://trello.com/b/wOgfu3P3/ai-based-application
- GitHub repository: https://github.com/Schalum/Groupe4AIApplication-

## Demo

![Chat with a PDF — demo screenshot](docs/screenshot.png)

*The app answering a question about an uploaded sustainability report, with the retrieved source passages and page numbers shown below the answer.*

## Tech Stack
| Component | Technology |
|---|---|
| Frontend / App | Python, Streamlit |
| RAG Pipeline | LangChain, FAISS, HuggingFace |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM | GWDG API (Qwen3-30B) |

## Features
- Upload a PDF sustainability report
- Ask questions about the document, with retrieved source passages and page numbers shown alongside each answer
- Document summarization
- Automatic JSON extraction (CO2, NOX, electric vehicles, risks, opportunities, strategy, actions, policies, targets)
- Downloadable structured JSON report
- 👍 / 👎 feedback on answers

## Setup
1. Clone the repo
2. Copy `.env.example` to `.env` and add your GWDG API key
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run streamlit_app.py`

## Architecture
- `rag/ingest.py` — PDF extraction (per page), chunking, embedding, FAISS index
- `rag/query.py` — question embedding, chunk retrieval, GWDG LLM call, document summarization, JSON report extraction
- `streamlit_app.py` — the web UI (upload, chat, document tools, JSON export)

## AI Tools Disclosure
Claude (Anthropic) — used for code assistance and project planning
