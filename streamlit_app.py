import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from rag.ingest import ingest_pdf
from rag.query import extract_json_report, query_rag, summarize_document


st.set_page_config(
    page_title="Vela",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = {
    "fast" : "qwen3-30b-a3b-instruct-2507"
}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

FEEDBACK_PATH = Path("feedback.csv")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None

if "json_content" not in st.session_state:
    st.session_state.json_content = None

def safe_html(value):
    """Escape text before inserting it into HTML."""
    return html.escape(str(value)).replace("\n", "<br>")

def normalise_rag_result(result):
    """Always return answer text and source snippets."""
    if isinstance(result, dict):
        return (
            str(result.get("answer", "No answer was returned.")),
            result.get("sources", []) or [],
        )
    return str(result), []

def run_query_with_sources(question, model):
    result = query_rag(
        question,
        model=model,
        return_sources=True,
    )
    return normalise_rag_result(result)

def run_document_summary(model):
    result = summarize_document(model=model)
    return normalise_rag_result(result)

def add_assistant_message(answer, sources, model, message_type="answer"):
    st.session_state.messages.append(
        {
            "id": uuid4().hex,
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "model": model,
            "type": message_type,
            "feedback": None,
        }
    )

def get_question_before(message_index):
    """Find the user question paired with a given assistant answer."""
    for message in reversed(st.session_state.messages[:message_index]):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""

def save_feedback(message_index, rating):
    """Persist a helpful / not helpful rating to feedback.csv."""
    message = st.session_state.messages[message_index]

    if message.get("feedback") is not None:
        return

    file_exists = FEEDBACK_PATH.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_name": st.session_state.pdf_name or "",
        "question": get_question_before(message_index),
        "answer": message.get("content", ""),
        "model": message.get("model", ""),
        "rating": rating,
    }

    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    st.session_state.messages[message_index]["feedback"] = rating

def show_retrieved_sources(sources):
    """Show yellow retrieved excerpts with their PDF page number."""
    unique_sources = set()
    snippets = []

    for source in sources or []:
        if isinstance(source, dict):
            source_text = str(source.get("text", "")).strip()
            page = source.get("page", "?")
        else:
            source_text = str(source).strip()
            page = "?"

        source_text = " ".join(source_text.split())
        source_key = (page, source_text)

        if not source_text or source_key in unique_sources:
            continue

        unique_sources.add(source_key)
        snippets.append({"page": page, "text": source_text})

    if not snippets:
        return

    with st.expander("show retrieved text"):
        for source in snippets[:3]:
            snippet = source["text"][:650]
            if len(source["text"]) > 650:
                snippet += "..."

            st.markdown(
                f"""
                <div class="retrieved-snippet">
                    <div class="source-page">
                        Page {safe_html(source["page"])}
                    </div>
                    <mark>{safe_html(snippet)}</mark>
                </div>
                """,
                unsafe_allow_html=True,
            )

def show_feedback_controls(message_index, message):
    """Render one feedback control for each assistant message."""
    rating = message.get("feedback")

    if rating == "positive":
        st.caption("Feedback saved: 👍 Helpful")
        return

    if rating == "negative":
        st.caption("Feedback saved: 👎 Not helpful")
        return

    st.caption("Was this answer helpful?")
    positive_column, negative_column = st.columns(2)

    with positive_column:
        if st.button("👍 Helpful", key=f"feedback_positive_{message['id']}"):
            save_feedback(message_index, "positive")
            st.rerun()

    with negative_column:
        if st.button("👎 Not helpful", key=f"feedback_negative_{message['id']}"):
            save_feedback(message_index, "negative")
            st.rerun()

with st.sidebar:
    dark_mode = st.toggle("dark mode", value=True)

if dark_mode:
    BG = "#0D0F14"
    BG2 = "#0A0C10"
    BORDER = "#1E2A40"
    TEXT = "#94A3B8"
    TEXT2 = "#475569"
    TEXT3 = "#334155"
    MAIN = "#E2E8F0"
    USER_COLOR = "#CBD5E1"
    ASSISTANT_COLOR = "#94A3B8"
    BADGE_BG = "#04150E"
    BADGE_BORDER = "#0F4A30"
    BADGE_COLOR = "#1D9E75"
    EMPTY2 = "#1E2A40"
    MODEL_COLOR = "#94A3B8"
    INPUT_BG = "#0A0C10"
    SUB_COLOR = "#475569"
else:
    BG = "#F8FAFC"
    BG2 = "#F1F5F9"
    BORDER = "#CBD5E1"
    TEXT = "#334155"
    TEXT2 = "#475569"
    TEXT3 = "#64748B"
    MAIN = "#0F172A"
    USER_COLOR = "#0F172A"
    ASSISTANT_COLOR = "#334155"
    BADGE_BG = "#ECFDF5"
    BADGE_BORDER = "#6EE7B7"
    BADGE_COLOR = "#047857"
    EMPTY2 = "#64748B"
    MODEL_COLOR = "#0F172A"
    INPUT_BG = "#FFFFFF"
    SUB_COLOR = "#64748B"

st.markdown(
    f"""
<style>
#MainMenu{{visibility:hidden;}}
footer{{visibility:hidden;}}

.block-container{{padding-top:0!important;}}

.stApp{{
    background-color:{BG}!important;
    font-family:Inter,sans-serif;
}}

.stApp > header{{
    background-color:{BG}!important;
}}

[data-testid="stSidebar"]{{
    background-color:{BG2}!important;
    border-right:0.5px solid {BORDER};
}}

[data-testid="stSidebar"] *{{
    color:{TEXT}!important;
}}

[data-testid="stSidebar"] label p{{
    color:{MAIN}!important;
}}

[data-testid="stSidebar"] [data-baseweb="toggle"] div{{
    background-color:{BORDER}!important;
}}

[data-testid="stFileUploader"]{{
    background-color:{BG2}!important;
    border:0.5px dashed {BORDER}!important;
    border-radius:4px!important;
    padding:8px!important;
}}

[data-testid="stFileUploader"] *{{
    color:{TEXT2}!important;
    background-color:transparent!important;
}}

[data-testid="stFileUploader"] button{{
    background-color:{BG}!important;
    border:0.5px solid {BORDER}!important;
    color:{TEXT2}!important;
}}

.stButton > button,
.stDownloadButton > button{{
    background-color:transparent!important;
    color:{TEXT2}!important;
    border:0.5px solid {BORDER}!important;
    border-radius:3px!important;
    padding:6px 12px!important;
    font-family:monospace!important;
    font-size:10px!important;
    width:100%!important;
    letter-spacing:0.5px!important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover{{
    background-color:{BG2}!important;
    color:{TEXT}!important;
    border-color:{BORDER}!important;
}}

[data-testid="stChatInput"]{{
    background-color:{INPUT_BG}!important;
    border:0.5px solid {BORDER}!important;
    border-radius:4px!important;
}}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div{{
    background-color:{INPUT_BG}!important;
}}

[data-testid="stChatInput"] textarea{{
    background-color:{INPUT_BG}!important;
    border:none!important;
    color:{MAIN}!important;
    font-size:13px!important;
    -webkit-text-fill-color:{MAIN}!important;
}}

[data-testid="stChatMessage"]{{
    background-color:{BG}!important;
    border:0.5px solid {BORDER}!important;
    border-radius:4px!important;
    margin:4px 0!important;
}}

div[data-testid="stSelectbox"] > div > div{{
    background-color:{BG2}!important;
    border:0.5px solid {BORDER}!important;
    border-radius:3px!important;
    font-family:monospace!important;
    font-size:11px!important;
    color:{MODEL_COLOR}!important;
}}

.retrieved-snippet{{
    margin:8px 0;
    padding:10px 12px;
    border:0.5px solid {BORDER};
    border-radius:4px;
    background:{BG2};
    line-height:1.6;
    font-size:12px;
}}

.source-page{{
    display:inline-block;
    margin-bottom:7px;
    padding:3px 7px;
    border:0.5px solid {BORDER};
    border-radius:3px;
    background:{BG};
    color:{MAIN};
    font-family:monospace;
    font-size:10px;
    letter-spacing:0.4px;
}}

.retrieved-snippet mark{{
    display:block;
    background:#FDE68A!important;
    color:#111827!important;
    border-radius:3px;
    padding:4px 5px;
}}

[data-testid="stExpander"] details{{
    border:0.5px solid {BORDER};
    border-radius:4px;
    background:{BG2};
}}
section[data-testid="stBottom"]{{
    background-color:{BG}!important;
}}

section[data-testid="stBottom"] > div{{
    background-color:{BG}!important;
}}

[data-testid="stBottomBlockContainer"]{{
    background-color:{BG}!important;
    border-top:0.5px solid {BORDER}!important;
    padding-top:18px!important;
    padding-bottom:18px!important;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span{{
    color:{TEXT}!important;
    font-weight:500!important;
}}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *{{
    color:{TEXT2}!important;
}}

div[data-testid="stSelectbox"] input,
div[data-testid="stSelectbox"] span{{
    color:{MAIN}!important;
    -webkit-text-fill-color:{MAIN}!important;
}}

[data-testid="stChatInput"] textarea::placeholder{{
    color:{TEXT2}!important;
    -webkit-text-fill-color:{TEXT2}!important;
    opacity:1!important;
}}
header[data-testid="stHeader"]{{
    background-color:{BG}!important;
    border-bottom:none!important;
}}

[data-testid="stDecoration"]{{
    display:none!important;
}}

[data-testid="stStatusWidget"]{{
    display:none!important;
}}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        f'<div style="height:0.5px;background:{BORDER};margin:16px 0 12px"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Document</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        current_pdf_id = hashlib.sha256(uploaded_bytes).hexdigest()

        if st.session_state.pdf_id != current_pdf_id:
            safe_filename = Path(uploaded_file.name).name
            pdf_path = UPLOAD_DIR / f"{current_pdf_id[:12]}_{safe_filename}"

            st.session_state.pdf_loaded = False
            st.session_state.messages = []
            st.session_state.json_content = None

            try:
                with pdf_path.open("wb") as file:
                    file.write(uploaded_bytes)

                with st.spinner("Processing PDF..."):
                    ingest_pdf(str(pdf_path))

                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = safe_filename
                st.session_state.pdf_id = current_pdf_id
            except Exception as error:
                st.error(f"Could not process the PDF: {error}")

    if st.session_state.pdf_loaded:
        st.markdown(
            f"""
            <div style="font-family:monospace;font-size:10px;color:{BADGE_COLOR};
            background:{BADGE_BG};border:0.5px solid {BADGE_BORDER};
            border-radius:3px;padding:3px 8px;display:inline-block;margin-top:4px">
            ✓ {safe_html(st.session_state.pdf_name)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="font-family:monospace;font-size:10px;color:{EMPTY2}">no document loaded</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Model</div>',
        unsafe_allow_html=True,
    )

    selected_label = st.selectbox("", list(MODELS.keys()), label_visibility="collapsed")
    selected_model = MODELS[selected_label]

    st.markdown(
        f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Document tools</div>',
        unsafe_allow_html=True,
    )

    if st.button("summarize document", disabled=not st.session_state.pdf_loaded):
        try:
            with st.spinner("Creating document summary..."):
                answer, sources = run_document_summary(selected_model)

            st.session_state.messages.append(
                {"role": "user", "content": "Summarize this document."}
            )
            add_assistant_message(
                answer=answer,
                sources=sources,
                model=selected_model,
                message_type="summary",
            )
            st.rerun()
        except Exception as error:
            st.error(f"Could not create the summary: {error}")

    st.markdown(
        f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">JSON Export</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.pdf_loaded:
        if st.button("generate json report"):
            try:
                with st.spinner("Generating JSON report..."):
                    report = extract_json_report(model=selected_model)

                st.session_state.json_content = json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                )
                st.success("JSON report ready.")
            except Exception as error:
                st.error(f"Could not generate JSON: {error}")

        if st.session_state.json_content:
            st.download_button(
                label="download json",
                data=st.session_state.json_content,
                file_name="report.json",
                mime="application/json",
            )
    else:
        st.markdown(
            f'<span style="font-family:monospace;font-size:10px;color:{EMPTY2}">upload document first</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Explore</div>',
        unsafe_allow_html=True,
    )

    example_questions = [
        "CO2 emissions overview",
        "NOX reduction targets",
        "Electric vehicle count",
        "Key risks identified",
        "Strategy and actions",
    ]

    for index, question in enumerate(example_questions):
        if st.button(question, key=f"example_question_{index}"):
            if st.session_state.pdf_loaded:
                try:
                    with st.spinner("Searching document..."):
                        answer, sources = run_query_with_sources(question, selected_model)

                    st.session_state.messages.append(
                        {"role": "user", "content": question}
                    )
                    add_assistant_message(
                        answer=answer,
                        sources=sources,
                        model=selected_model,
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not answer the question: {error}")

    st.markdown(
        f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',
        unsafe_allow_html=True,
    )

    if st.button("clear conversation"):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    f"""
    <div style="font-size:28px;font-weight:300;color:{MAIN};
    margin-bottom:4px;letter-spacing:-0.5px">
    Vela.
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.pdf_loaded:
    subtitle = (
        f"Navigate your documents with AI · "
        f"{st.session_state.pdf_name} · {selected_label}"
    )
else:
    subtitle = f"Navigate your documents with AI · {selected_label}"

st.markdown(
    f"""
    <div style="font-family:monospace;font-size:10px;color:{SUB_COLOR};
    margin-bottom:20px">
    {safe_html(subtitle)}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div style="height:0.5px;background:{BORDER};margin-bottom:24px"></div>',
    unsafe_allow_html=True,
)

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        color = USER_COLOR if message["role"] == "user" else ASSISTANT_COLOR

        if message.get("type") == "summary":
            st.caption("Document summary")

        st.markdown(
            f'<div style="font-size:13px;color:{color};line-height:1.6">'
            f'{safe_html(message["content"])}</div>',
            unsafe_allow_html=True,
        )

        if message["role"] == "assistant":
            show_retrieved_sources(message.get("sources", []))
            show_feedback_controls(message_index, message)

if not st.session_state.messages:
    empty_text = (
        "Document loaded. Ask a question."
        if st.session_state.pdf_loaded
        else "Upload a document to begin."
    )

    st.markdown(
        f"""
        <div style="text-align:center;padding:80px 20px">
            <div style="font-size:13px;color:{SUB_COLOR};
            margin-bottom:6px;font-weight:500">
                {empty_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

prompt = st.chat_input(
    "Ask a question about the document...",
    disabled=not st.session_state.pdf_loaded,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching document..."):
            answer, sources = run_query_with_sources(prompt, selected_model)

        add_assistant_message(
            answer=answer,
            sources=sources,
            model=selected_model,
        )
    except Exception as error:
        add_assistant_message(
            answer=f"Something went wrong while processing the question: {error}",
            sources=[],
            model=selected_model,
        )

    st.rerun()
