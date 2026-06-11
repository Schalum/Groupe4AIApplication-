import streamlit as st

st.set_page_config(page_title="Chat with a PDF",page_icon="",layout="wide",initial_sidebar_state="expanded")

MODELS = {
    "llama-3.1-8b · fast":        "meta-llama-3.1-8b-instruct",
    "qwen3-30b · recommended":     "qwen3-30b-a3b-instruct-2507",
    "deepseek-r1-70b · reasoning": "deepseek-r1-distill-llama-70b",
    "apertus-70b · quality":       "apertus-70b-instruct-2509",
    "teuken-7b · german":          "teuken-7b-instruct-research",
}

def ingest_pdf(f): return f.name
def query_rag(q, model): return f"pipeline not connected · model: {model} · you asked: '{q}'"

if "messages" not in st.session_state: st.session_state.messages=[]
if "pdf_loaded" not in st.session_state: st.session_state.pdf_loaded=False
if "pdf_name" not in st.session_state: st.session_state.pdf_name=None

with st.sidebar:
    dark_mode = st.toggle("dark mode", value=True)

if dark_mode:
    BG="#0D0F14"; BG2="#0A0C10"; BORDER="#1E2A40"
    TEXT="#94A3B8"; TEXT2="#475569"; TEXT3="#334155"
    MAIN="#E2E8F0"; UC="#CBD5E1"; AC="#94A3B8"
    BADGE_BG="#04150E"; BADGE_BORDER="#0F4A30"; BADGE_COLOR="#1D9E75"
    EMPTY2="#1E2A40"; MODEL_COLOR="#94A3B8"
    INPUT_BG="#0A0C10"; SUB_COLOR="#475569"
else:
    BG="#FAFAFA"; BG2="#F1F5F9"; BORDER="#CBD5E1"
    TEXT="#334155"; TEXT2="#64748B"; TEXT3="#94A3B8"
    MAIN="#0F172A"; UC="#0F172A"; AC="#475569"
    BADGE_BG="#ECFDF5"; BADGE_BORDER="#6EE7B7"; BADGE_COLOR="#059669"
    EMPTY2="#94A3B8"; MODEL_COLOR="#0F172A"
    INPUT_BG="#FFFFFF"; SUB_COLOR="#94A3B8"

st.markdown(f"""
<style>
#MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}header{{visibility:hidden;}}
.block-container{{padding-top:0!important;}}
.stApp{{background-color:{BG}!important;font-family:Inter,sans-serif;}}
.stApp > header{{background-color:{BG}!important;}}
[data-testid="stSidebar"]{{background-color:{BG2}!important;border-right:0.5px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;}}
[data-testid="stSidebar"] label p{{color:{MAIN}!important;}}
[data-testid="stSidebar"] [data-baseweb="toggle"] div{{background-color:{BORDER}!important;}}
[data-testid="stSidebar"] [role="checkbox"]{{background-color:{BORDER}!important;border-color:{BORDER}!important;}}
[data-testid="stSidebar"] [data-baseweb="checkbox"] span{{color:{MAIN}!important;}}
[data-testid="stFileUploader"]{{background-color:{BG2}!important;border:0.5px dashed {BORDER}!important;border-radius:4px!important;padding:8px!important;}}
[data-testid="stFileUploader"] *{{color:{TEXT2}!important;background-color:transparent!important;}}
[data-testid="stFileUploader"] button{{background-color:{BG}!important;border:0.5px solid {BORDER}!important;color:{TEXT2}!important;}}
.stButton>button{{background-color:transparent!important;color:{TEXT2}!important;border:0.5px solid {BORDER}!important;border-radius:3px!important;padding:6px 12px!important;font-family:monospace!important;font-size:10px!important;width:100%!important;letter-spacing:0.5px!important;}}
.stButton>button:hover{{background-color:{BG2}!important;color:{TEXT}!important;border-color:{BORDER}!important;}}
[data-testid="stChatInput"]{{background-color:{INPUT_BG}!important;border:0.5px solid {BORDER}!important;border-radius:4px!important;}}
[data-testid="stChatInput"] > div{{background-color:{INPUT_BG}!important;}}
[data-testid="stChatInput"] > div > div{{background-color:{INPUT_BG}!important;}}
[data-testid="stChatInput"] textarea{{background-color:{INPUT_BG}!important;border:none!important;color:{MAIN}!important;font-size:13px!important;-webkit-text-fill-color:{MAIN}!important;}}
[data-testid="stChatInput"] textarea::placeholder{{color:{SUB_COLOR}!important;-webkit-text-fill-color:{SUB_COLOR}!important;}}
[data-testid="stChatInputSubmitButton"]{{background-color:transparent!important;color:{MAIN}!important;}}
[data-testid="stChatInputSubmitButton"] svg{{fill:{MAIN}!important;color:{MAIN}!important;}}
section[data-testid="stBottom"]{{background-color:{BG}!important;}}
section[data-testid="stBottom"] > div{{background-color:{BG}!important;}}
[data-testid="stBottomBlockContainer"]{{background-color:{BG}!important;}}
[data-testid="stChatMessage"]{{background-color:{BG}!important;border:0.5px solid {BORDER}!important;border-radius:4px!important;margin:4px 0!important;}}
div[data-testid="stSelectbox"] > div > div{{background-color:{BG2}!important;border:0.5px solid {BORDER}!important;border-radius:3px!important;font-family:monospace!important;font-size:11px!important;color:{MODEL_COLOR}!important;}}
div[data-testid="stSelectbox"] svg{{fill:{MAIN}!important;}}
</style>""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<div style="height:0.5px;background:{BORDER};margin:16px 0 12px"></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Document</div>',unsafe_allow_html=True)
    uploaded_file=st.file_uploader("",type=["pdf"],label_visibility="collapsed")
    if uploaded_file is not None:
        if not st.session_state.pdf_loaded or st.session_state.pdf_name!=uploaded_file.name:
            with st.spinner(""):
                st.session_state.pdf_loaded=True
                st.session_state.pdf_name=uploaded_file.name
                st.session_state.messages=[]
    if st.session_state.pdf_loaded:
        st.markdown(f'<div style="font-family:monospace;font-size:10px;color:{BADGE_COLOR};background:{BADGE_BG};border:0.5px solid {BADGE_BORDER};border-radius:3px;padding:3px 8px;display:inline-block;margin-top:4px">✓ {st.session_state.pdf_name}</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="font-family:monospace;font-size:10px;color:{EMPTY2}">no document loaded</span>',unsafe_allow_html=True)

    st.markdown(f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Model</div>',unsafe_allow_html=True)
    selected_label = st.selectbox("",list(MODELS.keys()),label_visibility="collapsed")
    selected_model = MODELS[selected_label]

    st.markdown(f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:monospace;font-size:9px;color:{TEXT3};letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Explore</div>',unsafe_allow_html=True)
    for q in ["CO2 emissions overview","NOX reduction targets","Electric vehicle count","Key risks identified","Strategy and actions"]:
        if st.button(q,key=q):
            if st.session_state.pdf_loaded:
                st.session_state.messages.append({"role":"user","content":q})
                answer=query_rag(q,selected_model)
                st.session_state.messages.append({"role":"assistant","content":answer})
                st.rerun()

    st.markdown(f'<div style="height:0.5px;background:{BORDER};margin:12px 0"></div>',unsafe_allow_html=True)
    if st.button("clear conversation"): st.session_state.messages=[]; st.rerun()

st.markdown(f'<div style="font-size:28px;font-weight:300;color:{MAIN};margin-bottom:4px;letter-spacing:-0.5px">Chat with a PDF.</div>',unsafe_allow_html=True)
if st.session_state.pdf_loaded:
    st.markdown(f'<div style="font-family:monospace;font-size:10px;color:{SUB_COLOR};margin-bottom:20px">{st.session_state.pdf_name} · {selected_label}</div>',unsafe_allow_html=True)
else:
    st.markdown(f'<div style="font-family:monospace;font-size:10px;color:{SUB_COLOR};margin-bottom:20px">{selected_label}</div>',unsafe_allow_html=True)
st.markdown(f'<div style="height:0.5px;background:{BORDER};margin-bottom:24px"></div>',unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<span style="font-size:13px;color:{UC if msg["role"]=="user" else AC}">{msg["content"]}</span>',unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f'''<div style="text-align:center;padding:80px 20px">
        <div style="font-size:13px;color:{SUB_COLOR};margin-bottom:6px;font-weight:500">{"Document loaded. Ask a question." if st.session_state.pdf_loaded else "Upload a document to begin."}</div>
    </div>''',unsafe_allow_html=True)

if prompt:=st.chat_input("ask a question about the document...",disabled=not st.session_state.pdf_loaded):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(f'<span style="font-size:13px;color:{UC}">{prompt}</span>',unsafe_allow_html=True)
    with st.chat_message("assistant"):
        answer=query_rag(prompt,selected_model)
        st.markdown(f'<span style="font-size:13px;color:{AC}">{answer}</span>',unsafe_allow_html=True)
    st.session_state.messages.append({"role":"assistant","content":answer})
 