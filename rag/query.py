import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from rag.ingest import get_embedder

load_dotenv()


def retrieve_chunks(question):
    # load saved vectors and find the 3 most relevant chunks
    embedder = get_embedder()
    vectorstore = FAISS.load_local(
        "vectorstore/",
        embedder,
        allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(question, k=3)

    chunks = []
    for result in results:
        chunks.append(result.page_content)
    return chunks


def ask_llm(question, chunks, model):
    # combine all chunks into one block of text
    context = ""
    for chunk in chunks:
        context = context + chunk + "\n\n"

    # build a better prompt
    prompt = "You are a helpful assistant answering questions about a sustainability report.\n"
    prompt = prompt + "Use the context below to answer the question as clearly and completely as possible.\n"
    prompt = prompt + "If the answer is not in the context say: Not found in document.\n\n"
    prompt = prompt + "Context:\n" + context
    prompt = prompt + "\nQuestion: " + question
    prompt = prompt + "\nAnswer:"

    # send to gwdg and get answer
    client = OpenAI(
        api_key=os.getenv("GWDG_API_KEY"),
        base_url="https://chat-ai.academiccloud.de/v1"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def query_rag(question, model="qwen3-30b-a3b-instruct-2507", return_sources=False):
    chunks = retrieve_chunks(question)
    answer = ask_llm(question, chunks, model)
    
    if return_sources:
        return answer, chunks
    return answer
def extract_json_report(output_path="report.json"):
    # list of fields to extract from the pdf
    fields = {
        "CO2_emissions": "What are the CO2 emissions?",
        "NOX_emissions": "What are the NOX emissions?",
        "electric_vehicles": "How many electric vehicles are mentioned?",
        "risks": "What are the key risks mentioned?",
        "opportunities": "What are the opportunities mentioned?",
        "strategy": "What is the sustainability strategy?",
        "actions": "What actions are being taken?",
        "policies": "What policies are mentioned?",
        "targets": "What are the targets or goals mentioned?"
    }

    # ask the pipeline for each field
    report = {}
    for field, question in fields.items():
        answer = query_rag(question)
        report[field] = answer

    # save to json file
    import json
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("report saved to " + output_path)
    return report

def summarize_document(model="qwen3-30b-a3b-instruct-2507"):
    embedder = get_embedder()
    vectorstore = FAISS.load_local(
        "vectorstore/",
        embedder,
        allow_dangerous_deserialization=True
    )
    
    results = vectorstore.similarity_search("summary overview introduction", k=5)
    
    chunks = []
    for result in results:
        chunks.append(result.page_content)
    
    context = ""
    for chunk in chunks:
        context = context + chunk + "\n\n"
    
    prompt = "Please provide a comprehensive summary of this document based on the context below.\n\n"
    prompt = prompt + "Context:\n" + context
    prompt = prompt + "\nSummary:"
    
    client = OpenAI(
        api_key=os.getenv("GWDG_API_KEY"),
        base_url="https://chat-ai.academiccloud.de/v1"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content