"""
03_medrag_minimal.py
====================
Minimal Medical RAG pipeline: index PubMed-style abstracts, retrieve top-k,
ask an LLM to answer with citations.
Corresponds to: Ch16 (Medical RAG & Agentic AI).

Requirements
------------
- langchain langchain-community
- sentence-transformers
- faiss-cpu (or faiss-gpu)
- An LLM endpoint: ollama / OpenAI / Anthropic / vLLM.

For demo we use a local ollama model (e.g. `ollama run llama3.1`).
"""
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


SOURCE_FILE = "pubmed_chunks.txt"   # one abstract per blank-line-separated block
INDEX_DIR = "faiss_pubmed"
EMBED_MODEL = "BAAI/bge-m3"
QUERY = "What's the first-line therapy for HER2+ metastatic breast cancer in 2024?"


def build_index(source: str = SOURCE_FILE) -> FAISS:
    docs = TextLoader(source).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, emb)
    db.save_local(INDEX_DIR)
    return db


def load_or_build_index() -> FAISS:
    if Path(INDEX_DIR).exists():
        emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
    return build_index()


def make_prompt(question: str, hits) -> str:
    ctx = "\n\n".join(f"[{i+1}] {h.page_content}" for i, h in enumerate(hits))
    return (
        "You are a clinical research assistant. Answer the question using ONLY the "
        "provided snippets. Cite each claim as [n]. If snippets are insufficient, "
        f"say so.\n\nSNIPPETS:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"
    )


def main() -> None:
    db = load_or_build_index()
    hits = db.similarity_search(QUERY, k=5)
    print(f"\nTop-{len(hits)} hits:\n" + "-" * 60)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.page_content[:200]}...")
    prompt = make_prompt(QUERY, hits)
    print("\n--- Prompt for LLM ---\n")
    print(prompt[:1500])
    print("\n...send `prompt` to your LLM (Ollama / OpenAI / Anthropic / vLLM) "
          "and inspect the answer.")


if __name__ == "__main__":
    main()
