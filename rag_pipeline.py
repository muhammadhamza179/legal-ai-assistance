import os
import json
import re
import tiktoken
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader

# =========================
# MODELS
# =========================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model="gpt-4.1-mini")
embedding_model = OpenAIEmbeddings()

# =========================
# TOKEN CONTROL
# =========================
def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-4.1-mini")
    return len(enc.encode(text))

def build_context(docs, max_tokens=3000):
    context, total = [], 0

    for doc in docs:
        t = count_tokens(doc.page_content)
        if total + t > max_tokens:
            break
        context.append(doc.page_content)
        total += t

    return "\n\n".join(context)

# =========================
# QUERY REWRITE
# =========================
def rewrite_query(query):
    return llm.invoke(f"Rewrite for legal retrieval:\n{query}").content.strip()

# =========================
# INTENT CLASSIFIER (Step 38)
# =========================
def classify_intent(query):
    prompt = f"""
Classify query into:
FACTUAL, LEGAL_INTERPRETATION, COMPLEX_REASONING, DOCUMENT_LOOKUP

Query: {query}

Return only one label.
"""
    result = llm.invoke(prompt).content.strip().upper()

    valid = {
        "FACTUAL",
        "LEGAL_INTERPRETATION",
        "COMPLEX_REASONING",
        "DOCUMENT_LOOKUP"
    }

    return result if result in valid else "LEGAL_INTERPRETATION"

# =========================
# BM25 SEARCH (Step 41)
# =========================
def bm25_search(query, docs, bm25, k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]

# =========================
# HYBRID RETRIEVAL (Step 42)
# =========================
def hybrid_retrieval(vector_results, bm25_results, alpha=0.6, top_k=10):
    scores = {}
    doc_map = {}

    def norm(i, size):
        return 1 - (i / max(size, 1))

    for i, doc in enumerate(vector_results):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + alpha * norm(i, len(vector_results))
        doc_map[key] = doc

    for i, doc in enumerate(bm25_results):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + (1 - alpha) * norm(i, len(bm25_results))
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ranked[:top_k]]

# =========================
# QUERY ROUTING (Step 43)
# =========================
def route_query(intent, query, vector_store, docs, bm25):

    expanded_query = None

    if intent == "FACTUAL":
        vector_results = vector_store.similarity_search(query, k=5)
        bm25_results = []

    elif intent == "DOCUMENT_LOOKUP":
        vector_results = vector_store.similarity_search(query, k=5)
        bm25_results = bm25_search(query, docs, bm25)

    else:
        expanded_query = rewrite_query(query)
        vector_results = vector_store.similarity_search(expanded_query, k=5)
        bm25_results = bm25_search(query, docs, bm25)

    return vector_results, bm25_results, expanded_query

# =========================
# STEP 44 — HALLUCINATION DETECTION
# =========================
def detect_hallucination(answer, context_docs, threshold=0.3):

    if not answer or not context_docs:
        return True, 0.0

    context_text = " ".join([doc.page_content for doc in context_docs])

    answer_words = set(answer.lower().split())
    context_words = set(context_text.lower().split())

    overlap = answer_words.intersection(context_words)

    score = len(overlap) / max(len(answer_words), 1)

    return score < threshold, score

# =========================
# LOAD DOCUMENTS
# =========================
folder = "data"
documents = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        reader = PdfReader(f"{folder}/{file}")

        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"file": file, "page": i + 1}
                    )
                )

# =========================
# CHUNKING
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# =========================
# VECTOR STORE
# =========================
vector_store = FAISS.from_documents(docs, embedding_model)

# =========================
# BM25 INDEX
# =========================
tokenized_docs = [d.page_content.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized_docs)

# =========================
# USER QUERY
# =========================
query = "CAN SOMEONE MAKE PRIVATE ARMY?"

# =========================
# PIPELINE START
# =========================
rewritten_query = rewrite_query(query)
intent = classify_intent(rewritten_query)

print("\n🔎 INTENT:", intent)

# =========================
# ROUTING
# =========================
vector_results, bm25_results, expanded_query = route_query(
    intent,
    rewritten_query,
    vector_store,
    docs,
    bm25
)

print("\n🧠 EXPANDED QUERY:", expanded_query)

# =========================
# HYBRID RETRIEVAL
# =========================
final_docs = hybrid_retrieval(vector_results, bm25_results)

# =========================
# CONTEXT BUILDING
# =========================
context = build_context(final_docs)

# =========================
# PROMPT
# =========================
prompt = f"""
You are a strict legal AI assistant.

RULES:
- Use only given context
- Do not hallucinate
- Be precise

CONTEXT:
{context}

QUESTION:
{query}
"""

# =========================
# LLM RESPONSE
# =========================
response = llm.invoke(prompt)
answer = response.content

# =========================
# STEP 44 — HALLUCINATION CHECK
# =========================
is_hallucinated, score = detect_hallucination(answer, final_docs)

print("\n🔍 HALLUCINATION SCORE:", round(score, 3))

# =========================
# FINAL OUTPUT CONTROL
# =========================
if is_hallucinated:
    print("\n⚠️ POSSIBLE HALLUCINATION DETECTED\n")
    print("SAFE RESPONSE:\n")
    print("The available documents do not contain enough verified information to answer this reliably.")
else:
    print("\nFINAL ANSWER:\n")
    print(answer)