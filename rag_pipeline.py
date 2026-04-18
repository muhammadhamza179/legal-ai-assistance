import os
import json
import re
import tiktoken
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader

# =========================
# LOAD MODELS
# =========================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model="gpt-4.1-mini")
embedding_model = OpenAIEmbeddings()

# =========================
# SAFE JSON PARSER
# =========================
def parse_json_safe(text):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    text = text.replace("\n", " ").replace("'", '"')
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    try:
        return json.loads(text)
    except:
        return None

# =========================
# TOKEN CONTROL
# =========================
def count_tokens(text, model="gpt-4.1-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def build_context_with_budget(docs, max_tokens=3000):
    context, total = [], 0

    for d in docs:
        t = count_tokens(d.page_content)
        if total + t > max_tokens:
            break
        context.append(d.page_content)
        total += t

    return "\n\n".join(context)

# =========================
# STEP 36 — QUERY EXPANSION
# =========================
def expand_query(query):
    prompt = f"""
Rewrite into 4 search variations:
1. Formal legal
2. Simple
3. Keyword
4. Legal terminology

Return Python list only.

Query: {query}
"""
    res = llm.invoke(prompt).content
    parsed = parse_json_safe(res)
    return parsed if isinstance(parsed, list) else [query]

# =========================
# STEP 37 — QUERY WEIGHTING
# =========================
def weight_queries(queries):
    base = 0.6
    return [(q, base / (i + 1)) for i, q in enumerate(queries)]

# =========================
# STEP 38 — INTENT CLASSIFICATION
# =========================
def classify_intent(query):
    prompt = f"""
Classify query into one:

FACTUAL
LEGAL_INTERPRETATION
COMPLEX_REASONING
DOCUMENT_LOOKUP

Return only label.

Query: {query}
"""
    res = llm.invoke(prompt).content.strip().upper()

    valid = {
        "FACTUAL",
        "LEGAL_INTERPRETATION",
        "COMPLEX_REASONING",
        "DOCUMENT_LOOKUP"
    }

    return res if res in valid else "LEGAL_INTERPRETATION"

# =========================
# MULTI-SIGNAL RERANKING
# =========================
def multi_signal_ranking(query, docs, vector_results, top_k=6):
    semantic = {
        d.page_content: 1 - (i / len(vector_results))
        for i, d in enumerate(vector_results)
    }

    pairs = [(query, d.page_content) for d in docs]
    rerank_scores = reranker.predict(pairs)

    scored = []

    for doc, r in zip(docs, rerank_scores):
        text = doc.page_content.lower()
        words = query.lower().split()

        keyword = sum(w in text for w in words) / len(words)
        semantic_score = semantic.get(doc.page_content, 0)

        final = 0.5 * r + 0.3 * keyword + 0.2 * semantic_score

        scored.append((doc, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:top_k]]

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
                        metadata={"source": file, "page": i + 1}
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
# USER QUERY
# =========================
query = "CAN SOMEONE MAKE PRIVATE ARMY?"

# =========================
# QUERY REWRITE
# =========================
rewritten_query = llm.invoke(
    f"Rewrite for legal retrieval:\n{query}"
).content.strip()

# =========================
# STEP 38 — INTENT ROUTING
# =========================
intent = classify_intent(query)
print(f"\n🔎 INTENT: {intent}\n")

# =========================
# ROUTING ENGINE
# =========================

vector_results = []
keyword_results = []

# ---------- DOCUMENT LOOKUP ----------
if intent == "DOCUMENT_LOOKUP":
    vector_results = vector_store.similarity_search(query, k=3)
    keyword_results = []

# ---------- FACTUAL ----------
elif intent == "FACTUAL":
    vector_results = vector_store.similarity_search(query, k=5)

# ---------- LEGAL INTERPRETATION ----------
elif intent == "LEGAL_INTERPRETATION":

    expanded = expand_query(rewritten_query)
    weighted = weight_queries(expanded)

    score_map = {}

    for q, w in weighted:
        results = vector_store.similarity_search(q, k=5)

        for d in results:
            key = d.page_content[:120]
            if key not in score_map:
                score_map[key] = {"doc": d, "score": 0}
            score_map[key]["score"] += w

    sorted_docs = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    vector_results = [x["doc"] for x in sorted_docs]

# ---------- COMPLEX REASONING ----------
elif intent == "COMPLEX_REASONING":

    expanded = expand_query(rewritten_query)
    weighted = weight_queries(expanded)

    score_map = {}

    for q, w in weighted:
        results = vector_store.similarity_search(q, k=5)

        for d in results:
            key = d.page_content[:120]
            if key not in score_map:
                score_map[key] = {"doc": d, "score": 0}
            score_map[key]["score"] += w

    sorted_docs = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    vector_results = [x["doc"] for x in sorted_docs]

# =========================
# KEYWORD SEARCH
# =========================
def keyword_search(query, docs):
    words = query.lower().split()
    return [d for d in docs if any(w in d.page_content.lower() for w in words)][:5]

keyword_results = keyword_search(query, docs)

# =========================
# MERGE + DEDUP
# =========================
combined = vector_results + keyword_results

seen = set()
unique_docs = []

for d in combined:
    key = d.page_content[:100]
    if key not in seen:
        seen.add(key)
        unique_docs.append(d)

# =========================
# RERANK
# =========================
final_docs = multi_signal_ranking(query, unique_docs, vector_results)

# =========================
# CONTEXT BUILD
# =========================
context = build_context_with_budget(final_docs)

# =========================
# PROMPT
# =========================
prompt = f"""
You are a strict legal AI assistant.

RULES:
- Only use context
- No hallucination
- Each statement must cite source

CONTEXT:
{context}

QUESTION:
{query}
"""

# =========================
# LLM ANSWER
# =========================
response = llm.invoke(prompt)
print("\nFINAL RESPONSE:\n")
print(response.content)