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
Rewrite into 4 versions:
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
# STEP 38 — INTENT CLASSIFIER
# =========================
def classify_intent(query):
    prompt = f"""
Classify into:

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
# STEP 39 — CONTEXT OPTIMIZATION
# =========================
def optimize_context(docs):
    seen = set()
    optimized = []

    for d in docs:
        text = " ".join(d.page_content.split())
        sentences = text.split(". ")
        cleaned = []

        for s in sentences:
            s_clean = s.strip().lower()
            if s_clean in seen:
                continue
            seen.add(s_clean)
            cleaned.append(s)

        if cleaned:
            optimized.append(
                Document(
                    page_content=". ".join(cleaned),
                    metadata=d.metadata
                )
            )
    return optimized

# =========================
# STEP 40 — RETRIEVAL ANALYZER
# =========================
def analyze_retrieval(query, vector_results, bm25_results, expanded_queries):
    report = {
        "vector_results": len(vector_results),
        "bm25_results": len(bm25_results),
        "expanded_queries": len(expanded_queries)
    }

    vset = set([d.page_content[:100] for d in vector_results])
    bset = set([d.page_content[:100] for d in bm25_results])

    report["overlap"] = len(vset.intersection(bset))

    combined = vector_results + bm25_results
    unique = set([d.page_content[:120] for d in combined])

    report["diversity"] = len(unique) / len(combined) if combined else 0

    warnings = []
    if not vector_results:
        warnings.append("Vector retrieval failed")
    if not bm25_results:
        warnings.append("BM25 weak")
    if report["diversity"] < 0.5:
        warnings.append("Low diversity")
    if report["overlap"] == 0:
        warnings.append("No agreement")

    report["warnings"] = warnings
    return report

# =========================
# STEP 42 — SIMPLE HYBRID FUSION
# =========================
def hybrid_fusion_simple(vector_results, bm25_results, alpha=0.6, top_k=10):
    scores = {}

    for i, doc in enumerate(vector_results):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + alpha * (1 - i / len(vector_results))

    for i, doc in enumerate(bm25_results):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + (1 - alpha) * (1 - i / len(bm25_results))

    doc_map = {}
    for d in vector_results + bm25_results:
        doc_map[d.page_content[:120]] = d

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ranked[:top_k]]

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
        words = query.lower().split()
        text = doc.page_content.lower()

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
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# =========================
# VECTOR STORE
# =========================
vector_store = FAISS.from_documents(docs, embedding_model)

# =========================
# STEP 41 — BM25 INDEX
# =========================
tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_search(query, docs, bm25, top_k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored[:top_k]]

# =========================
# USER QUERY
# =========================
query = "CAN SOMEONE MAKE PRIVATE ARMY?"

rewritten_query = llm.invoke(
    f"Rewrite for legal retrieval:\n{query}"
).content.strip()

intent = classify_intent(query)
print(f"\n🔎 INTENT: {intent}\n")

# =========================
# VECTOR RETRIEVAL
# =========================
expanded = []
vector_results = []

if intent in ["LEGAL_INTERPRETATION", "COMPLEX_REASONING"]:
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
else:
    vector_results = vector_store.similarity_search(query, k=5)

# =========================
# BM25 SEARCH
# =========================
bm25_results = bm25_search(query, docs, bm25)

# =========================
# STEP 40 — ANALYSIS
# =========================
analysis = analyze_retrieval(query, vector_results, bm25_results, expanded if expanded else [query])
print("\n🔍 RETRIEVAL ANALYSIS:")
print(json.dumps(analysis, indent=2))

# =========================
# STEP 42 — HYBRID FUSION
# =========================
unique_docs = hybrid_fusion_simple(vector_results, bm25_results)

# =========================
# CONTEXT OPTIMIZATION
# =========================
optimized_docs = optimize_context(unique_docs)

final_docs = [
    Document(page_content=" ".join(d.page_content.split())[:500], metadata=d.metadata)
    for d in optimized_docs
]

# =========================
# RERANK
# =========================
final_docs = multi_signal_ranking(query, final_docs, vector_results)

# =========================
# CONTEXT
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
- Must cite sources

CONTEXT:
{context}

QUESTION:
{query}
"""

response = llm.invoke(prompt)

print("\nFINAL ANSWER:\n")
print(response.content)