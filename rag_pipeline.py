import os
import json
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

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
llm = ChatOpenAI(model="gpt-4.1-mini")
embedding_model = OpenAIEmbeddings()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================
# TOKEN CONTROL
# =========================
def count_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-4.1-mini")
    return len(enc.encode(text))

def build_context(docs, max_tokens=3000):
    context, total = [], 0
    for d in docs:
        t = count_tokens(d.page_content)
        if total + t > max_tokens:
            break
        context.append(d.page_content)
        total += t
    return "\n\n".join(context)

# =========================
# QUERY REWRITE
# =========================
def rewrite_query(q):
    return llm.invoke(f"Rewrite for legal retrieval:\n{q}").content.strip()

# =========================
# INTENT CLASSIFIER
# =========================
def classify_intent(q):
    prompt = f"""
Classify query into:
FACTUAL, LEGAL_INTERPRETATION, COMPLEX_REASONING, DOCUMENT_LOOKUP
Query: {q}
Return only label.
"""
    res = llm.invoke(prompt).content.strip().upper()
    return res if res in {
        "FACTUAL",
        "LEGAL_INTERPRETATION",
        "COMPLEX_REASONING",
        "DOCUMENT_LOOKUP"
    } else "LEGAL_INTERPRETATION"

# =========================
# BM25 SEARCH
# =========================
def bm25_search(query, docs, bm25, k=5):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k]]

# =========================
# HYBRID RETRIEVAL
# =========================
def hybrid_retrieval(v, b, alpha=0.6, top_k=10):
    scores, doc_map = {}, {}

    def norm(i, n): return 1 - (i / max(n, 1))

    for i, d in enumerate(v):
        k = d.page_content[:120]
        scores[k] = scores.get(k, 0) + alpha * norm(i, len(v))
        doc_map[k] = d

    for i, d in enumerate(b):
        k = d.page_content[:120]
        scores[k] = scores.get(k, 0) + (1 - alpha) * norm(i, len(b))
        doc_map[k] = d

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, _ in ranked[:top_k]]

# =========================
# ROUTING (STEP 43)
# =========================
def route_query(intent, query, vector_store, docs, bm25):

    if intent == "FACTUAL":
        return vector_store.similarity_search(query, k=5), [], None

    if intent == "DOCUMENT_LOOKUP":
        return (
            vector_store.similarity_search(query, k=5),
            bm25_search(query, docs, bm25),
            None
        )

    expanded = rewrite_query(query)

    return (
        vector_store.similarity_search(expanded, k=5),
        bm25_search(query, docs, bm25),
        expanded
    )

# =========================
# VALIDATION (STEP 44)
# =========================
def keyword_check(answer, docs):
    ctx = " ".join([d.page_content for d in docs])
    a = set(answer.lower().split())
    c = set(ctx.lower().split())
    return len(a & c) / max(len(a), 1)

# =========================
# STEP 45 — SEMANTIC GROUNDING
# =========================
def semantic_check(answer, docs, emb):
    ctx = " ".join([d.page_content for d in docs])
    a_emb = emb.embed_query(answer)
    c_emb = emb.embed_query(ctx)
    return cosine_similarity([a_emb], [c_emb])[0][0]

# =========================
# STEP 46 — CONFIDENCE
# =========================
def compute_confidence(k, s, docs):
    retrieval = min(len(docs) / 10, 1)
    return round(0.3*k + 0.5*s + 0.2*retrieval, 3)

# =========================
# STEP 47 — STRUCTURED OUTPUT + CITATIONS
# =========================
def format_answer_with_citations(answer, docs):

    structured_prompt = f"""
Convert into structured legal output.

RULES:
- Use ONLY provided answer
- Attach citations from context
- Return JSON ONLY

FORMAT:
{{
  "answer": [
    {{
      "statement": "...",
      "source": {{
        "file": "...",
        "page": "..."
      }}
    }}
  ]
}}

ANSWER:
{answer}
"""

    structured = llm.invoke(structured_prompt).content
    return structured

# =========================
# LOAD DATA
# =========================
documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        reader = PdfReader(f"data/{file}")
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
# PREPROCESS
# =========================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

vector_store = FAISS.from_documents(docs, embedding_model)
bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])

# =========================
# QUERY
# =========================
query = "CAN SOMEONE MAKE PRIVATE ARMY?"

rewritten = rewrite_query(query)
intent = classify_intent(rewritten)

# =========================
# RETRIEVAL
# =========================
v_docs, b_docs, expanded = route_query(intent, rewritten, vector_store, docs, bm25)
final_docs = hybrid_retrieval(v_docs, b_docs)

context = build_context(final_docs)

# =========================
# GENERATION
# =========================
prompt = f"""
You are a strict legal AI assistant.

Use ONLY context.
No hallucination.

CONTEXT:
{context}

QUESTION:
{query}
"""

answer = llm.invoke(prompt).content

# =========================
# STEP 44 + 45 + 46
# =========================
kw = keyword_check(answer, final_docs)
sem = semantic_check(answer, final_docs, embedding_model)
confidence = compute_confidence(kw, sem, final_docs)

print("\n--- SCORES ---")
print("Keyword:", round(kw, 3))
print("Semantic:", round(sem, 3))
print("Confidence:", confidence)

# =========================
# DECISION LAYER
# =========================
if confidence < 0.5:
    print("\n⚠️ LOW CONFIDENCE")
    final_output = "Insufficient verified information in documents."

elif confidence < 0.75:
    print("\n⚠️ MEDIUM CONFIDENCE")
    final_output = answer

else:
    print("\n✅ HIGH CONFIDENCE")
    final_output = answer

# =========================
# STEP 47 — STRUCTURED OUTPUT
# =========================
structured_output = format_answer_with_citations(final_output, final_docs)

print("\n--- FINAL STRUCTURED OUTPUT ---\n")
print(structured_output)