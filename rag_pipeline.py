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
# TOKEN FUNCTIONS
# =========================
def count_tokens(text, model="gpt-4.1-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def build_context_with_budget(docs, max_tokens=3000):
    context_chunks = []
    total_tokens = 0

    for doc in docs:
        tokens = count_tokens(doc.page_content)

        if total_tokens + tokens > max_tokens:
            break

        context_chunks.append(doc.page_content)
        total_tokens += tokens

    return "\n\n".join(context_chunks)

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
# STEP 36 — QUERY EXPANSION
# =========================
def expand_query(query: str):
    prompt = f"""
You are a legal AI search optimizer.

Rewrite this query into 4 versions:
1. Formal legal version
2. Simple version
3. Keyword version
4. Related legal terminology version

Return ONLY a Python list.

Query: {query}
"""

    response = llm.invoke(prompt).content
    parsed = parse_json_safe(response)

    if not parsed:
        return [query]

    return parsed if isinstance(parsed, list) else [query]

# =========================
# STEP 37 — QUERY WEIGHTING
# =========================
def weight_queries(queries):
    base_weight = 0.6
    return [(q, base_weight / (i + 1)) for i, q in enumerate(queries)]

# =========================
# MULTI-SIGNAL RANKING
# =========================
def multi_signal_ranking(query, docs, vector_results, top_k=6):
    semantic_scores = {
        doc.page_content: 1 - (i / len(vector_results))
        for i, doc in enumerate(vector_results)
    }

    query_words = query.lower().split()
    pairs = [(query, doc.page_content) for doc in docs]
    rerank_scores = reranker.predict(pairs)

    scored_docs = []

    for doc, rerank_score in zip(docs, rerank_scores):
        text = doc.page_content.lower()

        keyword_score = sum(word in text for word in query_words) / len(query_words)
        semantic_score = semantic_scores.get(doc.page_content, 0)

        final_score = (
            0.5 * rerank_score +
            0.3 * keyword_score +
            0.2 * semantic_score
        )

        scored_docs.append((doc, final_score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]

# =========================
# LOAD DOCUMENTS
# =========================
folder = "data"
documents = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        reader = PdfReader(f"{folder}/{file}")
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file, "page": page_num + 1}
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
# SAFE CHECKS
# =========================
if not query or len(query.strip()) < 5:
    print("⚠️ Query too short.")
    exit()

# =========================
# QUERY REWRITE
# =========================
rewritten_query = llm.invoke(
    f"Rewrite for legal retrieval:\n{query}"
).content.strip()

# =========================
# STEP 36 + 37 — MULTI QUERY + WEIGHTED FUSION
# =========================
expanded_queries = expand_query(rewritten_query)
weighted_queries = weight_queries(expanded_queries)

doc_scores = {}

for q, weight in weighted_queries:
    results = vector_store.similarity_search(q, k=5)

    for doc in results:
        key = doc.page_content[:120]

        if key not in doc_scores:
            doc_scores[key] = {
                "doc": doc,
                "score": 0
            }

        doc_scores[key]["score"] += weight

# sort fused results
sorted_docs = sorted(
    doc_scores.values(),
    key=lambda x: x["score"],
    reverse=True
)

vector_results = [item["doc"] for item in sorted_docs]

# =========================
# KEYWORD SEARCH
# =========================
def keyword_search(query, docs):
    words = query.lower().split()
    return [doc for doc in docs if any(w in doc.page_content.lower() for w in words)][:5]

keyword_results = keyword_search(query, docs)

# =========================
# HYBRID MERGE
# =========================
combined_docs = vector_results + keyword_results

# =========================
# DEDUP
# =========================
seen = set()
unique_docs = []

for doc in combined_docs:
    key = doc.page_content[:100]
    if key not in seen:
        seen.add(key)
        unique_docs.append(doc)

# =========================
# RERANKING
# =========================
filtered_docs = multi_signal_ranking(query, unique_docs, vector_results, top_k=6)

if not filtered_docs:
    print("⚠️ No relevant documents found.")
    exit()

# =========================
# CLEANING
# =========================
def clean_text(text):
    return " ".join(text.split())[:500]

cleaned_docs = [
    Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
    for doc in filtered_docs
]

# =========================
# CONTEXT BUILD
# =========================
context = build_context_with_budget(cleaned_docs)

if not context.strip():
    print("⚠️ Empty context.")
    exit()

# =========================
# PROMPT
# =========================
prompt = f"""
You are a strict legal AI assistant.

RULES:
- Use ONLY context
- No guessing
- Each statement must include source

FORMAT:
{{
  "answer": [
    {{
      "statement": "text",
      "source": {{"file": "...", "page": "..."}}
    }}
  ],
  "confidence": "high | medium | low"
}}

CONTEXT:
{context}

QUESTION:
{query}
"""

# =========================
# LLM RESPONSE
# =========================
response = llm.invoke(prompt)
result = parse_json_safe(response.content)

if not result:
    print("⚠️ JSON parsing failed")
    print(response.content)
    exit()

# =========================
# OUTPUT
# =========================
print("\nFINAL ANSWER:\n")

for item in result.get("answer", []):
    print(f"- {item['statement']}")
    print(f"  Source: {item['source']['file']} - Page {item['source']['page']}\n")

print("CONFIDENCE:", result.get("confidence", "N/A"))