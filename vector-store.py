from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

# Load PDF
reader = PdfReader("data/constitution.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

# Embeddings
embedding_model = OpenAIEmbeddings()

# Create vector DB
vector_store = FAISS.from_texts(chunks, embedding_model)

# Test search
query = "What are fundamental rights?"

results = vector_store.similarity_search(query, k=3)

# Print results
for i, result in enumerate(results):
    print(f"\nResult {i+1}:\n", result.page_content)