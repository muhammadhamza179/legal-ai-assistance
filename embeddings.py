from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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

# Create embeddings
embedding_model = OpenAIEmbeddings()

vectors = embedding_model.embed_documents(chunks)

# Print results
print("Total chunks:", len(chunks))
print("Vector length:", len(vectors[0]))
print("First vector sample:", vectors[0][:5])