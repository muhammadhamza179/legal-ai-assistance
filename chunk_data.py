from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Load PDF
reader = PdfReader("data/constitution.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Print results
print("Total chunks:", len(chunks))
print("\nFirst chunk:\n", chunks[0])