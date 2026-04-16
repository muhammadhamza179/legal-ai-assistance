from pypdf import PdfReader

reader = PdfReader("data/constitution.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

print(text[:1000])  # print first 1000 characters