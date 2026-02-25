from pypdf import PdfReader

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "text": text,
                "metadata": {
                    "source": file_path,
                    "page": page_number + 1
                }
            })

    return documents