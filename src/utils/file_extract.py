"""
file_extract.py
Extracts plain text from an uploaded CV file (PDF, DOCX, or TXT) so it can
be passed into the Career Advisor's prompt as extra context.
"""

import io


def extract_text_from_upload(uploaded_file) -> str:
    """uploaded_file: a Streamlit UploadedFile object."""
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        elif name.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)

        elif name.endswith(".txt"):
            return data.decode("utf-8", errors="ignore")

        else:
            return "[Unsupported file type — please upload PDF, DOCX, or TXT]"

    except Exception as e:
        return f"[Could not read file: {e}]"