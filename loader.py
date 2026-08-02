# the goal of this file is to read the text document or pdf .
import os
from pathlib import Path
from pypdf import PdfReader  # pip install pypdf

def load_documents(folder_path: str) -> list[dict]:
    """
    Reads all .pdf and .txt files from a folder.
    Returns a list of dicts: {"filename": ..., "text": ...}
    Filename is kept attached because later steps (citation)
    need to trace every answer back to its source file.
    """
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_path in folder.iterdir():
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            documents.append({"filename": file_path.name, "text": text})

        elif file_path.suffix.lower() == ".pdf":
            text = _read_pdf(file_path)
            documents.append({"filename": file_path.name, "text": text})

    if not documents:
        raise ValueError(f"No .pdf or .txt files found in {folder_path}")

    return documents


def _read_pdf(file_path: Path) -> str:
    """
    Extracts text page by page and joins it.
    Underscore prefix = internal helper, not called from outside this file.
    """
    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""  # some pages have no extractable text
        pages_text.append(page_text)
    return "\n".join(pages_text)