"""
ingestion.py
Handles document upload and parsing for .pdf, .txt, .docx, and .md files.
Extracts and cleans text for downstream processing.
"""

import os
from typing import List, Dict

# PDF
import fitz  # PyMuPDF
# DOCX
from docx import Document


def ingest_pdf(file_path: str) -> List[Dict]:
    """Extracts text from each page of a PDF, returns list of dicts with text and metadata."""
    doc = fitz.open(file_path)
    results = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            results.append({
                'text': text.strip(),
                'metadata': {
                    'source': os.path.basename(file_path),
                    'page': page_num + 1
                }
            })
    return results


def ingest_docx(file_path: str) -> List[Dict]:
    """Extracts text from a DOCX file, returns list of dicts with text and metadata."""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    if full_text:
        return [{
            'text': '\n'.join(full_text),
            'metadata': {
                'source': os.path.basename(file_path)
            }
        }]
    return []


def ingest_txt(file_path: str) -> List[Dict]:
    """Reads a plain text or markdown file, returns list of dicts with text and metadata."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if text:
        return [{
            'text': text,
            'metadata': {
                'source': os.path.basename(file_path)
            }
        }]
    return []


def ingest_document(file_path: str) -> List[Dict]:
    """
    Main entry point. Detects file type and calls appropriate ingestion function.
    Returns a list of dicts with 'text' and 'metadata'.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        return ingest_pdf(file_path)
    elif ext == '.docx':
        return ingest_docx(file_path)
    elif ext in ['.txt', '.md']:
        return ingest_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}") 