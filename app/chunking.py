"""
chunking.py
Splits extracted text into overlapping chunks and attaches metadata (source, page, etc.).
"""

def chunk_texts(
    docs,
    chunk_size=500,
    chunk_overlap=50
):
    """
    Splits each document's text into overlapping chunks.
    Args:
        docs: List of dicts with 'text' and 'metadata'.
        chunk_size: Number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
    Returns:
        List of dicts with 'text' and updated 'metadata' (including chunk index).
    """
    chunked = []
    for doc in docs:
        text = doc['text']
        metadata = doc.get('metadata', {})
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk'] = chunk_idx
                chunked.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_metadata
                })
                chunk_idx += 1
            start += chunk_size - chunk_overlap
    return chunked

# Functions for text chunking will go here. 