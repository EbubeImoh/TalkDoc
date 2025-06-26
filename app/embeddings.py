"""
embeddings.py
Generates embeddings for text chunks using OpenAI or HuggingFace APIs.
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

# Load the model once at module level for efficiency
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(
    chunks: List[Dict],
    batch_size: int = 32
) -> List[Dict]:
    """
    Generates embeddings for a list of text chunks using HuggingFace sentence-transformers.
    Args:
        chunks: List of dicts with 'text' and 'metadata'.
        batch_size: Number of texts to process in each batch.
    Returns:
        List of dicts with 'embedding' and 'metadata'.
    """
    texts = [chunk['text'] for chunk in chunks]
    embeddings = []
    
    # Generate embeddings in batches for efficiency
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=False)
            
            for j, embedding in enumerate(batch_embeddings):
                chunk_idx = i + j
                embeddings.append({
                    'embedding': embedding.tolist(),
                    'metadata': chunks[chunk_idx]['metadata']
                })
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {e}")
            continue
    
    return embeddings

# Functions for embedding generation will go here. 