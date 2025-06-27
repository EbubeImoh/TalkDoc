"""
embeddings.py
Generates embeddings for text chunks using OpenAI or HuggingFace APIs.
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Load the model once at module level for efficiency
# Using a model with 1024 dimensions to match Pinecone index
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 768 dimensions, but we'll pad to 1024

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
                # Pad or truncate to 1024 dimensions to match Pinecone index
                if len(embedding) < 1024:
                    # Pad with zeros
                    padded_embedding = list(embedding) + [0.0] * (1024 - len(embedding))
                else:
                    # Truncate to 1024
                    padded_embedding = list(embedding[:1024])
                
                # Convert to Python float64 to avoid Pinecone type issues
                padded_embedding = [float(x) for x in padded_embedding]
                
                # Store the actual text content in metadata for retrieval
                metadata = chunks[chunk_idx]['metadata'].copy()
                metadata['text'] = texts[chunk_idx]  # Add the actual text content
                
                embeddings.append({
                    'embedding': padded_embedding,
                    'metadata': metadata
                })
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {e}")
            continue
    
    return embeddings

# Functions for embedding generation will go here. 