"""
vector_store.py
Handles storing and retrieving embeddings and metadata in Pinecone.
"""

import os
from pinecone import Pinecone
from typing import List, Dict
from embeddings import model

def init_pinecone():
    """Initialize Pinecone connection and create index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "talkdoc-index")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set")
    
    # Initialize Pinecone client with new API
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine"
        )
    
    return pc.Index(index_name)

def store_embeddings(embeddings: List[Dict], index):
    """
    Store embeddings in Pinecone with metadata.
    Args:
        embeddings: List of dicts with 'embedding' and 'metadata'.
        index: Pinecone index instance.
    """
    vectors = []
    for i, emb in enumerate(embeddings):
        vector_id = f"chunk_{i}_{emb['metadata'].get('source', 'unknown')}"
        vectors.append({
            'id': vector_id,
            'values': emb['embedding'],
            'metadata': emb['metadata']
        })
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting batch {i//batch_size}: {e}")

def query_similar(query_text: str, index, top_k: int = 5):
    """
    Query Pinecone for similar embeddings.
    Args:
        query_text: Text to find similar embeddings for.
        index: Pinecone index instance.
        top_k: Number of similar results to return.
    Returns:
        List of similar embeddings with metadata.
    """
    # Generate embedding for query
    query_embedding = model.encode([query_text], convert_to_tensor=False)[0].tolist()
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.matches 