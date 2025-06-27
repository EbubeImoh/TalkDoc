"""
vector_store.py
Handles storing and retrieving embeddings and metadata in Pinecone.
"""

import os
from pinecone import Pinecone
from typing import List, Dict

def init_pinecone():
    """Initialize Pinecone connection and create index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "talkdoc-index")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set")
    
    # Initialize Pinecone client with new API
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
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
    # Generate embedding for the query text using the same model
    from .embeddings import model
    
    try:
        # Generate embedding for the query
        query_embedding = model.encode([query_text], convert_to_tensor=False)[0]
        
        # Pad or truncate to 1024 dimensions to match Pinecone index
        if len(query_embedding) < 1024:
            # Pad with zeros
            query_embedding = list(query_embedding) + [0.0] * (1024 - len(query_embedding))
        else:
            # Truncate to 1024
            query_embedding = list(query_embedding[:1024])
        
        # Convert to Python float64
        query_embedding = [float(x) for x in query_embedding]
        
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        # Fallback to placeholder embedding
        query_embedding = [0.0] * 1024
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results.matches 