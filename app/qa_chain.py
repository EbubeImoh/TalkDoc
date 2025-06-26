"""
qa_chain.py
Orchestrates retrieval of relevant chunks and answer generation using LangChain RetrievalQA.
"""

import os
from typing import List, Dict
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vector_store import query_similar

class PineconeRetriever(BaseRetriever):
    """Custom retriever for Pinecone vector store."""
    
    def __init__(self, index, top_k: int = 5):
        self.index = index
        self.top_k = top_k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from Pinecone."""
        matches = query_similar(query, self.index, self.top_k)
        documents = []
        for match in matches:
            # Reconstruct the original text from metadata
            # You might want to store the actual text in Pinecone metadata
            # For now, we'll use a placeholder
            doc = Document(
                page_content=f"Document chunk from {match.metadata.get('source', 'unknown')}",
                metadata=match.metadata
            )
            documents.append(doc)
        return documents

def setup_llm():
    """Setup open-source LLM using HuggingFace."""
    # Using a smaller, efficient model for local inference
    model_name = "microsoft/DialoGPT-medium"  # You can change this to other models
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def create_qa_chain(index):
    """Create LangChain RetrievalQA chain."""
    retriever = PineconeRetriever(index)
    llm = setup_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def answer_question(question: str, qa_chain) -> Dict:
    """
    Generate answer for a question using the QA chain.
    Args:
        question: User's question.
        qa_chain: LangChain RetrievalQA chain.
    Returns:
        Dict with 'answer' and 'sources'.
    """
    try:
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        } 