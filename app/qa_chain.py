"""
qa_chain.py
Orchestrates retrieval of relevant chunks and answer generation using LangChain RetrievalQA.
"""

import os
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from pydantic import Field
from .vector_store import query_similar

class PineconeRetriever(BaseRetriever):
    """Custom retriever for Pinecone vector store."""
    
    pinecone_index: Any = Field(description="Pinecone index instance")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    
    def __init__(self, pinecone_index, top_k: int = 5):
        """Initialize the retriever with a Pinecone index."""
        super().__init__(pinecone_index=pinecone_index, top_k=top_k)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from Pinecone."""
        matches = query_similar(query, self.pinecone_index, self.top_k)
        documents = []
        for match in matches:
            # Get the actual text content from metadata
            text_content = match.metadata.get('text', f"Document chunk from {match.metadata.get('source', 'unknown')}")
            
            doc = Document(
                page_content=text_content,
                metadata=match.metadata
            )
            documents.append(doc)
        return documents
    
    def invoke(self, input: Dict[str, Any], config: Any = None) -> List[Document]:
        """Invoke the retriever (newer method)."""
        query = input.get("query", "")
        return self.get_relevant_documents(query)

def setup_llm():
    """Setup Ollama LLM."""
    try:
        # Use llama3:latest model - you can change this to any model from your ollama list
        llm = OllamaLLM(
            model="llama3:latest",
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
        
        # Test the connection
        response = llm.invoke("Hello")
        print(f"✅ Ollama connection successful. Test response: {response[:50]}...")
        
        return llm
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return None

def create_qa_chain(index):
    """Create LangChain RetrievalQA chain."""
    retriever = PineconeRetriever(index)
    llm = setup_llm()

    if llm is None:
        # Fallback: return a simple function that just returns the retrieved documents
        def simple_qa_chain(query_dict):
            # Use invoke for future compatibility
            docs = retriever.invoke(query_dict)
            return {
                "result": f"Retrieved {len(docs)} relevant documents. Please check the sources below.",
                "source_documents": docs
            }
        return simple_qa_chain

    # Create a custom prompt template for better answers
    from langchain.prompts import PromptTemplate

    prompt_template = """You are a helpful AI assistant. Based on the following context, provide a detailed and informative answer to the question. If the context doesn't contain enough information to answer the question properly, say "I cannot provide a complete answer based on the available information."

Context: {context}

Question: {question}

Please provide a comprehensive answer that directly addresses the question using the information from the context."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Wrap the chain to ensure consistent return format
    def wrapped_qa_chain(query_dict):
        try:
            result = qa_chain.invoke(query_dict)
            # Handle different result formats
            if isinstance(result, dict):
                return result
            else:
                # If result is a string, create the expected format
                docs = retriever.invoke(query_dict)
                return {
                    "result": str(result),
                    "source_documents": docs if isinstance(docs, list) else []
                }
        except Exception as e:
            # Fallback to simple retrieval
            docs = retriever.invoke(query_dict)
            return {
                "result": f"Error in LLM processing: {str(e)}. Retrieved {len(docs) if isinstance(docs, list) else 0} relevant documents.",
                "source_documents": docs if isinstance(docs, list) else []
            }

    return wrapped_qa_chain

def answer_question(question: str, qa_chain) -> Dict:
    """
    Generate answer for a question using the QA chain.
    Args:
        question: User's question.
        qa_chain: Wrapped QA chain function.
    Returns:
        Dict with 'answer' and 'sources'.
    """
    try:
        # The qa_chain is now always a function that returns the expected format
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        print(f"DEBUG: Exception in answer_question: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        } 