"""
ui.py
Implements the user interface for document upload and Q&A using Streamlit or FastAPI.
"""

import streamlit as st
import os
import tempfile
from typing import List, Dict
from ingestion import ingest_document
from chunking import chunk_texts
from embeddings import generate_embeddings
from vector_store import init_pinecone, store_embeddings
from qa_chain import create_qa_chain, answer_question

def main():
    st.set_page_config(page_title="TalkDoc - Document Q&A System", layout="wide")
    st.title("📄 TalkDoc - Document Q&A System")
    st.markdown("Upload your documents and ask questions to get AI-powered answers!")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'index' not in st.session_state:
        st.session_state.index = None

    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🚀 Process Documents"):
                process_documents(uploaded_files)

    # Main area for Q&A
    if st.session_state.qa_chain:
        st.header("❓ Ask Questions")
        question = st.text_input("Enter your question:")
        
        if question:
            if st.button("🔍 Get Answer"):
                with st.spinner("Generating answer..."):
                    result = answer_question(question, st.session_state.qa_chain)
                    
                    st.subheader("💡 Answer")
                    st.write(result["answer"])
                    
                    if result["sources"]:
                        st.subheader("📚 Sources")
                        for source in result["sources"]:
                            st.write(f"- {source.get('source', 'Unknown')} (Page: {source.get('page', 'N/A')})")
    else:
        st.info("👈 Please upload and process documents first to start asking questions!")

def process_documents(uploaded_files: List):
    """Process uploaded documents through the entire pipeline."""
    try:
        # Initialize Pinecone
        with st.spinner("🔗 Connecting to Pinecone..."):
            index = init_pinecone()
            st.session_state.index = index

        all_chunks = []
        
        # Process each file
        for uploaded_file in st.progress_bar(uploaded_files, text="Processing documents..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Ingest document
                docs = ingest_document(tmp_path)
                
                # Chunk text
                chunks = chunk_texts(docs)
                all_chunks.extend(chunks)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue

        if all_chunks:
            # Generate embeddings
            with st.spinner("🧠 Generating embeddings..."):
                embeddings = generate_embeddings(all_chunks)

            # Store in Pinecone
            with st.spinner("💾 Storing in vector database..."):
                store_embeddings(embeddings, index)

            # Setup QA chain
            with st.spinner("🔧 Setting up Q&A system..."):
                qa_chain = create_qa_chain(index)
                st.session_state.qa_chain = qa_chain

            st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)! You can now ask questions.")

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 