# 📄 TalkDoc - Document Q&A System

An AI-powered application that allows users to upload documents and ask questions to get accurate, context-aware answers using LangChain and Pinecone.

## 🚀 Features

- **Multi-format Support**: Upload PDFs, DOCX, TXT, and Markdown files
- **Smart Text Processing**: Automatic text extraction and chunking
- **Vector Search**: Fast semantic search using Pinecone
- **Open Source**: Uses HuggingFace models (no API costs for embeddings/LLM)
- **User-friendly UI**: Clean Streamlit interface
- **Source Tracking**: See which documents and pages were used for answers

## 🛠️ Tech Stack

- **LangChain**: LLM orchestration and retrieval
- **HuggingFace**: Open-source embeddings (sentence-transformers) and LLM
- **Pinecone**: Vector database for similarity search
- **Streamlit**: Web interface
- **PyMuPDF**: PDF processing
- **python-docx**: DOCX processing

## 📋 Prerequisites

- Python 3.8+
- Pinecone account and API key
- (Optional) OpenAI API key for alternative LLM

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd TalkDoc
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# Required
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=talkdoc-index

# Optional
OPENAI_API_KEY=your-openai-api-key
```

### 3. Run the Application

```bash
python main.py
```

Or directly with Streamlit:

```bash
streamlit run app/ui.py
```

The app will be available at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, TXT, or MD files
2. **Process Documents**: Click "Process Documents" to extract, chunk, and embed your files
3. **Ask Questions**: Enter questions in natural language and get AI-powered answers
4. **View Sources**: See which documents and pages were used to generate answers

## 🏗️ Architecture

```
Document Upload → Text Extraction → Chunking → Embedding → Pinecone Storage
                                                           ↓
User Question → Embedding → Similarity Search → Context Retrieval → LLM Answer
```

### Core Modules

- **`app/ingestion.py`**: Document parsing and text extraction
- **`app/chunking.py`**: Text chunking with metadata preservation
- **`app/embeddings.py`**: Vector embedding generation
- **`app/vector_store.py`**: Pinecone integration
- **`app/qa_chain.py`**: LangChain RetrievalQA orchestration
- **`app/ui.py`**: Streamlit user interface

## 🔧 Configuration

### Chunking Parameters

Modify in `app/chunking.py`:
- `chunk_size`: Number of characters per chunk (default: 500)
- `chunk_overlap`: Overlapping characters between chunks (default: 50)

### Embedding Model

Change in `app/embeddings.py`:
- Model: `all-MiniLM-L6-v2` (384 dimensions, fast and accurate)

### LLM Model

Modify in `app/qa_chain.py`:
- Current: `microsoft/DialoGPT-medium`
- Alternatives: `gpt2`, `EleutherAI/gpt-neo-125M`, etc.

## 🚀 Deployment

### Local Development

```bash
python main.py
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/ui.py", "--server.port=8501"]
```

### Cloud Deployment

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Use Procfile and requirements.txt
- **AWS/GCP**: Container deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Pinecone Connection Error**
   - Verify API key in `.env`
   - Check Pinecone dashboard for index status

2. **Model Download Issues**
   - Ensure stable internet connection
   - Models are downloaded automatically on first use

3. **Memory Issues**
   - Reduce batch size in `app/embeddings.py`
   - Use smaller models for LLM

### Performance Tips

- Use GPU if available for faster embedding generation
- Adjust chunk size based on document complexity
- Consider using Pinecone's serverless option for cost optimization

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review Pinecone and HuggingFace documentation
