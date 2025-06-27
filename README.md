# 📄 TalkDoc - Document Q&A System

An AI-powered application that allows users to upload documents and ask questions to get accurate, context-aware answers using LangChain, Pinecone, and Ollama.

## 🚀 Features

- **Multi-format Support**: Upload PDFs, DOCX, TXT, and Markdown files
- **Smart Text Processing**: Automatic text extraction and chunking
- **Vector Search**: Fast semantic search using Pinecone
- **Local AI Models**: Uses Ollama for privacy and cost-free LLM inference
- **User-friendly UI**: Clean Streamlit interface
- **Source Tracking**: See which documents and pages were used for answers
- **Real-time Processing**: Live progress tracking during document processing

## 🛠️ Tech Stack

- **LangChain**: LLM orchestration and retrieval
- **Ollama**: Local LLM inference (supports Llama3, Qwen, and other models)
- **Sentence Transformers**: Open-source embeddings for semantic search
- **Pinecone**: Vector database for similarity search
- **Streamlit**: Web interface
- **PyMuPDF**: PDF processing
- **python-docx**: DOCX processing

## 📋 Prerequisites

- Python 3.8+
- Pinecone account and API key
- Ollama installed and running locally
- At least one Ollama model downloaded (e.g., llama3:latest)

## 🚀 Quick Start

### 1. Install Ollama

First, install Ollama on your system:

```bash
# macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Download a Model

```bash
# Start Ollama service
ollama serve

# Download a model (in a new terminal)
ollama pull llama3:latest
```

### 3. Clone and Setup

```bash
git clone <your-repo-url>
cd TalkDoc
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Required
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=talkdoc-index
```

### 5. Run the Application

```bash
streamlit run main.py
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
User Question → Embedding → Similarity Search → Context Retrieval → Ollama LLM → Answer
```

### Core Modules

- **`main.py`**: Application entry point and Streamlit interface
- **`app/ingestion.py`**: Document parsing and text extraction
- **`app/chunking.py`**: Text chunking with metadata preservation
- **`app/embeddings.py`**: Vector embedding generation using sentence-transformers
- **`app/vector_store.py`**: Pinecone integration and similarity search
- **`app/qa_chain.py`**: LangChain RetrievalQA orchestration with Ollama

## 🔧 Configuration

### Chunking Parameters

Modify in `app/chunking.py`:
- `chunk_size`: Number of characters per chunk (default: 500)
- `chunk_overlap`: Overlapping characters between chunks (default: 50)

### Embedding Model

Change in `app/embeddings.py`:
- Model: `all-mpnet-base-v2` (768 dimensions, high quality)
- Automatically padded to 1024 dimensions for Pinecone compatibility

### LLM Model

Modify in `app/qa_chain.py`:
- Current: `llama3:latest`
- Available models: `qwen2:7b`, `qwen2.5:latest`, `llama3.2:latest`, `tinyllama:latest`

To use a different model:
```python
llm = OllamaLLM(
    model="qwen2.5:latest",  # Change this to any model from ollama list
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)
```

## 🚀 Deployment

### Local Development

```bash
streamlit run main.py
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501"]
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

1. **Ollama Connection Error**
   ```bash
   # Make sure Ollama is running
   ollama serve
   
   # Check available models
   ollama list
   ```

2. **Pinecone Connection Error**
   - Verify API key in `.env`
   - Check Pinecone dashboard for index status

3. **Model Not Found**
   ```bash
   # Download the model
   ollama pull llama3:latest
   ```

4. **Memory Issues**
   - Use smaller models like `tinyllama:latest`
   - Reduce batch size in `app/embeddings.py`

### Performance Tips

- Use GPU if available for faster embedding generation
- Adjust chunk size based on document complexity
- Consider using Pinecone's serverless option for cost optimization
- Use smaller Ollama models for faster inference

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review Pinecone and Ollama documentation

## 🔄 Recent Updates

- **Ollama Integration**: Switched from local HuggingFace models to Ollama for better performance
- **Improved Embeddings**: Using all-mpnet-base-v2 for higher quality semantic search
- **Better Prompt Templates**: Enhanced answer generation with custom prompts
- **Source Tracking**: Improved metadata storage and retrieval
- **Error Handling**: Better error messages and fallback mechanisms
