# 🚀 QueryIT

A Retrieval Augmented Generation (RAG) web-app built with FastAPI, Inngest, Qdrant, and Google Gemini. This interactive app enables you to ingest PDF documents, store their embeddings in a vector database, and ability to query your documents with support of Google's Gemini LLM.

<img alt="image" src="assets/Demo.png" />

## ✨ Features

- **📄 PDF Ingestion**: Load and process PDF documents into searchable chunks
- **🔍 Vector Search**: Powered by Qdrant for fast similarity search
- **🤖 AI-Powered Q&A**: Uses Google Gemini for AI enabled queries
- **⚡ Async Workflows**: Inngest for event-driven, durable workflow orchestration platform
- **📊 Comprehensive Logging**: Full observability with emoji-rich logging
- **🎯 Production Ready**: Type-safe with Pydantic models and error handling

## 🏗️ Architecture

This system uses **[Inngest](https://www.inngest.com/)** - a workflow orchestration platform that provides:
- ⚡ **Event-driven workflows**: Trigger functions via events
- 🔄 **Automatic retries**: Built-in error handling and recovery
- 📊 **Observability**: Full visibility into workflow execution
- ⏱️ **Step management**: Break workflows into resumable steps
- 🎯 **Type safety**: Pydantic-based serialization


```
┌─────────────┐
│  Streamlit  │ ← Web UI (Optional)
│     UI      │
└──────┬──────┘
       │
┌──────▼──────┐
│   FastAPI   │ ← REST API Entry Point
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Inngest   │ ← Workflow Engine (Orchestrates everything)
└──────┬──────┘
       │
       ├──► 📖 Ingest PDF Workflow
       │    ├─ Load & Chunk PDF
       │    ├─ Generate Embeddings (Gemini)
       │    └─ Store in Qdrant
       │
       └──► 🔍 Query PDF Workflow
            ├─ Embed Question (Gemini)
            ├─ Search Vector DB (Qdrant)
            └─ Generate Answer (Gemini)
```

## 📋 Prerequisites

- Python 3.12+
- Qdrant (running locally or remote)
- Google Gemini API Key
- Inngest Dev Server (for local development)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Zenith369/QueryIT.git
cd QueryIT
```

### 2. Install Dependencies

Using `uv` :
```bash
uv sync
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Configuration (optional, defaults shown)
QDRANT_URL=http://localhost:6333
```

> **Note**: The `.env` file is already in `.gitignore` to keep your secrets safe.

### 4. Start Qdrant

```bash
docker run -d --name qdrantRag -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 5. Start Inngest Dev Server

In a separate terminal:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

This will start the Inngest Dev Server at `http://localhost:8288`.

**Services will be available at:**
- 🚀 FastAPI: http://localhost:8000
- 🎨 Streamlit UI: http://localhost:8501
- 🔍 Qdrant: http://localhost:6333
- ⚡ Inngest: http://localhost:8288


### 🔧 Local Deployment

For local development:

```bash
uvicorn main:app --reload --log-level info
```

The API will be available at `http://localhost:8000`.

### Quick Start with Streamlit UI 🎨

```bash
# Make sure all services are running (Qdrant, Inngest, FastAPI)
# Then start the Streamlit UI:
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. You'll see:
1. **📤 Upload Section**: Drag and drop your PDF file
2. **💬 Chat Interface**: For all your queries
3. **📊 Results**: AI's output with source citations
4. **⚙️ Settings**: Adjust retrieval parameters (top_k)

**No curl commands needed!** Perfect for demos and non-technical users.

---

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/
```

#### 2. Ingest a PDF

Send an event to ingest a PDF document:

```bash
curl -X POST http://localhost:8288/e/rag_app \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag/ingest_pdf",
    "data": {
      "pdf_path": "/path/to/your/document.pdf",
      "source": "my-document"
    }
  }'
```

**What happens:**
- 📖 Loads the PDF and splits it into chunks (1000 chars, 200 overlap)
- 🔢 Generates 3072-dimensional embeddings using Gemini
- 💾 Stores vectors in Qdrant with metadata

#### 3. Query the PDF

Ask questions about your ingested documents:

```bash
curl -X POST http://localhost:8288/e/rag_app \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag/query_pdf_ai",
    "data": {
      "question": "What is the main topic of the document?",
      "top_k": 5
    }
  }'
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": ["my-document"],
  "num_contexts": 5
}
```

**What happens:**
- 🔍 Embeds your question using Gemini
- 📊 Searches Qdrant for the top 5 most relevant chunks
- 🤖 Sends context to Gemini LLM for answer generation
- ✅ Returns the answer with source attribution

### 4. Use the Streamlit UI (Recommended for Easy Testing)

For a user-friendly interface, you can use the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

The UI will open at `http://localhost:8501` and provides:
- 📤 **File Upload**: Drag and drop PDF files to ingest
- 💬 **Chat Interface**: For your queries
- 📊 **Source Display**: Analyze document chunks that were used
- ⚙️ **Configuration**: Adjust `top_k` and other parameters
- 🎨 **Visual Feedback**: Real-time status updates and loading indicators

**Benefits of the Streamlit UI:**
- No need to write curl commands
- Interactive file upload
- Instant visual feedback
- Better for demos and testing
- User-friendly for non-technical users

### View Workflow Execution

Open the Inngest Dev Server UI at `http://localhost:8288` to see:
- Real-time workflow execution
- Step-by-step logs
- Error tracking
- Execution history
- Retry attempts and failures
- Detailed timing information

## 📁 Project Structure

```
Prodution_Ready_RAG/
├── main.py              # FastAPI app + Inngest functions
├── streamlit_app.py     # Streamlit UI
├── data_loader.py       # PDF loading and embedding logic
├── vector_db.py         # Qdrant vector database wrapper
├── constants.py         # Configuration constants
├── custom_types.py      # Pydantic models
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore file
├── pyproject.toml      # Project dependencies
├── README.md           # This file
└── qdrant_storage/     # Qdrant data directory
```

## 🎛️ Configuration

Edit `constants.py` to customize:

```python
# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docs"

# Embedding
MAX_EMBED_BATCH = 100
EMBED_BATCH_DELAY_S = 1.2
EMBED_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072

# Chunking
CHUNK_SIZE = 1000
OVERLAP = 200

# LLM
GEMINI_LLM_MODEL = "gemini-2.5-flash"
```

## 📊 Logging

The system includes comprehensive logging with emoji indicators:

```
INFO: 📖 Loading PDF: document.pdf
INFO: ✂️ Split into 42 chunks
INFO: 🔢 Embedding 42 chunks...
INFO: ✅ Generated 42 vectors (dim=3072)
INFO: 💾 Upserted 42 vectors to Qdrant
INFO: 🔍 Searching for question: What is RAG?
INFO: ✅ Embedded question into vector of dim=3072
INFO: 📊 Found 5 contexts
INFO: 🎯 Step completed: found 5 contexts from 1 sources
```

## 🔧 Advanced Usage

### Custom Embedding Model

To use a different embedding model:

1. Update `constants.py`:
```python
EMBED_MODEL = "your-model-name"
EMBEDDING_DIM = your_dimension
```

2. Modify `data_loader.py` if needed for different API calls

### Multiple Document Sources

Track different sources by setting the `source` parameter:

```json
{
  "name": "rag/ingest_pdf",
  "data": {
    "pdf_path": "/path/to/doc.pdf",
    "source": "research-papers"
  }
}
```

### Adjust Retrieval

Modify the `top_k` parameter to retrieve more or fewer context chunks:

```json
{
  "name": "rag/query_pdf_ai",
  "data": {
    "question": "Your question?",
    "top_k": 10  // Retrieve top 10 chunks
  }
}
```


## 📚 Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[Inngest](https://www.inngest.com/)** - Workflow orchestration
- **[Qdrant](https://qdrant.tech/)** - Vector database
- **[Google Gemini](https://ai.google.dev/)** - Embeddings and LLM
- **[LlamaIndex](https://www.llamaindex.ai/)** - PDF parsing and chunking
- **[Streamlit](https://streamlit.io/)** - Interactive web UI
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** - Environment management