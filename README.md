qdrant : docker run -d --name qdrantRag -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant

inngest-cli : npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

fastapi-server : uv run uvicorn main:app --reload