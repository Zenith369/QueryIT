QDRANT_URL = "http://localhost:6333"
EMBEDDING_DIM = 3072  # 1536
MAX_EMBED_BATCH = 100
EMBED_BATCH_DELAY_S = 1.2 # for rate limit
COLLECTION_NAME = "docs"

EMBED_MODEL = "gemini-embedding-001"  # "gemini-embedding-004
CHUNK_SIZE = 1000
OVERLAP = 200#80

GEMINI_LLM_MODEL = "gemini-2.5-flash"
