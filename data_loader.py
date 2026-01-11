import os
# from openai import OpenAI
from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# EMBED_MODEL = "text-embedding-3-large" # for openai
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# def get_client() -> OpenAI:
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError("OPENAI_API_KEY is not set")
#     return OpenAI()

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    _client = genai.Client(api_key=api_key)
    return _client


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path) # type: ignore
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )

    return [emb.values for emb in response.embeddings]
