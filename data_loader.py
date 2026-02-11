import os
import time
import hashlib
from typing import List

from google import genai
from google.genai import types

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from constants import *

load_dotenv()

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """
    Retrieves Google gemini client using an API key
    from .env file.

    Raises:
        RuntimeError: If the api key is not set.

    Returns:
        genai.Client: Google gemini client
    """
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    _client = genai.Client(api_key=api_key)
    return _client


splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)


def load_and_chunk_pdf(path: str):
    """
    Loads pdf files calling PDFReader() from llama.index library
    and then split the texts into chunks using SentenceSplitter()
    from llama.index library.

    Args:
        path (str): Source of the pdf file

    Returns:
        chunks: Split texts from the source file
    """

    docs = PDFReader().load_data(path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    # Split each document correctly
    chunks = splitter.split_texts(texts)

    # Deduplicate chunks (huge quota saver)
    chunks = list(dict.fromkeys(c.strip() for c in chunks if c.strip()))

    return chunks

def _batch_iter(items: list[str], batch_size: int):
    """_summary_

    Args:
        items (list[str]): Texts from the source
        batch_size (int): Batch size

    Yields:
        batch(int): single batch of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# Embedder
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds input text using gemini client with throttling support.

    Args:
        texts (list[str]): Input texts

    Returns:
        list[list[float]]: Embeddings of the input texts
    """
    gemini_client = get_client()

    all_embeddings: list[list[float]] = []

    for batch in _batch_iter(texts, MAX_EMBED_BATCH):
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )

        # Gemini returns embeddings in the same order as inputs
        batch_embeddings = [item.values for item in result.embeddings]
        all_embeddings.extend(batch_embeddings)

        # Throttle to avoid quota exhaustion
        time.sleep(EMBED_BATCH_DELAY_S)

    return all_embeddings
