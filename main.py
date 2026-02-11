import os
import logging
import uuid
from fastapi import FastAPI

import inngest
import inngest.fast_api
from inngest.experimental.ai import gemini
from dotenv import load_dotenv

from custom_types import *
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
import constants as const

# Load environment variables
load_dotenv()

# Inngest Client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# RAG: Ingest PDF
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    )

async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        """
        Retrieving data from inngest.Context, loading and chunking
        the data calling load_and_chunk_pdf function then returning
        it as a RAGChunkAndSrc.

        Args:
            ctx (inngest.Context): Inngest native Context object

        Returns:
            RAGChunkAndSrc: Pydantic object holding data chunks and source ids 
        """

        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source", pdf_path)

        logging.info(f"📖 Loading PDF: {pdf_path}")

        chunks = load_and_chunk_pdf(pdf_path)

        logging.info(f"✂️ Split into {len(chunks)} chunks")

        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        """
        Retrieves the data chunks and source ids then
        updates and inserts points into Qdrant database
        using QdrantStorage().upsert()

        Args:
            chunks_and_src (RAGChunkAndSrc): Pydantic object storing
                                             chunks and source ids

        Returns:
            RAGUpsertResult: Number of points inserted into the Qdrant database
        """
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        logging.info(f"🔢 Embedding {len(chunks)} chunks...")

        # Using Gemini embedding model
        vectors = embed_texts(chunks)

        logging.info(
            f"✅ Generated {len(vectors)} vectors (dim={len(vectors[0]) if vectors else 0})"
        )
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vectors, payloads)

        logging.info(f"💾 Upserted {len(ids)} vectors to Qdrant")

        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()


# RAG: Query PDF (Gemini)
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        """
        Enables search functionality through search() for a Qdrant
        database and supports converting input to embedding by leveraging
        embed_texts().

        Args:
            question (str): Input query
            top_k (int, optional): Top k results from the Qdrant database.
                                   Defaults to 5.

        Returns:
            RAGSearchResult: Contexts and sources relating to input query
        """
        logging.info(f"🔍 Searching for question: {question}")

        query_vector = embed_texts([question])[0]
        logging.info(f"✅ Embedded question into vector of dim={len(query_vector)}")

        db = QdrantStorage()
        found = db.search(query_vector, top_k)

        logging.info(f"📊 Found {len(found['contexts'])} contexts")
        logging.debug(f"Contexts: {found['contexts'][:2]}")  # Log first 2 contexts

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
        )

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    logging.info(f"📝 Query received: question='{question}', top_k={top_k}")

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    logging.info(
        f"🎯 Step completed: found {len(found.contexts)} contexts from {len(set(found.sources))} sources"
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    logging.debug(f"📄 Context block:\n{context_block[:500]}...")  # Log first 500 chars

    user_content = (
        "Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
    )

    adapter = gemini.Adapter(
        auth_key=os.getenv("GEMINI_API_KEY"),
        model=const.GEMINI_LLM_MODEL,
    )

    # Querying Gemini LLM
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}],
                }
            ],
        },
    )

    answer = res["candidates"][0]["content"]["parts"][0]["text"].strip()

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


# FastAPI Server
app = FastAPI()
inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai],
)