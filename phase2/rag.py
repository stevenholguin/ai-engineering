"""
RAG (Retrieval-Augmented Generation) - Phase 2.

Loads the care-protocol documents in knowledge/, splits them into chunks,
embeds them, and stores them in a local Chroma vectorstore. This is what
lets the agent answer questions like "what should I do if my fever is
37.8°C?" while citing an actual source document instead of relying purely
on the model's training data.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

_vectorstore = None  # simple module-level cache, no framework magic here


def _build_vectorstore() -> Chroma:
    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    else:
        _vectorstore = _build_vectorstore()

    return _vectorstore


def search_care_protocols(query: str, k: int = 3) -> str:
    """Returns the top-k relevant chunks from the care protocol documents,
    formatted with their source file so the agent can cite them."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    if not results:
        return "No relevant care protocol information found."

    parts = []
    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n".join(parts)
