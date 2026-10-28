"""Answer questions using the local documentation index and Ollama Cloud."""

import os
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def run_llm(query: str) -> dict[str, Any]:
    """Return an Ollama answer and the documentation used as context."""
    load_dotenv()
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("Set OLLAMA_API_KEY in the environment or .env first")

    vectorstore = Chroma(
        collection_name="documentation",
        persist_directory="chroma_db",
        embedding_function=FastEmbedEmbeddings(model_name=EMBEDDING_MODEL),
    )
    if not vectorstore.get(limit=1)["ids"]:
        raise RuntimeError("The index is empty. Run python ingestion.py first")

    documents = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in documents
    )
    response = requests.post(
        "https://ollama.com/api/chat",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": os.getenv("OLLAMA_MODEL", "gemma4:31b"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using the provided documentation. If it does not "
                        "contain the answer, say so. Cite source URLs."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Documentation:\n{context}\n\nQuestion: {query}",
                },
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return {"answer": response.json()["message"]["content"], "context": documents}


if __name__ == "__main__":
    print(run_llm("what are deep agents?"))
