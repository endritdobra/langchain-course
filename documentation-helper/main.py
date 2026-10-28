"""Ask the indexed documentation a question using an Ollama Cloud model."""

import argparse
import os

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from ingestion import EMBEDDING_MODEL


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question about the indexed documentation")
    args = parser.parse_args()

    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        parser.error("Set OLLAMA_API_KEY in the environment or .env first")

    store = Chroma(
        collection_name="documentation",
        persist_directory="chroma_db",
        embedding_function=FastEmbedEmbeddings(model_name=EMBEDDING_MODEL),
    )
    if not store.get(limit=1)["ids"]:
        parser.error("The index is empty. Run python ingestion.py first")
    documents = store.similarity_search(args.question, k=4)
    context = "\n\n".join(
        f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in documents
    )
    response = requests.post(
        "https://ollama.com/api/chat",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": os.getenv("OLLAMA_MODEL", "gemma4:31b"),
            "messages": [
                {"role": "system", "content": "Answer using the provided documentation. If it does not contain the answer, say so. Cite source URLs."},
                {"role": "user", "content": f"Documentation:\n{context}\n\nQuestion: {args.question}"},
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    print(response.json()["message"]["content"])


if __name__ == "__main__":
    main()
