import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from pinecone import Pinecone

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "mediumblog1.txt"


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")

    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Source document not found: {SOURCE_FILE}")

    for name in ("PINECONE_API_KEY", "INDEX_NAME", "PINECONE_NAMESPACE"):
        if not os.environ.get(name):
            raise RuntimeError(f"Set {name} in {PROJECT_DIR / '.env'} or the environment")

    index_name = os.environ["INDEX_NAME"]
    pinecone = Pinecone()
    if index_name not in pinecone.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index {index_name!r} does not exist. "
            "Set INDEX_NAME to an existing 1024-dimensional index."
        )
    index_dimension = pinecone.describe_index(index_name).dimension
    if index_dimension != 1024:
        raise RuntimeError(
            f"Pinecone index {index_name!r} has {index_dimension} dimensions; "
            "llama-text-embed-v2 requires 1024."
        )

    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path=str(SOURCE_FILE),
        content_type="text/plain",
        encoding="utf-8",
        chunking_strategy="basic",
        max_characters=1_000_000,
    )
    documents = loader.load()

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise ValueError(f"Source document contains no text: {SOURCE_FILE}")
    print(f"created {len(texts)} chunks")

    embeddings = PineconeEmbeddings(model="llama-text-embed-v2")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=index_name,
        namespace=os.environ["PINECONE_NAMESPACE"],
    )
    print("finish")


if __name__ == "__main__":
    main()
