"""Crawl public documentation and build a local Chroma index."""

import argparse
from collections import deque
from hashlib import sha256
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

from logger import log_header, log_info, log_success, log_warning


DEFAULT_URL = "https://docs.langchain.com/oss/python/langchain/overview"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def crawl(start_url: str, max_pages: int, max_depth: int) -> list[Document]:
    """Follow same-host links and extract the main text of each HTML page."""
    start_url = urldefrag(start_url)[0].rstrip("/")
    start = urlparse(start_url)
    session = requests.Session()
    session.headers["User-Agent"] = "documentation-helper/1.0 (documentation indexing)"
    pending = deque([(start_url, 0)])
    seen = {start_url}
    documents = []

    while pending and len(documents) < max_pages:
        url, depth = pending.popleft()
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as exc:
            log_warning(f"Skipping {url}: {exc}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            continue

        if depth < max_depth:
            for link in main.find_all("a", href=True):
                candidate = urldefrag(urljoin(url, link["href"]))[0].rstrip("/")
                parsed = urlparse(candidate)
                if (parsed.scheme in ("http", "https") and parsed.netloc == start.netloc
                        and parsed.path.startswith("/oss/python/") and candidate not in seen):
                    seen.add(candidate)
                    pending.append((candidate, depth + 1))

        for element in main.select("nav, footer, header, script, style, noscript"):
            element.decompose()
        content = main.get_text(" ", strip=True)
        if content:
            documents.append(Document(page_content=content, metadata={"source": url}))
            log_info(f"Crawled {url}")

    return documents


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=2)
    args = parser.parse_args()
    if args.max_pages < 1 or args.max_depth < 0:
        parser.error("--max-pages must be positive and --max-depth nonnegative")

    log_header("DOCUMENTATION INGESTION")
    documents = crawl(args.url, args.max_pages, args.max_depth)
    if not documents:
        raise RuntimeError("No documentation pages were fetched; the index was not changed")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(documents)
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name="documentation",
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )
    ids = [sha256(f"{doc.metadata['source']}:{i}:{doc.page_content}".encode()).hexdigest()
           for i, doc in enumerate(chunks)]
    for offset in range(0, len(chunks), 50):
        vectorstore.add_documents(chunks[offset:offset + 50], ids=ids[offset:offset + 50])
    stale_ids = set(vectorstore.get(include=[])["ids"]) - set(ids)
    if stale_ids:
        vectorstore.delete(ids=list(stale_ids))
    log_success(f"Indexed {len(chunks)} chunks from {len(documents)} pages")


if __name__ == "__main__":
    main()
