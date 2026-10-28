"""Compare a direct Ollama call with two local retrieval approaches."""

import os
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "mediumblog1.txt"

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def create_retriever():
    """Embed the source document with Ollama and index it for this run."""
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Source document not found: {SOURCE_FILE}")

    content = SOURCE_FILE.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"Source document contains no text: {SOURCE_FILE}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = splitter.split_documents(
        [Document(page_content=content, metadata={"source": str(SOURCE_FILE)})]
    )
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    )
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str, retriever, llm):
    """Retrieve, format, and answer the question one step at a time."""
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel(retriever, llm):
    """Create an LCEL chain invoked with {"question": "..."}."""
    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )


def main():
    load_dotenv(PROJECT_DIR / ".env")
    print("Initializing local Ollama components...")
    llm = ChatOllama(model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b"))
    retriever = create_retriever()

    query = "What is a vector database in machine learning?"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    print(llm.invoke([HumanMessage(content=query)]).content)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    print(retrieval_chain_without_lcel(query, retriever, llm))

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)
    chain = create_retrieval_chain_with_lcel(retriever, llm)
    print(chain.invoke({"question": query}))


if __name__ == "__main__":
    main()
