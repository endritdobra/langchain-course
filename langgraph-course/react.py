import os

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()


@tool
def triple(num: float) -> float:
    """Multiply a number by three.

    :param num: the number to triple
    :return: the tripled number
    """

    return float(num) * 3


@tool
def search(query: str) -> str:
    """Search the web with DuckDuckGo and return the top results.

    :param query: the search query
    :return: a formatted list of results
    """

    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for r in results
    )


tools = [search, triple]

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:cloud"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
).bind_tools(tools)
