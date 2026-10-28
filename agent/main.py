import os
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

class Source(BaseModel):
    """ Schema for a source used by the agent """
    url:str = Field(description="Source URL")

class AgentResponse(BaseModel):
    """ Schema for an agent response wth answer and sources"""
    answer:str = Field(description="Answer")
    sources:List[Source] = Field(default_factory=list, description="List of sources")

load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")
if not api_key:
    raise RuntimeError("Set OLLAMA_API_KEY to use Ollama Cloud")

@tool
def search(query: str) -> str:
    """
    :param query:
    :return:
    """
    print(f'Searching...{query}')
    return "Tokyo weather is sunny"

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:31b"),
    base_url="https://v1.ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    temperature=0,
)
tools = [search]
agent = create_agent(llm, tools, response_format=AgentResponse)

def main():
    print("Hello World")
    result = agent.invoke({
        "messages": [HumanMessage("what is the weather in Tokyo")],
    })
    print(result)


if __name__ == "__main__":
    main()
