import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = '''
        Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman, industrialist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    '''
    summary_template = '''
        Given the following information about the person:
        {information}

        Create:
        1. A short summary
        2. two interesting facts about them
    '''

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template,
    )

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma4:latest"),
        temperature=0,
        reasoning=False,
    )
    chain = summary_prompt_template | llm
    print(f"Generating summary with {llm.model}...", flush=True)
    for chunk in chain.stream({"information": information}):
        print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    main()
