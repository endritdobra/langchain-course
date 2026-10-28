# LangChain Documentation Helper

This version crawls public LangChain documentation directly, stores a local
Chroma index, and answers questions with an Ollama Cloud chat model. Crawling
does not need a Tavily account. Embeddings run locally with FastEmbed because
Ollama currently has no cloud embedding models.

## Setup

1. Install Python 3.11 and Pipenv, then run `pipenv install` in this folder.
2. Create an [Ollama API key](https://ollama.com/settings/keys). Copy
   `.env.example` to `.env` and set `OLLAMA_API_KEY` there. Never commit it.
3. Run `pipenv run python ingestion.py --max-pages 50` to build the index.
   The first run downloads the free local embedding model.
4. Ask a question with
   `pipenv run python main.py "How do I create a LangChain agent?"`.

`OLLAMA_MODEL` defaults to `gemma4:31b`. Set it to a model name from
[Ollama's cloud model list](https://ollama.com/search?c=cloud) if needed.
Direct cloud API names omit the `:cloud` suffix used by the Ollama app.

The crawler starts at the current LangChain Python overview and follows
same-site Python documentation links. Use `--url`, `--max-pages`, and
`--max-depth` to change its scope. The two Tavily tutorial notebooks are
original reference material and are not part of this pipeline.
