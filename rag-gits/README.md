# Document ingestion and retrieval

Loads `mediumblog1.txt`, splits it into chunks of at most 1,000 characters,
generates embeddings with Pinecone's hosted `llama-text-embed-v2` model, and
stores the vectors in a Pinecone index.

Install the dependencies from the project directory:

```powershell
uv sync --locked
```

Set these values in `.env` beside `main.py` (or in the environment):

```dotenv
PINECONE_API_KEY=your-pinecone-api-key
INDEX_NAME=my-index
PINECONE_NAMESPACE=rag-gits-cloud
```

`my-index` is an empty 1024-dimensional Pinecone index in this project. The
`rag-gits-cloud` namespace keeps these vectors separate from other uses of the
index. If you use another Pinecone project, choose a 1024-dimensional index and
set its name here. Keep the same namespace for ingestion and retrieval.

Run the ingestion:

```powershell
uv run main.py
```

The document and `.env` paths are resolved relative to `main.py`, so launching
the script from another working directory also works. Ingestion sends document
chunks to Pinecone's hosted embedding model and writes their vectors to the
index.

## Query locally with Ollama

`main2.py` runs independently of the Pinecone ingestion above. It reads
`mediumblog1.txt`, embeds its chunks with Ollama's `embeddinggemma` model,
stores them in memory, and compares a direct Ollama answer with two retrieval
chains using `qwen3:1.7b`. Install Ollama and pull the models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:1.7b
```

Then run:

```powershell
uv run main2.py
```

Set `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL` in `.env` to use other installed
models. This local example does not require Pinecone credentials. Its in-memory
index is rebuilt each time the script runs.
