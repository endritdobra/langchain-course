# Ollama agent

This example uses LangChain's [ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama) with [Ollama Cloud](https://docs.ollama.com/cloud).

Create an [Ollama API key](https://ollama.com/settings/keys) and set it in `.env`:

```dotenv
OLLAMA_API_KEY=your_api_key
OLLAMA_MODEL=gemma4:31b
```

Then run the example:

```powershell
uv sync
uv run python main.py
```

Choose a cloud model that supports tool calling. The default is `gemma4:31b`.
If Ollama returns `401 Unauthorized`, replace `OLLAMA_API_KEY` in `.env` with a valid key from your Ollama account.
The `search` tool currently returns a fixed sample weather response.
