"""The one place where the AI model is set up. Every worker gets its model from here.

Shared by everyone. Ask the team before changing it.

Usage in a worker:
    from callverse.llm import get_llm
    llm = get_llm()
    answer = llm.invoke("Hello").content

The model runs in Ollama. By default Ollama runs on your own computer.
To use a teammate's computer instead, put OLLAMA_BASE_URL in your .env file.
"""

import os

from callverse.config import load_settings

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _model_name(section: str) -> str:
    name = load_settings()[section]["model"]
    if not name or name == "TO_DECIDE":
        raise RuntimeError(
            f"No model chosen yet: set '{section}: model' in settings.yaml "
            "(the team picks it together)."
        )
    return name


def _ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)


def get_llm(temperature: float | None = None):
    """The chat model every worker uses. A worker may ask for a different temperature."""
    from langchain_ollama import ChatOllama

    settings = load_settings()["llm"]
    return ChatOllama(
        model=_model_name("llm"),
        temperature=settings["temperature"] if temperature is None else temperature,
        base_url=_ollama_url(),
    )


def get_embeddings():
    """The model that turns text into vectors, used only by knowledge.py and build_index.py."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=_model_name("embeddings"), base_url=_ollama_url())
