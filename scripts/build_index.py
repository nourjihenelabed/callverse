"""Reads the company rules in data/policies and prepares them so the workers can search them.

Run it from the project folder, with Ollama running:
    python scripts/build_index.py

Run it again whenever a file in data/policies changes.
Add a question to try a search right after building:
    python scripts/build_index.py "Combien coûte la Box Fibre ?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callverse.config import load_settings  # noqa: E402
from callverse.knowledge import build_index, search_policies  # noqa: E402


def main() -> None:
    settings = load_settings()
    print(f"Embedding model: {settings['embeddings']['model']}   store: {settings['rag']['store']}")
    count = build_index()
    print(f"Indexed {count} sections from {settings['paths']['policies']}")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nTest search: {question}")
        for hit in search_policies(question):
            print(f"  {hit['score']:.3f}  {hit['source']}  >  {hit['section']}")


if __name__ == "__main__":
    main()
