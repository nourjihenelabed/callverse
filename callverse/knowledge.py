"""The one place where workers search the company rules (RAG).

Every worker calls search_policies(question). Nobody talks to Chroma directly,
so we can move to pgvector later by changing only this file.

Shared by everyone. Ask the team before changing it.

Usage in a worker:
    from callverse.knowledge import search_policies
    for hit in search_policies("Combien coûte la Box Fibre ?"):
        print(hit["source"], hit["section"], hit["text"])

Before the first search, build the index once:  python scripts/build_index.py
Build it again whenever a file in data/policies changes.

Where the index lives is set by `rag.store` in settings.yaml:
- "chroma": a Chroma database in data/index (needs: pip install -e ".[rag]")
- "local":  a plain JSON file in data/index, no extra package. Use it if Chroma
            cannot be installed on your computer. Results are the same.
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from callverse.config import PROJECT_ROOT, load_settings

COLLECTION = "policies"
MAX_CHUNK_CHARS = 1500          # longer sections are cut at paragraph breaks


# ---------------------------------------------------------------- reading the documents

def _paths() -> tuple[Path, Path]:
    paths = load_settings()["paths"]
    return PROJECT_ROOT / paths["policies"], PROJECT_ROOT / paths["index"]


def split_policies(folder: Path | None = None) -> list[dict]:
    """Cuts every Markdown file in data/policies into sections.

    Each chunk is {"id", "source", "section", "text"}. The text starts with the
    document and section titles so a chunk still makes sense on its own.
    """
    folder = folder or _paths()[0]
    chunks = []
    for path in sorted(folder.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        title, headings, lines = path.stem, {}, []

        def flush():
            body = "\n".join(lines).strip()
            lines.clear()
            if not body:
                return
            section = " > ".join(headings[level] for level in sorted(headings)) or title
            header = title if section == title else f"{title} — {section}"
            for n, part in enumerate(_cut(body)):
                chunks.append({
                    "id": f"{path.stem}#{len(chunks)}",
                    "source": path.name,
                    "section": section,
                    "text": f"{header}\n{part}" if n == 0 else f"{header} (suite)\n{part}",
                })

        for line in path.read_text(encoding="utf-8").splitlines():
            heading = re.match(r"^(#{1,3})\s+(.*)", line)
            if heading:
                flush()
                level, name = len(heading.group(1)), heading.group(2).strip()
                if level == 1:
                    title, headings = name, {}
                else:
                    headings = {k: v for k, v in headings.items() if k < level}
                    headings[level] = name
            else:
                lines.append(line)
        flush()
    return chunks


def _cut(body: str) -> list[str]:
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    parts, current = [], ""
    for paragraph in body.split("\n\n"):
        if current and len(current) + len(paragraph) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


# ---------------------------------------------------------------- embeddings

def _embeddings(embeddings=None):
    if embeddings is not None:
        return embeddings
    from callverse.llm import get_embeddings
    return get_embeddings()


def _prefixes() -> tuple[str, str]:
    """nomic-embed-text works better when documents and questions are labeled."""
    if "nomic" in str(load_settings()["embeddings"]["model"]):
        return "search_document: ", "search_query: "
    return "", ""


def _store() -> str:
    store = load_settings()["rag"]["store"]
    if store not in ("chroma", "local"):
        raise ValueError(f"rag.store must be 'chroma' or 'local', not {store!r}")
    return store


# ---------------------------------------------------------------- building the index

def build_index(embeddings=None, folder: Path | None = None, index_dir: Path | None = None) -> int:
    """Reads data/policies, embeds every chunk and saves the index. Returns the number of chunks."""
    default_folder, default_index = _paths()
    folder, index_dir = folder or default_folder, index_dir or default_index
    chunks = split_policies(folder)
    doc_prefix, _ = _prefixes()
    vectors = _embeddings(embeddings).embed_documents([doc_prefix + c["text"] for c in chunks])
    index_dir.mkdir(parents=True, exist_ok=True)

    if _store() == "chroma":
        client = _chroma_client(index_dir)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass                                    # first build: nothing to delete
        collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=vectors,
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
        )
    else:
        data = [{**c, "vector": v} for c, v in zip(chunks, vectors)]
        (index_dir / "policies.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    _load_local.cache_clear()
    _chroma_client.cache_clear()
    return len(chunks)


# ---------------------------------------------------------------- searching

def search_policies(question: str, k: int | None = None, embeddings=None, index_dir: Path | None = None) -> list[dict]:
    """Finds the k sections of the company rules closest to the question.

    Returns a list of {"text", "source", "section", "score"}, best first.
    score goes from 0 (unrelated) to 1 (same meaning).
    """
    k = k or load_settings()["rag"].get("top_k", 4)
    index_dir = index_dir or _paths()[1]
    _, query_prefix = _prefixes()
    query = _embeddings(embeddings).embed_query(query_prefix + question)

    if _store() == "chroma":
        try:
            collection = _chroma_client(index_dir).get_collection(COLLECTION)
        except Exception as error:
            raise RuntimeError("No rules index yet. Run: python scripts/build_index.py") from error
        found = collection.query(query_embeddings=[query], n_results=k)
        return [
            {"text": text, "source": meta["source"], "section": meta["section"], "score": round(1 - distance, 3)}
            for text, meta, distance in zip(found["documents"][0], found["metadatas"][0], found["distances"][0])
        ]

    scored = [(_cosine(query, c["vector"]), c) for c in _load_local(index_dir)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"text": c["text"], "source": c["source"], "section": c["section"], "score": round(score, 3)}
        for score, c in scored[:k]
    ]


@lru_cache
def _chroma_client(index_dir: Path):
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            'Chroma is not installed. Run: pip install -e ".[rag]"  '
            "(or set rag.store to 'local' in settings.yaml if it cannot be installed)"
        ) from error
    return chromadb.PersistentClient(path=str(index_dir / "chroma"))


@lru_cache
def _load_local(index_dir: Path) -> list[dict]:
    path = index_dir / "policies.json"
    if not path.exists():
        raise RuntimeError("No rules index yet. Run: python scripts/build_index.py")
    return json.loads(path.read_text(encoding="utf-8"))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
