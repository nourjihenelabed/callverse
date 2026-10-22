"""Checks the rules search in callverse/knowledge.py.

No Ollama needed: a simple word-counting embedding replaces the real model.
The Chroma test runs only if Chroma is installed (pip install -e ".[rag]").
"""

import importlib.util
import math
import re
import unicodedata

import pytest

from callverse import knowledge
from callverse.config import PROJECT_ROOT, load_settings
from callverse.state import ComplaintCategory


class WordCountEmbeddings:
    """A fake embedding: counts words into 512 slots. Same interface as the real one."""

    def _vector(self, text: str) -> list[float]:
        text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
        vector = [0.0] * 512
        for word in re.findall(r"[a-z0-9]{3,}", text):
            vector[sum(ord(ch) * (i + 1) for i, ch in enumerate(word)) % 512] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


POLICY_FILES = ["agent_guidelines.md", "catalogue.md", "compensation_grid.md",
                "complaint_procedures.md", "contracts.md", "faq.md", "procedures.md"]


def test_every_policy_file_is_split_into_sections():
    chunks = knowledge.split_policies()
    assert sorted({c["source"] for c in chunks}) == POLICY_FILES     # README.md is skipped
    assert all(len(c["text"]) <= knowledge.MAX_CHUNK_CHARS + 200 for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)              # ids are unique
    faq = [c for c in chunks if c["source"] == "faq.md"]
    assert any("code RIO" in c["section"] for c in faq)                # each FAQ question is its own chunk


def test_complaint_categories_match_state_py():
    text = (PROJECT_ROOT / "data" / "policies" / "complaint_procedures.md").read_text(encoding="utf-8")
    for category in ComplaintCategory.__args__:
        assert f"| {category} |" in text, f"{category} is missing from complaint_procedures.md"


@pytest.mark.parametrize("question, expected_source", [
    ("Comment obtenir mon code RIO pour changer d'opérateur ?", "faq.md"),
    ("Quel est le plafond automatique de compensation pour une gravité 3 ?", "compensation_grid.md"),
    ("Quand faut-il transmettre une réclamation à un superviseur ?", "complaint_procedures.md"),
    ("frais de résiliation engagement 24 mois après le 13e mois", "contracts.md"),
])
def test_local_search_finds_the_right_document(tmp_path, monkeypatch, question, expected_source):
    monkeypatch.setitem(load_settings()["rag"], "store", "local")
    fake = WordCountEmbeddings()
    knowledge.build_index(embeddings=fake, index_dir=tmp_path)
    hits = knowledge.search_policies(question, k=3, embeddings=fake, index_dir=tmp_path)
    assert len(hits) == 3
    assert hits[0]["score"] >= hits[-1]["score"]
    assert expected_source in [h["source"] for h in hits]


def test_search_before_build_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setitem(load_settings()["rag"], "store", "local")
    with pytest.raises(RuntimeError, match="build_index"):
        knowledge.search_policies("bonjour", embeddings=WordCountEmbeddings(), index_dir=tmp_path)


@pytest.mark.skipif(importlib.util.find_spec("chromadb") is None, reason="Chroma not installed")
def test_chroma_gives_the_same_first_answer_as_local(tmp_path, monkeypatch):
    fake = WordCountEmbeddings()
    question = "Combien coûte une nouvelle carte SIM après une perte ?"

    monkeypatch.setitem(load_settings()["rag"], "store", "local")
    knowledge.build_index(embeddings=fake, index_dir=tmp_path / "local")
    local = knowledge.search_policies(question, k=3, embeddings=fake, index_dir=tmp_path / "local")

    monkeypatch.setitem(load_settings()["rag"], "store", "chroma")
    knowledge.build_index(embeddings=fake, index_dir=tmp_path / "chroma")
    chroma = knowledge.search_policies(question, k=3, embeddings=fake, index_dir=tmp_path / "chroma")

    assert chroma[0]["section"] == local[0]["section"]
    assert abs(chroma[0]["score"] - local[0]["score"]) < 0.01
