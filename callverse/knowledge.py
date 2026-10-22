"""The one place where workers search the company rules (RAG).

Every worker calls search_policies(question). Nobody talks to Chroma directly,
so we can move to pgvector later by changing only this file.

Shared by everyone. Ask the team before changing it.
"""
