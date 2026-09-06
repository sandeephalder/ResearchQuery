"""HTTP in front of the agent graph, with Clerk on the door.

    uv run uvicorn backend.main:app --reload

Nothing is imported here: `main` builds the whole retrieval stack at startup,
which is a BGE-M3 load and an exclusive lock on the embedded Qdrant store, and
that should happen when a server starts rather than when a module is imported.
"""
