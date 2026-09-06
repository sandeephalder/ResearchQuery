"""Server configuration. Everything the deployment decides, and nothing else.

The agent flow's own knobs stay in `agents/constants.py` — this file is about
who may call it, from where, and how many at a time.
"""

import os

from dotenv import load_dotenv

# Loaded here rather than relied upon. `llm/constants.py` also calls this, and
# importing it first is what made these variables resolve — an import-order
# dependency that would break auth silently the day someone sorted the imports
# in main.py. A module that reads the environment should be the module that
# ensures it is populated. load_dotenv() walks up from the working directory,
# so this resolves whether uvicorn is started from POC/ or above it, and it
# will not overwrite a variable the environment already set.
load_dotenv()

# --------------------------------------------------------------------------- #
# Clerk
# --------------------------------------------------------------------------- #
# The secret key is required. There is deliberately no development bypass: an
# "if DEBUG: return a fake user" branch is the kind of thing that survives into
# production, and the cost of not having one is setting an environment variable.
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")

# Optional, and worth setting. With the JWT verification key present, Clerk's
# SDK verifies the session token from the signature alone — no round trip to
# Clerk's JWKS endpoint on every request, and no dependency on their API being
# reachable to serve a request. Copy it from the Clerk dashboard.
CLERK_JWT_KEY = os.getenv("CLERK_JWT_KEY")

# Origins allowed to present a token from your Clerk instance. A session token
# is bound to the front end that obtained it, and this is what stops one that
# leaked to another site from being replayed here. Comma-separated.
CLERK_AUTHORIZED_PARTIES = [
    party.strip() for party in os.getenv("CLERK_AUTHORIZED_PARTIES", "").split(",")
    if party.strip()
]

# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
CORS_ORIGINS = [origin.strip() for origin in
                os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
                if origin.strip()]

# A question is a question. The retrieval side embeds whatever it is given, and
# a megabyte of text would be embedded, BM25-tokenised and sent to a reranker
# before anything noticed.
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

# How long one question may take before the client is told it failed. The flow
# is six model calls, some on a home LAN; measured end to end at 90-150 s.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
# How many questions may be inside the *local* models at once — BGE-M3, the
# cross-encoder, and the embedded Qdrant store. One, because that store is
# single-process by design and the two models are a shared, stateful GPU
# context rather than something to fan out over. The API calls that follow are
# not covered by this and overlap freely, which is where the real waiting is.
LOCAL_CONCURRENCY = int(os.getenv("LOCAL_CONCURRENCY", "1"))

# Questions in flight overall. Past this the server says 503 rather than
# queueing arrivals behind a lock until they time out.
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "8"))
