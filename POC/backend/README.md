# Backend

The agent graph behind HTTP, with Clerk on the door.

```bash
uv run uvicorn backend.main:app --port 8000
```

| route | auth | |
| --- | --- | --- |
| `POST /ask` | Clerk | a question in, a cited answer out |
| `GET /graph` | Clerk | the flow as mermaid, from the compiled graph |
| `GET /me` | Clerk | who Clerk says you are — the cheapest token check |
| `GET /health` | open | readiness, and what is loaded |

`/health` and the OpenAPI schema are the only open routes. That list is short on
purpose: adding a route means deciding which side of it the route belongs on.

## Configuration

```bash
CLERK_SECRET_KEY=sk_...                     # required; the server will not start without it
CLERK_JWT_KEY=-----BEGIN PUBLIC KEY-----... # optional, and worth setting
CLERK_AUTHORIZED_PARTIES=http://localhost:3000
CORS_ORIGINS=http://localhost:3000
```

`CLERK_SECRET_KEY` is required and there is deliberately **no development
bypass**. An `if DEBUG: return a fake user` branch is exactly the kind of thing
that survives into production, and the cost of not having one is setting an
environment variable.

`CLERK_JWT_KEY` makes verification networkless — the session token is checked
against a key you already hold, so there is no round trip to Clerk's JWKS
endpoint per request and no dependency on their API being reachable to serve
one.

`CLERK_AUTHORIZED_PARTIES` is the allowlist of front ends whose tokens this
server accepts. A Clerk session token is bound to the origin that obtained it,
and this is what stops one that leaked elsewhere from being replayed here.
Unset, that check is skipped.

## Two things about this server

**It owns the corpus.** The Qdrant store is embedded, which means
single-process: this server holds an exclusive lock on it. Nothing else can run
while it does — no notebook kernel, no `db_populate.py`, no second worker.

```bash
uv run uvicorn backend.main:app --workers 1   # the only correct number
```

A second worker fails to start with "the embedded Qdrant store is locked by
another process". The fix, when you need more than one, is a Qdrant server and
`QDRANT_URL` — not more workers.

**The local models are one resource.** BGE-M3, the cross-encoder and that store
are shared and stateful, so a semaphore admits `LOCAL_CONCURRENCY` (default 1)
questions to that stretch of the flow. The model calls that follow it are
ordinary async I/O and overlap freely — which is where nearly all of the
wall-clock goes, so serialising the local part costs much less throughput than
it sounds like.

Two blocking calls were moved off the event loop to make that true: retrieval
(`asyncio.to_thread` around `_retrieve`, measured at 10-18 s) and the
cross-encoder. Awaited directly, either one stops the server answering anything
else for its whole duration — which a CLI never notices.

Past `MAX_IN_FLIGHT` questions the server returns 503 rather than queueing.
An arrival that waits behind the local lock spends its entire timeout not being
served and then fails anyway; saying so immediately is more useful.

## Startup

Building the pipeline loads BGE-M3, opens Qdrant and compiles the NeMo rails —
tens of seconds. That happens once, in the app's lifespan, so the first question
is not the one that pays for it.

If it fails, the server still starts and `/health` reports `unavailable` with
the reason, while `/ask` returns 503. A server that can say *why* it is not
working beats one that exited before anything could ask it.

Clerk is the exception: a missing secret key stops startup outright, because a
server that cannot authenticate should not be listening.

## Responses

A refusal is a **200**, not a 4xx. Both kinds — the guardrail blocking a
question, and the scope floor finding nothing in the corpus that bears on it —
are what the system decided rather than faults in the request, and a client
should render them as answers. `refused` says which:

```json
{"question": "...", "answer": "Nothing in the indexed papers bears on that ...",
 "refused": "below scope floor (0.001)", "citations": [], "trace": {...}}
```

Errors are what you would expect: 401 unauthenticated, 503 not ready or too
busy, 504 timeout, 502 a provider failed.

The response carries citations rather than the graph's own dict. That dict
holds the full payload of every candidate — the whole text of twenty-odd rows,
several times over. `include_text: true` adds the text of the cited passages
only.

`trace` is the run's working: what the router decided, the scope score, the
facts extracted, the verifier's verdicts, and per-stage timings. The pre-repair
`draft` is deliberately not included — returning both invites a client to
display the one the verifier rejected.

## Checking it works

```bash
curl localhost:8000/health
```

```bash
curl -H "Authorization: Bearer $CLERK_TOKEN" localhost:8000/me
```

```bash
curl -X POST localhost:8000/ask -H "Authorization: Bearer $CLERK_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "how does detection probability vary with range"}'
```

A browser session token comes from `await window.Clerk.session.getToken()` in
the front end's console.

Verified without a Clerk account: the server refuses to start with no secret
key; `/health` reports the corpus open (80,365 points, 80,221 BM25 rows); and
`/me`, `/graph` and `/ask` all return 401 unauthenticated, including with a
malformed bearer token, with a generic message that does not say why.
