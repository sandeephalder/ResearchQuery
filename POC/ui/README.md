# UI

The flow behind a browser. Next.js App Router, Clerk for sign-in, and a typed
client over the FastAPI app in [`../backend/`](../backend/).

```bash
npm install
npm run dev                          # http://localhost:3000
```

It needs the API running beside it:

```bash
cd .. && uv run uvicorn backend.main:app --port 8000
```

`.env.local` holds the publishable key and the API origin; it is generated from
`POC/.env`, and `.env.local.example` documents both fields.

## What it shows

One page, rendering the three things a run produces.

**The answer**, with its `[n]` markers turned into links into the citation list.
A refusal — the input rail blocked the question, or nothing retrieved cleared
the scope floor — is rendered as an answer in a note, not as an error, because
that is what it is. The backend returns it as `200` with `refused` set for the
same reason.

**The citations**, numbered as the answer cites them. That numbering is the
contract running through the whole flow: the reranker orders by it, the
extractor attributes to it, the generator cites it, the verifier checks against
it. `n` arrives already numbered from the backend; nothing is re-derived here.

**The trace**, folded away by default — what the router decided and why, the
scope score, the extracted facts, per-stage timings, and any problem that
survived the repair. A run showing its own working is the point, but it is not
what someone reading an answer wants first.

## A `[n]` is only linked when passage n exists

`linkCitations` checks the number against the citation count before making it a
link, and that guard is not cosmetic.

Bracketed numerals occur in the source passages themselves — bibliography
references like `[55]` — and the generator has been observed copying one into an
answer, where it is indistinguishable from a citation marker. The DeepEval suite
catches it as a citation failure ([`../tests/`](../tests/)). Linking it would
invent a source; leaving it as plain text shows it for what it is.

## Auth

Clerk, and the token travels in the `Authorization` header rather than a cookie.

That is forced rather than chosen: the UI is served from `:3000` and the API from
`:8000`, so Clerk's `__session` cookie is not sent cross-origin. `getToken()`
from `useAuth()` supplies it, and [`lib/api.ts`](lib/api.ts) attaches it to every
call.

**`middleware.ts` protects nothing.** The backend is what enforces access —
every route but `/health` sits behind `require_user`, verified server-side — and
a UI that hides a page it cannot actually protect is theatre. What the middleware
provides is the session in the browser, so the app can offer a sign-in button
instead of a failed request.

Two backend settings matter here:

| | |
| --- | --- |
| `CORS_ORIGINS` | must include this origin; defaults to `http://localhost:3000` |
| `CLERK_AUTHORIZED_PARTIES` | the allowlist of front ends whose tokens the API accepts. **Unset skips the check** — worth setting to `http://localhost:3000` even in development, since it is what stops a token that leaked elsewhere from being replayed |

## `/health` is the one open route

Which is why the badge in the header works before sign-in, and why it is worth
having: "is the API even running" is the first question when a call fails, and it
should not itself need a token to answer.

Startup is genuinely slow — BGE-M3, the cross-encoder, the NeMo rails, and an
exclusive lock on the embedded Qdrant store — so `starting` for the first
half-minute is normal, and the badge says so rather than reporting an error.

A failed `fetch` names the two things it is almost always caused by — the server
is not running, or its `CORS_ORIGINS` does not include this origin — because the
browser reports both identically.

## Types are hand-written

[`lib/types.ts`](lib/types.ts) mirrors `backend/schemas.py` by hand. There are
four shapes; a generator would be a build step to maintain for a contract that
small. If a field is added there, it is added here.

## What is not here

- **No streaming.** `/ask` returns once, after 30-60 seconds. Streaming would
  need the backend to stream, and the flow's stages are not incremental — the
  verifier can send a whole draft back for a repair.
- **No history.** Every question gets its own `thread_id` server-side, and
  nothing is stored client-side.
- **No `/graph` page.** The route exists and `lib/api.ts` calls it; rendering
  mermaid in the browser needs a renderer this does not yet pull in.
