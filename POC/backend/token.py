"""A session token for testing, minted from the Clerk secret key.

    uv run python -m backend.token                       # print a token
    export CLERK_TOKEN=$(uv run python -m backend.token) # use it in curl

    curl -X POST localhost:8000/ask \\
      -H "Authorization: Bearer $CLERK_TOKEN" \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "how does detection probability vary with range"}'

A Clerk session token is what `/ask` verifies, and the front end normally
produces it — `await window.Clerk.session.getToken()` after a sign-in. With no
front end there is no session, so this makes one against the Backend API and
mints a token from it.

**Development instances only.** It requires the Clerk secret key, which means
anything holding that key can mint a token for any user — which is true of the
secret key generally, and the reason it never reaches a browser. `sk_live_`
keys are refused here rather than trusted to be used carefully.

Tokens are short-lived by design: about a minute. Re-run this rather than
saving one.
"""

import asyncio
import sys

import httpx

from .constants import CLERK_SECRET_KEY

API = "https://api.clerk.com/v1"


async def session_token(user_id=None, reuse=True):
    """A fresh JWT for `user_id`, or for the instance's only user.

    Reuses an active session when there is one — sessions accumulate otherwise,
    and a test run should not leave a trail of them behind.
    """
    if not CLERK_SECRET_KEY:
        raise SystemExit("CLERK_SECRET_KEY is not set — see backend/README.md")
    if CLERK_SECRET_KEY.startswith("sk_live_"):
        raise SystemExit(
            "This is a live Clerk key. Minting tokens for arbitrary users is a "
            "development convenience, not something to point at production — "
            "sign in through the front end and use its token instead.")

    headers = {"Authorization": f"Bearer {CLERK_SECRET_KEY}",
               "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        if user_id is None:
            users = (await client.get(f"{API}/users?limit=2")).json()
            if not users:
                raise SystemExit(
                    "No users in this Clerk instance. Create one in the Clerk "
                    "dashboard (Users -> Create user), or sign up through the "
                    "front end.")
            if len(users) > 1:
                raise SystemExit(
                    "More than one user — pass the id: "
                    "python -m backend.token user_xxx")
            user_id = users[0]["id"]

        session_id = None
        if reuse:
            active = (await client.get(f"{API}/sessions?user_id={user_id}"
                                       f"&status=active&limit=1")).json()
            if isinstance(active, list) and active:
                session_id = active[0]["id"]

        if session_id is None:
            created = await client.post(f"{API}/sessions", json={"user_id": user_id})
            if created.status_code not in (200, 201):
                raise SystemExit(f"could not create a session: HTTP "
                                 f"{created.status_code} — {created.text[:200]}")
            session_id = created.json()["id"]

        minted = await client.post(f"{API}/sessions/{session_id}/tokens", json={})
        if minted.status_code != 200:
            raise SystemExit(f"could not mint a token: HTTP "
                             f"{minted.status_code} — {minted.text[:200]}")
        return minted.json()["jwt"]


if __name__ == "__main__":
    # Only the token on stdout, so `$(...)` captures it and nothing else.
    print(asyncio.run(session_token(sys.argv[1] if len(sys.argv) > 1 else None)))
