"""Clerk on the door.

Every route that touches the corpus depends on `require_user`, which verifies
the caller's Clerk session token and hands back who they are. A route without
that dependency is open, so there are exactly two — `/health` and the OpenAPI
schema — and both are listed in the module docstring of `main.py` for that
reason.

Verification is Clerk's own `authenticate_request`, not a hand-rolled JWT
check. Session tokens carry more than a signature: an authorized-party claim
that binds the token to the front end that obtained it, an expiry with a clock
skew allowance, and a key that rotates. Those are the parts that are easy to
verify slightly wrong and impossible to notice.
"""

import dataclasses

from clerk_backend_api import Clerk
from clerk_backend_api.security import AuthenticateRequestOptions, AuthStatus
from fastapi import HTTPException, Request, status

from .constants import CLERK_AUTHORIZED_PARTIES, CLERK_JWT_KEY, CLERK_SECRET_KEY


@dataclasses.dataclass(frozen=True)
class Principal:
    """Who is asking. Built only from claims Clerk has verified."""

    user_id: str
    session_id: str | None = None
    org_id: str | None = None
    claims: dict = dataclasses.field(default_factory=dict, repr=False)


class AuthNotConfigured(RuntimeError):
    """No Clerk secret. Raised at startup, so it cannot surface per request."""


def check_configuration():
    """Called from the app's lifespan, so a misconfigured server fails to start.

    The alternative — discovering it on the first request — is a server that
    looks healthy and rejects everybody, or worse, one where a future edit adds
    a fallback that lets everybody through.
    """
    if not CLERK_SECRET_KEY:
        raise AuthNotConfigured(
            "CLERK_SECRET_KEY is not set. Copy it from the Clerk dashboard "
            "(API keys -> Secret keys) into POC/.env. Setting CLERK_JWT_KEY as "
            "well makes verification networkless.")


_clerk = None


def clerk():
    """The SDK client, built once. Holds the JWKS cache between requests."""
    global _clerk
    if _clerk is None:
        check_configuration()
        _clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)
    return _clerk


async def require_user(request: Request) -> Principal:
    """FastAPI dependency: a verified caller, or 401.

    Starlette's `Request` satisfies Clerk's `Requestish` protocol — it wants a
    `.headers` mapping and nothing more — so the request goes straight in, and
    the SDK finds the token in the Authorization header or the `__session`
    cookie itself.
    """
    options = AuthenticateRequestOptions(
        secret_key=CLERK_SECRET_KEY,
        jwt_key=CLERK_JWT_KEY,
        authorized_parties=CLERK_AUTHORIZED_PARTIES or None,
    )
    try:
        state = await clerk().authenticate_request_async(request, options)
    except Exception as error:                  # noqa: BLE001 — the SDK raises broadly
        # Never echo the exception to the caller: it can name the reason a token
        # failed, which tells someone probing exactly what to change.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"}) from error

    if state.status != AuthStatus.SIGNED_IN or not state.payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})

    claims = state.payload
    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})

    return Principal(user_id=subject, session_id=claims.get("sid"),
                     org_id=claims.get("org_id"), claims=claims)
