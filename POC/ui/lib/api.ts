/**
 * The backend, from the browser, with a Clerk token attached.
 *
 * Every route but `/health` is behind `require_user`, and the token travels in
 * the `Authorization` header rather than a cookie. That is not a preference:
 * the UI is served from :3000 and the API from :8000, so Clerk's `__session`
 * cookie is not sent cross-origin. `getToken()` from `useAuth()` is what the
 * caller passes in here.
 *
 * The API base is `NEXT_PUBLIC_API_URL`, and it must be an origin the backend's
 * `CORS_ORIGINS` allows — it defaults to `http://localhost:3000`, which is
 * where `npm run dev` serves.
 */

import type { AskRequest, AskResponse, HealthResponse, Me } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** A failed call, carrying the status so a caller can tell 503 from 401. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /**
   * The backend refuses rather than queues when it is busy, and times out at
   * five minutes. Both are worth retrying; a 401 or a 422 is not.
   */
  get retryable(): boolean {
    return this.status === 503 || this.status === 504;
  }
}

type TokenGetter = () => Promise<string | null>;

async function call<T>(
  path: string,
  getToken: TokenGetter | null,
  init: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init.headers as Record<string, string>) ?? {}),
  };

  if (getToken) {
    const token = await getToken();
    if (!token) {
      throw new ApiError("Not signed in", 401);
    }
    headers.Authorization = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch (cause) {
    // A network-level failure here is almost always one of two things, and
    // both are worth naming: the server is not running, or its CORS_ORIGINS
    // does not include this origin. The browser reports them identically.
    throw new ApiError(
      `Cannot reach the API at ${API_URL}. Is uvicorn running, and does its ` +
        `CORS_ORIGINS include ${typeof window === "undefined" ? "this origin" : window.location.origin}?`,
      0,
    );
  }

  if (!response.ok) {
    throw new ApiError(await detail(response), response.status);
  }
  return (await response.json()) as T;
}

/** FastAPI puts the message in `detail`; a validation error puts a list there. */
async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((e: { loc?: string[]; msg?: string }) =>
          `${e.loc?.slice(1).join(".") ?? ""} ${e.msg ?? ""}`.trim(),
        )
        .join("; ");
    }
  } catch {
    /* fall through to the status text */
  }
  return response.statusText || `HTTP ${response.status}`;
}

export function ask(getToken: TokenGetter, request: AskRequest): Promise<AskResponse> {
  return call<AskResponse>("/ask", getToken, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function me(getToken: TokenGetter): Promise<Me> {
  return call<Me>("/me", getToken);
}

export function graph(getToken: TokenGetter): Promise<{ mermaid: string }> {
  return call<{ mermaid: string }>("/graph", getToken);
}

/** The one open route — no token, so it works before sign-in. */
export function health(): Promise<HealthResponse> {
  return call<HealthResponse>("/health", null);
}
