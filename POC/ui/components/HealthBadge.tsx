"use client";

import { useEffect, useState } from "react";

import { health } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

/**
 * Is the server up, and what has it loaded?
 *
 * Uses `/health`, the one route without `require_user`, so it answers before
 * sign-in — which is the point: "is the API even running" is the first question
 * when a call fails, and it should not itself need a token to answer.
 *
 * Startup is slow by design (BGE-M3, the cross-encoder, the NeMo rails, and an
 * exclusive lock on the embedded Qdrant store), so a `starting` status is
 * normal for the first half-minute and is shown as such rather than as an error.
 */
export function HealthBadge() {
  const [state, setState] = useState<HealthResponse | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;

    const poll = async () => {
      try {
        const next = await health();
        if (!live) return;
        setState(next);
        setFailed(false);
      } catch {
        if (live) setFailed(true);
      }
    };

    poll();
    // Slow, because nothing here changes quickly and a badge should not be a
    // meaningful share of the server's traffic.
    const timer = setInterval(poll, 15_000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  if (failed) {
    return (
      <span className="badge" title="The API did not respond">
        <span className="dot bad" /> api unreachable
      </span>
    );
  }
  if (!state) {
    return (
      <span className="badge">
        <span className="dot" /> checking
      </span>
    );
  }

  const ready = state.status === "ok";
  return (
    <span
      className="badge"
      title={
        ready
          ? `${state.corpus_points?.toLocaleString() ?? "?"} points · ` +
            `${state.bm25_rows?.toLocaleString() ?? "?"} bm25 rows · ` +
            `reranker ${state.reranker ?? "?"} · ${state.agent_provider ?? "?"}`
          : (state.detail ?? state.status)
      }
    >
      <span className={`dot ${ready ? "ok" : ""}`} />
      {ready
        ? `${state.corpus_points?.toLocaleString() ?? "corpus"} points`
        : state.status}
      {state.in_flight > 0 ? ` · ${state.in_flight} in flight` : ""}
    </span>
  );
}
