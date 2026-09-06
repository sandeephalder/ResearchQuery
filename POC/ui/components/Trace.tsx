"use client";

import { useState } from "react";

import type { AskResponse, Trace as TraceData } from "@/lib/types";

/**
 * What each stage decided, folded away by default.
 *
 * A run showing its own working is the point of the trace, but it is not what
 * someone reading an answer wants first — so it opens on request. The order
 * mirrors the flow: guardrail, route, scope, extract, verify, timings.
 */
export function Trace({ result }: { result: AskResponse }) {
  const [open, setOpen] = useState(false);
  const trace = result.trace as TraceData;
  const timings = trace.timings ?? {};
  const total = Object.values(timings).reduce((sum, value) => sum + value, 0);

  return (
    <section className="panel">
      <div className="row" style={{ margin: 0, justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Trace</h2>
        <button className="ghost" onClick={() => setOpen((was) => !was)}>
          {open ? "Hide" : `Show · ${total.toFixed(1)}s`}
        </button>
      </div>

      {open && (
        <div style={{ marginTop: 14 }}>
          <dl className="kv">
            <dt>guardrail</dt>
            <dd>
              {trace.guardrail_in
                ? trace.guardrail_in.allowed
                  ? `passed (${trace.guardrail_in.rail})`
                  : "BLOCKED"
                : "not run"}
            </dd>

            {trace.route && (
              <>
                <dt>route</dt>
                <dd>
                  kind={trace.route.kind} conf={trace.route.confidence?.toFixed(2)}{" "}
                  {trace.route_effect ? `(${trace.route_effect})` : ""}
                </dd>
              </>
            )}

            {trace.best_score != null && (
              <>
                <dt>scope</dt>
                <dd>best rerank {trace.best_score.toFixed(3)}</dd>
              </>
            )}

            <dt>reranker</dt>
            <dd>{trace.reranker ?? "none"}</dd>

            {trace.repairs ? (
              <>
                <dt>repairs</dt>
                <dd>{trace.repairs}</dd>
              </>
            ) : null}
          </dl>

          {trace.route?.reason && (
            <p className="cite-text" style={{ marginTop: 10 }}>
              {trace.route.reason}
            </p>
          )}

          {trace.facts && trace.facts.length > 0 && (
            <>
              <h2 style={{ marginTop: 18 }}>Extracted facts</h2>
              <ul className="facts">
                {trace.facts.map((fact, index) => (
                  <li key={index}>
                    {fact.passage ? `[${fact.passage}] ` : "[-] "}
                    {fact.fact}
                  </li>
                ))}
              </ul>
            </>
          )}

          {trace.unresolved && trace.unresolved.length > 0 && (
            <p className="note warn" style={{ marginTop: 14 }}>
              {trace.unresolved.length} problem(s) survived the repair:{" "}
              {trace.unresolved.map((p) => p.type).join(", ")}. The draft still
              stands — suppressing a flagged answer turns a partly wrong answer
              into no answer.
            </p>
          )}

          {Object.keys(timings).length > 0 && (
            <>
              <h2 style={{ marginTop: 18 }}>Timings</h2>
              <div className="chips">
                {Object.entries(timings).map(([stage, seconds]) => (
                  <span className="chip" key={stage}>
                    {stage} {seconds}s
                  </span>
                ))}
              </div>
            </>
          )}

          {Object.keys(result.metrics).length > 0 && (
            <>
              <h2 style={{ marginTop: 18 }}>Metrics</h2>
              <div className="chips">
                {Object.entries(result.metrics).map(([name, value]) => (
                  <span className="chip" key={name}>
                    {name} {typeof value === "number" ? value.toFixed(3) : String(value)}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
