"use client";

import { useState } from "react";
import { SignedIn, SignedOut, SignInButton, useAuth } from "@clerk/nextjs";

import { Answer } from "@/components/Answer";
import { Citations } from "@/components/Citations";
import { Trace } from "@/components/Trace";
import { ApiError, ask } from "@/lib/api";
import type { AskResponse } from "@/lib/types";

const EXAMPLES = [
  "how does detection probability vary with range",
  "what methods are used for adversarial robustness evaluation",
];

export default function Home() {
  const { getToken } = useAuth();
  const [question, setQuestion] = useState("");
  const [includeText, setIncludeText] = useState(true);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [highlighted, setHighlighted] = useState<number | null>(null);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await ask(getToken, { question: trimmed, include_text: includeText }));
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught : new ApiError(String(caught), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <p className="sub">
        Hybrid retrieval over 80,365 indexed rows — prose, figure descriptions and
        table transcriptions — then a routed, extracted, verified answer that
        cites the passages it used.
      </p>

      <SignedOut>
        <div className="center">
          <p>Sign in to ask the corpus.</p>
          <SignInButton mode="modal">
            <button>Sign in</button>
          </SignInButton>
        </div>
      </SignedOut>

      <SignedIn>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            submit(question);
          }}
        >
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about a method, a result or a figure in the literature…"
            maxLength={2000}
            onKeyDown={(event) => {
              // Enter submits, shift+enter breaks the line — the convention for
              // a box that is a question rather than a document.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit(question);
              }
            }}
          />
          <div className="row">
            <button type="submit" disabled={busy || !question.trim()}>
              {busy ? "Thinking…" : "Ask"}
            </button>
            <label className="toggle">
              <input
                type="checkbox"
                checked={includeText}
                onChange={(event) => setIncludeText(event.target.checked)}
              />
              include passage text
            </label>
            <span className="spacer" style={{ flex: 1 }} />
            {!busy &&
              !result &&
              EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setQuestion(example);
                    submit(example);
                  }}
                >
                  {example.slice(0, 34)}…
                </button>
              ))}
          </div>
        </form>

        {busy && (
          <p className="sub" style={{ marginTop: 24 }}>
            Six model calls across five stages — routing, retrieval, reranking,
            extraction, generation and verification. Typically 30–60 seconds.
          </p>
        )}

        {error && (
          <p className="note error" style={{ marginTop: 24 }}>
            {error.message}
            {error.retryable && " — this one is worth retrying."}
          </p>
        )}

        {result && (
          <>
            <Answer result={result} onHover={setHighlighted} />
            <Citations
              citations={result.citations}
              highlighted={highlighted}
              onHover={setHighlighted}
            />
            <Trace result={result} />
          </>
        )}
      </SignedIn>
    </main>
  );
}
