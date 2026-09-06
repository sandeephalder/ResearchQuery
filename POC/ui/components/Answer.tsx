"use client";

import { Fragment } from "react";

import type { AskResponse } from "@/lib/types";

/**
 * The answer, with its `[n]` markers turned into links to the citation list.
 *
 * A refusal is rendered as an answer, not as an error, because that is what it
 * is: the input rail blocked the question, or nothing retrieved cleared the
 * scope floor. The backend returns it as 200 with `refused` set for the same
 * reason. It is styled as a note so it reads as a decision rather than a fault.
 */
export function Answer({
  result,
  onHover,
}: {
  result: AskResponse;
  onHover: (n: number | null) => void;
}) {
  const count = result.citations.length;

  return (
    <section className="panel">
      <h2>Answer</h2>
      {result.refused && (
        <p className="note warn" style={{ marginTop: 0, marginBottom: 14 }}>
          The flow stopped early: <strong>{result.refused}</strong>. That is a
          decision, not a failure — the question was blocked by a rail, or
          nothing retrieved was relevant enough to answer from.
        </p>
      )}
      <div className="answer">{linkCitations(result.answer ?? "", count, onHover)}</div>
    </section>
  );
}

/**
 * Split on `[n]` and make each marker a link, but only when `n` is a citation
 * that actually exists.
 *
 * That guard is not cosmetic. Bracketed numerals occur in the source passages
 * themselves — bibliography references like `[55]` — and the generator has been
 * observed copying them into an answer, where they are indistinguishable from a
 * citation marker. Linking one would invent a source; leaving it as plain text
 * shows it for what it is.
 */
function linkCitations(text: string, count: number, onHover: (n: number | null) => void) {
  const parts = text.split(/(\[\d+\])/g);

  return parts.map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    const n = match ? Number(match[1]) : null;

    if (n === null || n < 1 || n > count) {
      return <Fragment key={index}>{part}</Fragment>;
    }
    return (
      <a
        key={index}
        href={`#cite-${n}`}
        className="cite"
        onMouseEnter={() => onHover(n)}
        onMouseLeave={() => onHover(null)}
      >
        {part}
      </a>
    );
  });
}
