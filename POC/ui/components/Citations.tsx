"use client";

import type { Citation } from "@/lib/types";

/**
 * The passages the answer cites, numbered as the answer cites them.
 *
 * The numbering is the contract that runs through the whole flow — the reranker
 * orders by it, the extractor attributes to it, the generator cites it, the
 * verifier checks against it — so [3] here must be [3] there. `n` comes from the
 * backend already numbered; nothing is re-derived in the browser.
 */
export function Citations({
  citations,
  highlighted,
  onHover,
}: {
  citations: Citation[];
  highlighted: number | null;
  onHover: (n: number | null) => void;
}) {
  if (citations.length === 0) return null;

  return (
    <section className="panel">
      <h2>Citations</h2>
      <ol className="citations">
        {citations.map((citation) => (
          <li
            key={citation.n}
            id={`cite-${citation.n}`}
            className={highlighted === citation.n ? "target" : undefined}
            onMouseEnter={() => onHover(citation.n)}
            onMouseLeave={() => onHover(null)}
          >
            <div className="cite-head">
              <span className="cite-n">[{citation.n}]</span>
              <span className="cite-label">{citation.label}</span>
              {citation.rerank_score !== null && (
                <span className="cite-score">{citation.rerank_score.toFixed(3)}</span>
              )}
            </div>
            {citation.text && <p className="cite-text">{citation.text}</p>}
          </li>
        ))}
      </ol>
    </section>
  );
}
