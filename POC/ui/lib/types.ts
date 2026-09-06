/**
 * The backend's response shapes, mirrored.
 *
 * Hand-written rather than generated, because there are four of them and a
 * generator is a build step to maintain for a contract this small. They are
 * `backend/schemas.py` — if a field is added there, it is added here.
 */

export interface Citation {
  n: number;
  label: string;
  doc_id: string | null;
  kind: string | null;
  rerank_score: number | null;
  /** Only present when the request asked for `include_text`. */
  text: string | null;
}

export interface AskRequest {
  question: string;
  top_n?: number;
  k_vector?: number;
  k_bm25?: number;
  max_repairs?: number;
  rerank?: boolean;
  include_text?: boolean;
}

export interface AskResponse {
  question: string;
  answer: string | null;
  /**
   * Set when the flow stopped early: "guardrail", "no candidates", or
   * "below scope floor (...)". A refusal is a real answer, not an error — it
   * arrives as 200 with this filled in, and should be rendered as an answer.
   */
  refused: string | null;
  citations: Citation[];
  trace: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  corpus_points: number | null;
  bm25_rows: number | null;
  reranker: string | null;
  agent_provider: string | null;
  guardrail_flows: string[];
  in_flight: number;
  detail: string | null;
}

export interface Me {
  user_id: string;
  session_id: string | null;
  org_id: string | null;
}

/** What the trace carries. Every field is optional — an early exit sets few. */
export interface Trace {
  route?: { kind?: string; confidence?: number; lexical?: number; reason?: string } | null;
  route_effect?: string | null;
  best_score?: number | null;
  facts?: { passage: number | null; fact: string }[];
  repairs?: number;
  unresolved?: { type?: string; detail?: string }[];
  refused?: string | null;
  reranker?: string | null;
  vector_mode?: string | null;
  timings?: Record<string, number>;
  guardrail_in?: { allowed?: boolean; rail?: string; message?: string } | null;
  guardrail_out?: { allowed?: boolean; problems?: { type?: string; detail?: string }[] }[];
}
