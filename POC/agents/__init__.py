"""Query-time agents: the flow from a question to an answer that has been checked.

    guardrail   is this safe to answer at all?                     (llm/guard model)
    router      which modality does it need, and with which knobs?
    retrieval   retrieval.Pipeline, unchanged — hybrid search then rerank
    extractor   the raw numbers, bounds and table rows in the passages
    generator   a draft citing [1]..[n], from those facts and those passages
    verifier    does the draft survive a reading of the passages? one repair

`orchestrator.AgentPipeline` runs all of it and returns the same keys
`retrieval.Pipeline.arun` does, so `evals/answer_eval.py` can score one against
the other and say what the extra calls bought.

Nothing is imported here on purpose. `retrieval` keeps its `__init__` empty for
the same reason: importing the orchestrator pulls in qdrant, torch and BGE-M3,
which is a 15-second import to pay only when it is actually wanted.
"""
