"""Compare vision models on real figures from the parsed corpus.

Cost and speed are easy to measure and rarely decide anything; what decides a
model here is whether it reads a scientific plot correctly. So this prints the
descriptions side by side and leaves the judging to a human.

Any OpenAI-compatible endpoint works, which includes a local Ollama server and
the LiteLLM gateway:

    uv run python -m llm.compare_vlm --models ollama:qwen3-vl:4b --limit 6
    uv run python -m llm.compare_vlm --models ollama:qwen3-vl:4b gemini openai:gpt-4.1-nano

Figures are sampled from documents that have captions, because a caption is the
only reference we have for what the figure actually shows — read the description
against it and the errors stand out.
"""

import argparse
import asyncio
import glob
import json
import os
import random
import time

from .chains import describe_figure_chain
from .constants import PROVIDERS
from .images import encode_image

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "Ingestion/data/processed/pdf")


def load_figures(limit, seed=0, require_caption=True):
    """Sample (image path, caption, title, heading) from the parsed corpus."""
    figures = []
    for path in sorted(glob.glob(f"{PROCESSED_DIR}/docs/*.json")):
        document = json.load(open(path))
        for section in document["sections"]:
            for record in section["images"].values():
                if require_caption and not record["caption"]:
                    continue
                image = os.path.join(PROCESSED_DIR, "images", record["path"])
                if os.path.exists(image):
                    figures.append({"image": image, "caption": record["caption"],
                                    "title": document["title"], "heading": section["heading"],
                                    "doc": document["id"], "page": record["page"]})
    random.Random(seed).shuffle(figures)
    return figures[:limit]


def endpoint(spec):
    """"ollama:qwen3-vl:4b", "openai:gpt-4.1-mini" or bare "gemini" -> (provider, model)."""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r}; known: {', '.join(PROVIDERS)}")
    return provider, model or None          # None -> the provider's default vision model


async def describe_all(spec, figures):
    """One chain per endpoint, reused across the sample.

    Built once rather than per figure: the point of the comparison is the
    model's time, and rebuilding a client for each call would put connection
    setup inside the measurement.
    """
    provider, model = endpoint(spec)
    chain = describe_figure_chain(provider, model)
    results = []
    for figure in figures:
        start = time.perf_counter()
        try:
            text = await chain.ainvoke({"image_uri": encode_image(figure["image"]),
                                        "caption": figure["caption"], "title": figure["title"],
                                        "heading": figure["heading"], "context": None})
        except Exception as error:
            text = f"<FAILED: {type(error).__name__}: {str(error)[:160]}>"
        results.append({"model": spec, "seconds": time.perf_counter() - start,
                        "description": text})
    return results


async def main(specs, limit, seed, out_path):
    figures = load_figures(limit, seed)
    if not figures:
        raise SystemExit(f"No captioned figures under {PROCESSED_DIR} — run data_process.py first")
    print(f"{len(figures)} figures x {len(specs)} models = {len(figures)*len(specs)} calls\n")

    by_model = {}
    for spec in specs:
        print(f"running {spec} ...", flush=True)
        by_model[spec] = await describe_all(spec, figures)

    for index, figure in enumerate(figures):
        print("\n" + "=" * 100)
        print(f"[{index+1}] {figure['doc']} p{figure['page']} — {figure['image']}")
        print(f"CAPTION (reference): {figure['caption'][:300]}")
        for spec in specs:
            result = by_model[spec][index]
            print(f"\n  --- {spec}  ({result['seconds']:.1f}s)")
            for line in result["description"].splitlines():
                print(f"      {line}")

    print("\n" + "=" * 100)
    print(f"{'model':28} {'median s':>9} {'mean chars':>11} {'failures':>9}")
    for spec in specs:
        results = by_model[spec]
        times = sorted(r["seconds"] for r in results)
        failures = sum(r["description"].startswith("<FAILED") for r in results)
        chars = [len(r["description"]) for r in results if not r["description"].startswith("<FAILED")]
        print(f"{spec:28} {times[len(times)//2]:9.1f} "
              f"{(sum(chars)/len(chars) if chars else 0):11.0f} {failures:9}")

    if out_path:
        with open(out_path, "w") as handle:
            json.dump({"figures": figures, "results": by_model}, handle, indent=2)
        print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare vision models on parsed figures")
    parser.add_argument("--models", nargs="+", required=True,
                        help="<provider>[:<model>], e.g. ollama:qwen3-vl:4b, gemini, openai:gpt-4.1-nano")
    parser.add_argument("--limit", type=int, default=6, help="figures to sample")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--out", help="also write the raw results to this JSON file")
    args = parser.parse_args()
    asyncio.run(main(args.models, args.limit, args.seed, args.out))
