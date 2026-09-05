"""Describe every parsed figure with a VLM, via the OpenAI Batch API.

Batch is half price and the work is offline, so the only cost of the 24-hour
window is that the run outlives the process. Everything here is built around
that: state lives on disk, and any phase can be re-run.

    uv run python -m llm.enrich submit     # build, upload, create batches
    uv run python -m llm.enrich status     # where are they
    uv run python -m llm.enrich collect    # merge finished results back
    uv run python -m llm.enrich merge      # write descriptions into the doc JSONs

Durability, in order of what actually goes wrong:

- A description is written to `descriptions/{doc_id}.json` and never to the
  parse output, so re-running data_process.py cannot destroy paid-for work.
- Files are written `.part` then atomically renamed, so a kill cannot leave a
  truncated file that the next run trusts.
- Work is skipped by *content hash*, not filename. Re-parsing renumbers
  img-N.png; hashing the encoded image means you do not pay twice for the same
  picture, and a changed prompt correctly re-pays.
- Submitted batch ids are recorded before anything else happens. Lose the
  process and `collect` still finds the results.
- Failures land in `failed_descriptions.json` with the provider's own message,
  and `submit --retry-failed` re-queues only those.
"""

import argparse
import asyncio
import collections
import glob
import hashlib
import json
import os
import time

from .batch import BatchAPI, DONE_STATES, parse_result_line, request_line
from .client import LLMClient, LLMError, _parse_messages, encode_image
from .constants import PARSE_SYSTEM_PROMPT, PARSE_TEMPERATURE

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "Ingestion/data/processed/pdf")
DESCRIPTIONS_DIR = os.path.join(PROCESSED_DIR, "descriptions")
BATCH_STATE_DIR = os.path.join(PROCESSED_DIR, "batches")
FAILURES_PATH = os.path.join(PROCESSED_DIR, "failed_descriptions.json")

BATCH_MODEL = os.getenv("LLM_BATCH_MODEL", "gpt-4.1-mini")
BATCH_MAX_TOKENS = 400
# Conservative against the provider's per-file limits; images are base64 inline,
# so the byte cap binds long before the request count does.
MAX_BATCH_BYTES = 180 * 1024 * 1024
MAX_BATCH_REQUESTS = 40_000

# Bump when the prompt changes so existing descriptions are recomputed.
PROMPT_VERSION = hashlib.sha256(PARSE_SYSTEM_PROMPT.encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# On-disk state
# --------------------------------------------------------------------------- #

def _write_atomic(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path, default):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_descriptions(doc_id):
    return _read_json(os.path.join(DESCRIPTIONS_DIR, f"{doc_id}.json"), {})


def save_descriptions(doc_id, records):
    _write_atomic(os.path.join(DESCRIPTIONS_DIR, f"{doc_id}.json"), records)


def custom_id(doc_id, image_id):
    return f"{doc_id}::{image_id}"


def split_custom_id(value):
    doc_id, _, image_id = value.partition("::")
    return doc_id, image_id


# --------------------------------------------------------------------------- #
# What still needs describing
# --------------------------------------------------------------------------- #

def pending_figures(limit=None, only_failed=False):
    """Figures with no current description, newest parse wins.

    A figure is done when a stored description matches the image's content hash,
    the model and the prompt version — so a reparse costs nothing, a new prompt
    costs everything, and a swapped model costs only what it changes.
    """
    failures = _read_json(FAILURES_PATH, {}) if only_failed else None
    pending = []

    for path in sorted(glob.glob(os.path.join(PROCESSED_DIR, "docs", "*.json"))):
        document = json.load(open(path))
        doc_id = document["id"]
        stored = load_descriptions(doc_id)

        for section in document["sections"]:
            for image_id, record in section["images"].items():
                image_path = os.path.join(PROCESSED_DIR, "images", record["path"])
                if not os.path.exists(image_path):
                    continue
                if only_failed and custom_id(doc_id, image_id) not in failures:
                    continue

                previous = stored.get(image_id)
                digest = _image_digest(image_path)
                if (previous and not only_failed
                        and previous.get("image_sha256") == digest
                        and previous.get("model") == BATCH_MODEL
                        and previous.get("prompt_version") == PROMPT_VERSION):
                    continue

                pending.append({
                    "custom_id": custom_id(doc_id, image_id),
                    "doc_id": doc_id, "image_id": image_id, "image_path": image_path,
                    "image_sha256": digest, "caption": record.get("caption"),
                    "title": document.get("title"), "heading": section.get("heading"),
                })
                if limit and len(pending) >= limit:
                    return pending
    return pending


_digest_cache = {}


def _image_digest(path):
    key = (path, os.path.getmtime(path))
    if key not in _digest_cache:
        with open(path, "rb") as handle:
            _digest_cache[key] = hashlib.sha256(handle.read()).hexdigest()
    return _digest_cache[key]


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

def build_chunks(figures):
    """Group request lines into files under the provider's per-file limits."""
    chunks, current, size = [], [], 0
    for figure in figures:
        messages = _parse_messages(encode_image(figure["image_path"]), figure["caption"],
                                   figure["title"], figure["heading"])
        line = request_line(figure["custom_id"], BATCH_MODEL, messages,
                            PARSE_TEMPERATURE, BATCH_MAX_TOKENS)
        encoded = len(line.encode()) + 1
        if current and (size + encoded > MAX_BATCH_BYTES or len(current) >= MAX_BATCH_REQUESTS):
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += encoded
    if current:
        chunks.append(current)
    return chunks


async def submit(limit=None, only_failed=False, dry_run=False):
    figures = pending_figures(limit, only_failed)
    if not figures:
        print("Nothing to describe — every figure already has a current description.")
        return

    print(f"{len(figures)} figures to describe with {BATCH_MODEL} "
          f"(prompt {PROMPT_VERSION})")
    print("encoding images ...", flush=True)
    chunks = build_chunks(figures)
    total_bytes = sum(len(line.encode()) for chunk in chunks for line in chunk)
    print(f"{len(chunks)} batch file(s), {total_bytes/2**20:.0f} MB total")

    if dry_run:
        print("dry run — nothing uploaded")
        return

    os.makedirs(BATCH_STATE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    async with BatchAPI("openai") as api:
        for index, chunk in enumerate(chunks):
            name = f"{stamp}_{index:02d}"
            jsonl_path = os.path.join(BATCH_STATE_DIR, f"{name}.jsonl")
            with open(jsonl_path, "w") as handle:
                handle.write("\n".join(chunk) + "\n")

            print(f"uploading {name} ({len(chunk)} requests) ...", flush=True)
            file_id = await api.upload(jsonl_path)
            batch = await api.create(file_id)

            # Recorded before anything else can fail: losing this file means
            # paying twice, and the batch id is the only handle on the results.
            _write_atomic(os.path.join(BATCH_STATE_DIR, f"{name}.json"), {
                "name": name, "batch_id": batch["id"], "input_file_id": file_id,
                "model": BATCH_MODEL, "prompt_version": PROMPT_VERSION,
                "requests": len(chunk), "status": batch.get("status"),
                "submitted_at": stamp, "collected": False,
            })
            print(f"  batch {batch['id']} — {batch.get('status')}")

    print(f"\nSubmitted. Results land within 24h; check with:  "
          f"uv run python -m llm.enrich status")


def _batch_states():
    return sorted(glob.glob(os.path.join(BATCH_STATE_DIR, "*.json")))


async def status():
    states = _batch_states()
    if not states:
        print("No batches submitted yet.")
        return
    async with BatchAPI("openai") as api:
        print(f"{'name':20} {'batch id':32} {'status':12} {'done/total':>12} {'collected':>10}")
        for path in states:
            state = _read_json(path, {})
            try:
                batch = await api.retrieve(state["batch_id"])
            except LLMError as error:
                print(f"{state.get('name',''):20} {state.get('batch_id',''):32} "
                      f"ERROR {str(error)[:60]}")
                continue
            counts = batch.get("request_counts") or {}
            state["status"] = batch.get("status")
            state["output_file_id"] = batch.get("output_file_id")
            state["error_file_id"] = batch.get("error_file_id")
            _write_atomic(path, state)
            print(f"{state['name']:20} {state['batch_id']:32} {batch.get('status',''):12} "
                  f"{str(counts.get('completed',0))+'/'+str(counts.get('total',0)):>12} "
                  f"{str(state.get('collected', False)):>10}")


async def collect():
    """Download finished batches and write the descriptions to disk."""
    states = [p for p in _batch_states() if not _read_json(p, {}).get("collected")]
    if not states:
        print("Nothing to collect.")
        return

    failures = _read_json(FAILURES_PATH, {})
    written = collections.Counter()

    async with BatchAPI("openai") as api:
        for path in states:
            state = _read_json(path, {})
            batch = await api.retrieve(state["batch_id"])
            if batch.get("status") not in DONE_STATES:
                print(f"{state['name']}: {batch.get('status')} — not ready")
                continue
            if not batch.get("output_file_id"):
                print(f"{state['name']}: {batch.get('status')} with no output file")
                state.update(status=batch.get("status"), collected=True)
                _write_atomic(path, state)
                continue

            print(f"{state['name']}: downloading ...", flush=True)
            text = await api.download(batch["output_file_id"])

            # Group by document so each file is written once, atomically.
            by_doc = collections.defaultdict(dict)
            for line in text.splitlines():
                if not line.strip():
                    continue
                cid, content, error = parse_result_line(line)
                doc_id, image_id = split_custom_id(cid)
                if error or not content:
                    failures[cid] = {"reason": error or "empty completion",
                                     "batch": state["batch_id"]}
                    written["failed"] += 1
                    continue
                failures.pop(cid, None)
                by_doc[doc_id][image_id] = content
                written["ok"] += 1

            index = {f["custom_id"]: f for f in pending_figures()}
            for doc_id, descriptions in by_doc.items():
                stored = load_descriptions(doc_id)
                for image_id, content in descriptions.items():
                    figure = index.get(custom_id(doc_id, image_id), {})
                    stored[image_id] = {
                        "description": "" if content.upper().startswith("UNREADABLE") else content,
                        "model": state["model"], "prompt_version": state["prompt_version"],
                        "image_sha256": figure.get("image_sha256"),
                        "batch": state["batch_id"],
                    }
                save_descriptions(doc_id, stored)

            state.update(status=batch.get("status"), collected=True)
            _write_atomic(path, state)

    _write_atomic(FAILURES_PATH, failures) if failures else _remove(FAILURES_PATH)
    print(f"\n{written['ok']} descriptions written, {written['failed']} failed")
    if failures:
        print(f"{len(failures)} outstanding in {FAILURES_PATH} — "
              f"re-queue with: uv run python -m llm.enrich submit --retry-failed")


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def merge():
    """Copy descriptions into the parsed documents' image metadata."""
    merged = docs = 0
    for path in sorted(glob.glob(os.path.join(PROCESSED_DIR, "docs", "*.json"))):
        document = json.load(open(path))
        stored = load_descriptions(document["id"])
        if not stored:
            continue
        touched = False
        for section in document["sections"]:
            for image_id, record in section["images"].items():
                entry = stored.get(image_id)
                if entry and record.get("description") != entry["description"]:
                    record["description"] = entry["description"]
                    record["description_model"] = entry["model"]
                    merged += 1
                    touched = True
        if touched:
            _write_atomic(path, document)
            docs += 1
    print(f"merged {merged} descriptions into {docs} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe parsed figures via the Batch API")
    parser.add_argument("phase", choices=["submit", "status", "collect", "merge", "pending"])
    parser.add_argument("--limit", type=int, help="cap the number of figures submitted")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-queue only the ids in failed_descriptions.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the batch files and report size, upload nothing")
    args = parser.parse_args()

    if args.phase == "submit":
        asyncio.run(submit(args.limit, args.retry_failed, args.dry_run))
    elif args.phase == "status":
        asyncio.run(status())
    elif args.phase == "collect":
        asyncio.run(collect())
    elif args.phase == "merge":
        merge()
    else:
        figures = pending_figures(args.limit)
        print(f"{len(figures)} figures pending (model {BATCH_MODEL}, prompt {PROMPT_VERSION})")
        for figure in figures[:10]:
            print(f"  {figure['custom_id']:34} {str(figure['caption'])[:60]}")
