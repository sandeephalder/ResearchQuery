"""Pull the corpus back from the Hugging Face Hub.

`data/` is 6.8 GB and is not in git. It lives in a private dataset repo instead,
and this script restores whatever part of it you need:

    uv run fetch_data.py --part descriptions   # 508 KB — the only paid artifact
    uv run fetch_data.py --part docs           # 105 MB — parse output, no images
    uv run fetch_data.py --part processed      # 3.0 GB — adds the figures
    uv run fetch_data.py --part all            # 6.8 GB — everything, PDFs included

Downloads resume, and files already present with a matching hash are skipped, so
re-running costs nothing. Run from `Ingestion/`, like the other scripts here.
"""

import argparse
import os
import sys

from constants import (CORPUS_PARTS, HF_CORPUS_REPO, HF_TOKEN_ENV_VARS,
                       LOCAL_RAW_DATA_DIR)


def _token():
    """The repo is private, so a token is required."""
    try:
        from dotenv import load_dotenv
        load_dotenv()                       # POC/.env, found by walking up
    except ImportError:
        pass
    for name in HF_TOKEN_ENV_VARS:
        token = os.getenv(name)
        if token:
            return token
    sys.exit(f"No Hugging Face token. Set one of: {', '.join(HF_TOKEN_ENV_VARS)}")


def fetch(part="all", repo_id=HF_CORPUS_REPO, target=LOCAL_RAW_DATA_DIR, force=False):
    from huggingface_hub import snapshot_download

    if part not in CORPUS_PARTS:
        sys.exit(f"Unknown part {part!r}. Choose from: {', '.join(CORPUS_PARTS)}")

    os.makedirs(target, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=CORPUS_PARTS[part],
        force_download=force,
        token=_token(),
    )
    return path


def _describe(target):
    """What landed, so the caller can see it worked without hunting."""
    for name in ("dataset", "raw_dataset", "processed"):
        directory = os.path.join(target, name)
        if not os.path.isdir(directory):
            continue
        files = total = 0
        for root, _, names in os.walk(directory):
            for filename in names:
                files += 1
                total += os.path.getsize(os.path.join(root, filename))
        print(f"  {name:12} {files:6,} files  {total/2**30:6.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the corpus from the Hugging Face Hub")
    parser.add_argument("--part", default="all", choices=sorted(CORPUS_PARTS),
                        help="which subset to pull (default: all)")
    parser.add_argument("--repo", default=HF_CORPUS_REPO, help="dataset repo id")
    parser.add_argument("--target", default=LOCAL_RAW_DATA_DIR, help="where to put it")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the local file matches")
    args = parser.parse_args()

    print(f"Fetching {args.part!r} from {args.repo} into {args.target} ...")
    fetch(args.part, args.repo, args.target, args.force)
    print("\nOn disk:")
    _describe(args.target)
