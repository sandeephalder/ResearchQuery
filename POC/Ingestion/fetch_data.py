"""Restore the corpus from the Hugging Face Hub onto a clean machine.

`data/` is 6.8 GB and deliberately not in git; it lives in a private HF dataset
repo. This rebuilds it — directory skeleton included, so every stage of the
pipeline finds the paths it expects even for the parts you chose not to pull.

    uv run fetch_data.py                        # everything (6.8 GB)
    uv run fetch_data.py --part descriptions    # 508 KB — the only paid artifact
    uv run fetch_data.py --part docs            # 105 MB — parse output, no images
    uv run fetch_data.py --part processed       # 3.0 GB — adds the figures
    uv run fetch_data.py --check                # compare local against the Hub

Downloads resume, and a file whose hash already matches is skipped, so re-running
costs nothing. Paths resolve from this file's location, so it works from any
working directory.
"""

import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import CORPUS_PARTS, HF_CORPUS_REPO, HF_TOKEN_ENV_VARS  # noqa: E402

# Anchored to this file, not the working directory — a restore script that only
# works from one directory is a restore script that fails when you need it.
DATA_DIR = Path(__file__).resolve().parent / "data"

# The full tree the pipeline expects. Empty directories carry no files on the
# Hub, so they have to be recreated explicitly or the first write fails.
EXPECTED_DIRS = [
    "dataset",
    "raw_dataset/pdf/raw_pdf",
    "processed/pdf/docs",
    "processed/pdf/images",
    "processed/pdf/layout",
    "processed/pdf/descriptions",
    "processed/pdf/batches",
]

TOP_LEVEL = ("dataset", "raw_dataset", "processed")
SKIP = (".cache", ".DS_Store", ".gitattributes")


def token():
    """The dataset repo is private, so a token is required."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    for name in HF_TOKEN_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    sys.exit(f"No Hugging Face token. Set one of: {', '.join(HF_TOKEN_ENV_VARS)}")


def ensure_structure(target):
    """Create every directory the pipeline writes into."""
    created = []
    for relative in EXPECTED_DIRS:
        path = Path(target) / relative
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(relative)
    return created


def local_stats(target):
    """(files, bytes) per top-level directory, ignoring caches and metadata."""
    stats = collections.defaultdict(lambda: [0, 0])
    for name in TOP_LEVEL:
        root = Path(target) / name
        if not root.is_dir():
            continue
        for directory, _, filenames in os.walk(root):
            if ".cache" in directory:
                continue
            for filename in filenames:
                if filename in SKIP:
                    continue
                stats[name][0] += 1
                stats[name][1] += (Path(directory) / filename).stat().st_size
    return stats


def remote_stats(repo_id):
    from huggingface_hub import HfApi
    info = HfApi(token=token()).repo_info(repo_id, repo_type="dataset", files_metadata=True)
    stats = collections.defaultdict(lambda: [0, 0])
    for sibling in info.siblings:
        top = sibling.rfilename.split("/")[0]
        if top in SKIP:
            continue
        size = sibling.size if sibling.size is not None else (
            sibling.lfs.size if sibling.lfs else 0)
        stats[top][0] += 1
        stats[top][1] += size or 0
    return stats


def report(target, repo_id):
    """Side-by-side local vs Hub, so a partial restore is obvious."""
    local, remote = local_stats(target), remote_stats(repo_id)
    print(f"\n{'directory':14} {'local files':>12} {'local size':>11} "
          f"{'hub files':>10} {'hub size':>10}  state")
    for name in TOP_LEVEL:
        lf, lb = local.get(name, [0, 0])
        rf, rb = remote.get(name, [0, 0])
        if rf == 0:
            state = "not on hub"
        elif lf == rf and abs(lb - rb) <= max(1, rb * 0.001):
            state = "complete"
        elif lf == 0:
            state = "not pulled"
        elif lf > rf:
            state = f"local ahead ({lf - rf:,} files not yet on hub)"
        else:
            state = f"partial ({lf/rf:.0%} of hub)"
        print(f"{name:14} {lf:12,} {lb/2**30:10.2f}G {rf:10,} {rb/2**30:9.2f}G  {state}")


def fetch(part, repo_id, target, force):
    from huggingface_hub import snapshot_download

    created = ensure_structure(target)
    if created:
        print(f"created {len(created)} directories: {', '.join(created)}")

    print(f"pulling {part!r} from {repo_id} ...")
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(target),
        allow_patterns=CORPUS_PARTS[part], force_download=force, token=token(),
    )
    ensure_structure(target)          # anything the download left empty
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore the corpus from the Hugging Face Hub")
    parser.add_argument("--part", default="all", choices=sorted(CORPUS_PARTS),
                        help="which subset to pull (default: all)")
    parser.add_argument("--repo", default=HF_CORPUS_REPO, help="dataset repo id")
    parser.add_argument("--target", default=str(DATA_DIR), help="where to restore into")
    parser.add_argument("--force", action="store_true", help="re-download matching files")
    parser.add_argument("--check", action="store_true",
                        help="compare local against the Hub, download nothing")
    args = parser.parse_args()

    if args.check:
        ensure_structure(args.target)
        report(args.target, args.repo)
    else:
        fetch(args.part, args.repo, Path(args.target), args.force)
        report(args.target, args.repo)
