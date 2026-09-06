"""A fixed sample of the benchmark, so two runs can be compared.

    uv run python -m evals.sample                  # write evals/sample_100.json
    uv run python -m evals.sample --size 200 --out evals/sample_200.json
    uv run python -m evals.sample --show           # describe the saved sample

The full split is 3,045 questions, and a sweep over it makes six model calls per
question plus five RAGAS metrics. A hundred is enough to see a regression and
cheap enough to run on a change, which is why it is the default — but only if it
is *the same* hundred every time. A fresh random draw per run turns every
comparison into a comparison of two different question sets, and the differences
that matter here are smaller than that noise.

So the sample is drawn once, written to disk, and committed. `answer_eval.py`
reads it by default.

## Stratified, with a floor

Proportional allocation over the four query sources gives:

    text              63
    text-image        25
    text-table-image   7
    text-table         5

The two table cells are too small to say anything, and modality is precisely
what this corpus is weak at — image queries scored 2.41 against text's 2.73 on
the old scale, with four times the abstentions. A sample that cannot resolve the
thing you are measuring is the wrong sample.

So each source gets at least `MIN_PER_SOURCE`, and the remainder is allocated
proportionally by largest remainder so the total lands exactly on `size`.

**What that costs:** the sample is no longer corpus-representative, so the "all"
row of a report is a mean over *this* sample rather than an estimate of pipeline
quality over the corpus. The per-source rows are the ones to read, and they are
the reason for the floor. A pure proportional draw is `--no-floor`.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.paths import RetrievalError                          # noqa: E402

DEFAULT_SIZE = 100
DEFAULT_SEED = 0
# Enough that a per-source mean is worth printing. Below about ten, one bad
# question moves the cell by ten points and the column reads as noise.
MIN_PER_SOURCE = 10

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_100.json")


def daily_seed(date=None):
    """A stable seed for one UTC date.

    Derived from the date rather than from a counter, so every machine agrees on
    what today's hundred are without sharing state, and yesterday's run is
    reproducible by naming yesterday.
    """
    day = date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    return int(hashlib.sha256(day.encode()).hexdigest()[:8], 16)


def allocate(counts, size, floor=MIN_PER_SOURCE):
    """{source: how many to draw}, summing exactly to `size`.

    Every source gets `floor` (or all it has, if it has fewer), and what is left
    is shared out in proportion to the pool by largest remainder — which is what
    makes the total land on `size` exactly rather than one or two short after
    rounding.
    """
    sources = sorted(counts)
    if sum(counts.values()) <= size:
        return dict(counts)

    quota = {s: min(floor, counts[s]) for s in sources}
    remaining = size - sum(quota.values())
    if remaining < 0:
        raise RetrievalError(
            f"a floor of {floor} across {len(sources)} sources needs at least "
            f"{sum(quota.values())} questions; asked for {size}")

    headroom = {s: counts[s] - quota[s] for s in sources}
    total = sum(headroom.values())
    exact = {s: (remaining * headroom[s] / total) if total else 0 for s in sources}
    whole = {s: int(exact[s]) for s in sources}

    # Largest remainder, then source name, so the tie-break is deterministic.
    short = remaining - sum(whole.values())
    for source in sorted(sources, key=lambda s: (-(exact[s] - whole[s]), s))[:short]:
        whole[source] += 1

    return {s: quota[s] + min(whole[s], headroom[s]) for s in sources}


def draw(queries, size=DEFAULT_SIZE, seed=DEFAULT_SEED, floor=MIN_PER_SOURCE, date=None):
    """The sample, as {"seed", "size", "sources", "qids"}.

    Sorted pools and a per-source `Random(seed)` make this reproducible: the same
    seed and the same benchmark give the same hundred questions on any machine.

    `date` switches to the rotating daily sample — the seed comes from the date,
    so each day draws a different hundred and thirty days cover 3,000 of the
    3,045. That is the trade against the fixed sample: **a daily sample is not
    comparable across days**, because a change in the mean could be the change
    you made or could be the different questions. Use the fixed sample to compare
    two configurations, and the daily one to accumulate coverage.
    """
    if date is not None:
        seed = daily_seed(date)
    pools = {}
    for qid, query in queries.items():
        pools.setdefault(query.get("source") or "unknown", []).append(qid)
    for source in pools:
        pools[source].sort()

    counts = {s: len(p) for s, p in pools.items()}
    quota = allocate(counts, size, floor)

    qids, chosen = [], {}
    for source in sorted(quota):
        take = min(quota[source], len(pools[source]))
        picked = sorted(random.Random(seed).sample(pools[source], take))
        chosen[source] = take
        qids.extend(picked)

    return {"seed": seed, "size": len(qids), "floor": floor, "date": date,
            "sources": chosen, "pool_sizes": counts, "qids": sorted(qids)}


def load(path=SAMPLE_PATH):
    """The saved sample's qids, or a failure that says how to make one."""
    if not os.path.exists(path):
        raise RetrievalError(
            f"No sample at {path}. Draw one — it costs nothing and no API call:\n"
            f"    uv run python -m evals.sample")
    with open(path) as handle:
        return json.load(handle)


def describe(sample):
    when = f", rotating sample for {sample['date']}" if sample.get("date") else " (fixed)"
    print(f"{sample['size']} questions, seed {sample['seed']}, "
          f"floor {sample.get('floor', 0)} per source{when}\n")
    print(f"{'source':20} {'sampled':>8} {'pool':>7} {'share':>7}")
    for source in sorted(sample["sources"]):
        taken = sample["sources"][source]
        pool = sample.get("pool_sizes", {}).get(source, 0)
        print(f"{source:20} {taken:>8} {pool:>7} {taken / sample['size']:>7.0%}")


def main():
    parser = argparse.ArgumentParser(description="Draw a fixed benchmark sample")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="same seed = same questions")
    parser.add_argument("--floor", type=int, default=MIN_PER_SOURCE,
                        help=f"minimum questions per source (default {MIN_PER_SOURCE})")
    parser.add_argument("--no-floor", action="store_true",
                        help="pure proportional allocation, no minimum")
    parser.add_argument("--daily", nargs="?", const="today", metavar="YYYY-MM-DD",
                        help="the rotating sample for a date (default today, UTC) "
                             "instead of the fixed one")
    parser.add_argument("--out", default=SAMPLE_PATH)
    parser.add_argument("--show", action="store_true",
                        help="describe the saved sample and exit")
    arguments = parser.parse_args()

    from evals.answer_eval import load_benchmark

    if arguments.show:
        describe(load(arguments.out))
        return

    queries, _, _ = load_benchmark()
    date = None
    if arguments.daily:
        date = (datetime.datetime.now(datetime.timezone.utc).date().isoformat()
                if arguments.daily == "today" else arguments.daily)
    sample = draw(queries, arguments.size, arguments.seed,
                  0 if arguments.no_floor else arguments.floor, date)
    describe(sample)

    if date and arguments.out == SAMPLE_PATH:
        # A rotating sample is derived from its date, so writing it over the
        # committed fixed sample would replace the one thing every comparison
        # depends on with something that changes daily.
        print("\n(not written — a daily sample is derived from its date; pass "
              "--out to save one)")
        return
    os.makedirs(os.path.dirname(arguments.out) or ".", exist_ok=True)
    with open(arguments.out, "w") as handle:
        json.dump(sample, handle, indent=2)
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    try:
        main()
    except RetrievalError as error:
        sys.exit(f"FAILED: {error}")
