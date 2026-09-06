"""A hard ceiling on how many questions may be scored in a day.

    from evals.budget import Budget

    budget = Budget()
    allowed = budget.clamp(requested=100)     # what you may actually run
    budget.consume(len(records))              # after they are scored

The full split is 3,045 questions and this flow makes six model calls each, plus
five RAGAS metrics on top. In one working session that was enough to exhaust
groq's 200,000 tokens-per-day allowance, OpenRouter's 50 free-model requests and
DeepSeek's balance — measured, not predicted. A sweep that runs until the keys
die produces no result and costs the rest of the day.

So a hundred a day is the budget, and it is enforced here rather than left to
whoever types `--limit`. The number is what one day's evaluation is worth: enough
to see a regression, cheap enough to run on a change, and small enough that the
providers survive it.

## What it counts, and what it does not

It counts **questions scored**, not API calls and not tokens — because that is
the number a caller controls and the one the sample is expressed in. It is a
guard against an accidental 3,045-question sweep, not a token accountant.

Records already in the output file do not count again. A resumed run scores only
what is left, so resuming after an interruption is not charged twice for work
already paid for.

## The ledger

A JSON file keyed by UTC date. Not a lock, and not synchronised: two sweeps
started at once can both pass the check and overshoot. That is a deliberate
non-feature — the embedded Qdrant store already admits one process, so two
concurrent sweeps cannot happen, and a lock here would be machinery guarding
against a state the corpus makes impossible.

Dates are UTC so the ceiling does not move with the machine's timezone, and old
entries are kept: the ledger is also the record of how much evaluation has been
run, which is worth having when a metric moves and the question is what changed.
"""

import datetime
import json
import os

DAILY_LIMIT = int(os.getenv("EVAL_DAILY_LIMIT", "100"))

_HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_PATH = os.getenv("EVAL_BUDGET_PATH",
                        os.path.join(os.path.dirname(_HERE), "runs", "eval_budget.json"))


class BudgetExhausted(RuntimeError):
    """Today's allowance is spent."""


def today():
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


class Budget:
    """The day's remaining allowance, and the ledger behind it."""

    def __init__(self, limit=DAILY_LIMIT, path=LEDGER_PATH, date=None):
        self.limit = limit
        self.path = path
        self.date = date or today()

    # -- the ledger ---------------------------------------------------------- #

    def _read(self):
        try:
            with open(self.path) as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # `.part` then rename, so a kill cannot leave a truncated ledger that
        # the next run reads as "nothing spent today".
        with open(self.path + ".part", "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(self.path + ".part", self.path)

    # -- the allowance ------------------------------------------------------- #

    @property
    def spent(self):
        return int(self._read().get(self.date, 0))

    @property
    def remaining(self):
        return max(0, self.limit - self.spent)

    def clamp(self, requested):
        """How many of `requested` may actually be scored today.

        Raises when nothing is left, rather than returning 0 and letting a sweep
        report an empty result as though the pipeline had produced one.
        """
        if self.limit <= 0:
            return requested                     # EVAL_DAILY_LIMIT=0 disables it
        remaining = self.remaining
        if remaining <= 0:
            raise BudgetExhausted(
                f"today's evaluation budget is spent: {self.spent}/{self.limit} "
                f"questions scored on {self.date}.\n"
                f"    raise it for one run with EVAL_DAILY_LIMIT=200\n"
                f"    turn it off entirely with EVAL_DAILY_LIMIT=0\n"
                f"    or wait — it resets at 00:00 UTC")
        return min(requested, remaining)

    def consume(self, count):
        """Record `count` questions against today. Returns what is left."""
        if self.limit <= 0 or count <= 0:
            return self.remaining
        data = self._read()
        data[self.date] = int(data.get(self.date, 0)) + int(count)
        self._write(data)
        return self.remaining

    def describe(self):
        if self.limit <= 0:
            return "daily budget: off (EVAL_DAILY_LIMIT=0)"
        return (f"daily budget: {self.spent}/{self.limit} spent on {self.date}, "
                f"{self.remaining} left")


def history(path=LEDGER_PATH, days=14):
    """[(date, count)] most recent first — what has been evaluated lately."""
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return sorted(((d, int(n)) for d, n in data.items()), reverse=True)[:days]
