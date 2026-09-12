"""Pools several runs of the same match into one verdict.

A single 60-game run carries a +/-60 Elo interval, which cannot resolve the
size of change search tweaks actually make. Running the same pairing several
times and pooling the games is the cheapest way to a real answer: the interval
narrows with the square root of the game count, at no cost in fidelity because
every game is still played at the competition control.

    python pool.py run1/ run2/ run3/          # dirs of results_*.csv
    python pool.py runs/**/results_*.csv      # or the files directly
"""
import csv
import glob
import math
import os
import sys


def load(paths):
    rows, header = [], None
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "results_*.csv"),
                                      recursive=True))
        else:
            files += sorted(glob.glob(p))
    for f in files:
        with open(f, encoding="utf-8") as fh:
            r = list(csv.reader(fh))
        if len(r) < 2:
            continue
        header = header or r[0]
        rows += r[1:]
    return header, rows, files


def elo(p):
    if p <= 0 or p >= 1:
        return float("inf") if p >= 1 else float("-inf")
    return -400 * math.log10(1 / p - 1)


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    header, rows, files = load(sys.argv[1:])
    if not rows:
        print("no results found")
        return

    name_a = header[6].replace("score_", "")
    name_b = header[7].replace("score_", "")
    n = len(rows)
    sa = sum(float(r[6]) for r in rows)
    wins_a = sum(1 for r in rows if float(r[6]) == 1)
    wins_b = sum(1 for r in rows if float(r[7]) == 1)
    draws = sum(1 for r in rows if float(r[6]) == 0.5)
    aborted = sum(1 for r in rows if r[4] == "*")

    p = sa / n
    # Standard error of the mean score, from the actual per-game outcomes.
    var = sum((float(r[6]) - p) ** 2 for r in rows) / n
    se = math.sqrt(var / n) if n else 0.0
    lo, hi = p - 1.96 * se, p + 1.96 * se
    # Likelihood of superiority: P(true score > 0.5) under a normal approx.
    los = phi((p - 0.5) / se) if se > 0 else float("nan")

    print(f"pooled from {len(files)} shard file(s)")
    print(f"games        : {n}")
    print(f"{name_a:<13}: {wins_a} wins")
    print(f"{name_b:<13}: {wins_b} wins")
    print(f"draws        : {draws}" + (f"   ABORTED: {aborted}" if aborted else ""))
    print(f"score        : {sa} - {n - sa}  ({p:.1%} for {name_a})")
    print(f"elo          : {elo(p):+.0f}   95% CI [{elo(lo):+.0f}, {elo(hi):+.0f}]")
    print(f"LOS          : {los:.1%} chance {name_a} is genuinely stronger")
    print()
    if lo > 0.5:
        print(f"VERDICT: {name_a} is stronger (interval excludes equality).")
    elif hi < 0.5:
        print(f"VERDICT: {name_b} is stronger (interval excludes equality).")
    else:
        # How many games would be needed to resolve the observed margin?
        need = ""
        if abs(p - 0.5) > 1e-9 and var > 0:
            n_need = var * (1.96 / abs(p - 0.5)) ** 2
            need = f"  ~{int(n_need)} games would be needed to confirm this margin."
        print(f"VERDICT: inconclusive -- the interval still includes equality.{need}")


if __name__ == "__main__":
    main()
