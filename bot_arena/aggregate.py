"""Merges shard CSVs into a final standings table."""
import csv
import glob
import math
import sys


def main():
    rows = []
    header = None
    for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                                 else "results_*.csv")):
        with open(path, encoding="utf-8") as f:
            r = list(csv.reader(f))
        if not r:
            continue
        header = header or r[0]
        rows.extend(r[1:])

    if not rows:
        print("no results found")
        return

    name_a = header[6].replace("score_", "")
    name_b = header[7].replace("score_", "")
    sa = sum(float(r[6]) for r in rows)
    sb = sum(float(r[7]) for r in rows)
    n = len(rows)
    wins_a = sum(1 for r in rows if float(r[6]) == 1)
    wins_b = sum(1 for r in rows if float(r[7]) == 1)
    draws = sum(1 for r in rows if float(r[6]) == 0.5)
    white_wins = sum(1 for r in rows if r[4] == "1-0")
    black_wins = sum(1 for r in rows if r[4] == "0-1")

    pct = sa / n if n else 0
    if 0 < pct < 1:
        elo = -400 * math.log10(1 / pct - 1)
        # Standard error of the score, converted to Elo at the current slope.
        se = math.sqrt(sum((float(r[6]) - pct) ** 2 for r in rows)) / n
        lo, hi = pct - 1.96 * se, pct + 1.96 * se
        band = ""
        if 0 < lo < 1 and 0 < hi < 1:
            band = (f"  (95% CI {-400 * math.log10(1 / lo - 1):+.0f} .. "
                    f"{-400 * math.log10(1 / hi - 1):+.0f})")
        elo_line = f"{elo:+.0f} Elo for {name_a}{band}"
    else:
        elo_line = "score is 0% or 100%; Elo undefined"

    print(f"games played : {n}")
    print(f"{name_a:<13}: {wins_a} wins")
    print(f"{name_b:<13}: {wins_b} wins")
    print(f"draws        : {draws}")
    print(f"score        : {sa} - {sb}  ({pct:.1%} for {name_a})")
    print(f"elo          : {elo_line}")
    print(f"colour split : White won {white_wins}, Black won {black_wins}, "
          f"{draws} drawn")
    print()
    print("Decisive games by opening:")
    for r in rows:
        if r[4] not in ("1/2-1/2", "*"):
            print(f"  {r[1]:<28} {r[2]:>4}(W) vs {r[3]:<4}(B)  {r[4]:>7}  {r[5]}")


if __name__ == "__main__":
    main()
