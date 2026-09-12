"""Paired-opening match on this machine, several games at once.

Writes the same CSV as bot_arena/run_shard.py, so bot_arena/pool.py reads it.

    python tools/local_match.py <dirA> <dirB> --pairs 30 --jobs 3 \
        --ms 120000 --inc 500 --out runs/x/results_0.csv

Each game runs two engines that alternate on the clock, so a game needs about
one core; --jobs should stay at or under the physical core count or the wall
clock punishes whichever engine is thinking during contention.

Engines stay loaded between games (a `newgame` resets their tables), so only
the first game on each worker pays the minute of numba compilation.
"""
import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bot_arena"))
from chess_match import Engine, play  # noqa: E402
from run_shard import load_book  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--book", default=os.path.join(HERE, "..", "bot_arena", "book.epd"))
    p.add_argument("--pairs", type=int, default=30)
    p.add_argument("--skip", type=int, default=0, help="start at this book index")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--ms", type=int, default=120000)
    p.add_argument("--inc", type=int, default=500)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    name_a = os.path.basename(os.path.abspath(a.dir_a))
    name_b = os.path.basename(os.path.abspath(a.dir_b))
    book = load_book(a.book)
    book = (book * 20)[a.skip:a.skip + a.pairs]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    lock = threading.Lock()
    tally = [0.0, 0]
    local = threading.local()
    everyone = []
    f = open(a.out, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["pos", "opening", "white", "black", "result", "reason",
                f"score_{name_a}", f"score_{name_b}"])
    f.flush()
    pgn = open(a.out.replace(".csv", ".pgn"), "w", encoding="utf-8")

    def engines():
        pair = getattr(local, "pair", None)
        if pair is None or not all(e.alive() for e in pair):
            if pair:
                for e in pair:
                    e.close()
            pair = (Engine("A", a.dir_a, a.ms, a.inc), Engine("B", a.dir_b, a.ms, a.inc))
            local.pair = pair
            with lock:
                everyone.extend(pair)
        return pair

    def one(g):
        idx, a_white = g // 2, g % 2 == 0
        fen, opening = book[idx]
        ea, eb = engines()
        white, black = (ea, eb) if a_white else (eb, ea)
        try:
            res, why, game = play(None, None, a.ms, a.inc, None, lambda m="": None,
                                  start_fen=fen, opening=opening, reuse=(white, black))
        except Exception as e:
            res, why, game = "*", f"harness error: {e}", None
            for e in (ea, eb):
                e.close()
            local.pair = None
        if "error" in why or "died" in why:
            # A crashed or confused engine must not carry into the next game.
            for e in (ea, eb):
                e.close()
            local.pair = None
        if res == "1-0":
            sa = 1.0 if a_white else 0.0
        elif res == "0-1":
            sa = 0.0 if a_white else 1.0
        elif res == "1/2-1/2":
            sa = 0.5
        else:
            sa = None
        with lock:
            if sa is not None:
                tally[0] += sa
                tally[1] += 1
            w.writerow([idx, opening, name_a if a_white else name_b,
                        name_b if a_white else name_a, res, why,
                        0 if sa is None else sa, 0 if sa is None else 1 - sa])
            f.flush()
            if game is not None:
                game.headers["White"] = name_a if a_white else name_b
                game.headers["Black"] = name_b if a_white else name_a
                print(game, file=pgn, end="\n\n")
                pgn.flush()
            print(f"[{g}] {opening}: {'A' if a_white else 'B'} white -> {res} ({why})"
                  f"   {name_a} {tally[0]}/{tally[1]}", flush=True)

    try:
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(one, range(2 * len(book))))
    finally:
        for e in everyone:
            e.close()
        f.close()
        pgn.close()


if __name__ == "__main__":
    main()
