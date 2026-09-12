"""Plays one shard of a paired-opening match and writes a CSV of results.

A shard is a list of book positions. Each position is played twice, once with
each engine as White, which is what cancels the first-move advantage out of
the final score. The two games of a pair run concurrently: only the side to
move is thinking, so a pair needs about two cores, not four.

    python run_shard.py <dirA> <dirB> --book book.epd --shard 0 --shards 30 \
                        --out results_0.csv --pgn games_0.pgn
"""
import argparse
import csv
import os
import sys
import threading

from chess_match import play


def load_book(path):
    positions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fen, _, name = line.partition(";")
            positions.append((fen.strip(), name.strip() or "?"))
    return positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--book", required=True)
    p.add_argument("--positions", type=int, default=0,
                   help="use only the first N book positions (games = 2N)")
    p.add_argument("--gshard", type=int, default=0, help="which slice of games")
    p.add_argument("--gshards", type=int, default=1, help="how many slices")
    p.add_argument("--game", type=int, default=None,
                   help="play exactly one game: index 0..2N-1 over the book, "
                        "position = index // 2, colour = index %% 2. One game "
                        "per process means one thinking engine at a time.")
    p.add_argument("--ms", type=int, default=120000)
    p.add_argument("--inc", type=int, default=500)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--pgn", default="games.pgn")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    name_a = os.path.basename(os.path.abspath(a.dir_a))
    name_b = os.path.basename(os.path.abspath(a.dir_b))
    book = load_book(a.book)
    if a.positions:
        book = book[:a.positions]
    total = 2 * len(book)
    if a.game is not None:
        indices = [a.game]
    else:
        indices = [g for g in range(total) if g % a.gshards == a.gshard]
    print(f"shard {a.gshard}/{a.gshards}: playing games {indices} "
          f"concurrently, {name_a} vs {name_b}", flush=True)

    rows = []
    games = []
    lock = threading.Lock()

    results = {}

    def one(game_index):
        idx = game_index // 2
        a_white = (game_index % 2 == 0)
        fen, opening = book[idx]
        w, b = (a.dir_a, a.dir_b) if a_white else (a.dir_b, a.dir_a)
        try:
            res, why, game = play(w, b, a.ms, a.inc, None, lambda m="": None,
                                  start_fen=fen, opening=opening)
        except Exception as e:
            res, why, game = "*", f"harness error: {e}", None
        with lock:
            print(f"[game {game_index}] {opening}: "
                  f"{'A' if a_white else 'B'} as White -> {res} ({why})",
                  flush=True)
        results[game_index] = (idx, opening, a_white, res, why, game)

    threads = [threading.Thread(target=one, args=(g,)) for g in indices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for game_index in sorted(results):
        idx, opening, a_white, res, why, game = results[game_index]
        if res == "1-0":
            sa, sb = (1, 0) if a_white else (0, 1)
        elif res == "0-1":
            sa, sb = (0, 1) if a_white else (1, 0)
        elif res == "1/2-1/2":
            sa = sb = 0.5
        else:
            sa = sb = 0  # aborted; recorded so it cannot pass silently
        rows.append([idx, opening, name_a if a_white else name_b,
                     name_b if a_white else name_a, res, why, sa, sb])
        if game is not None:
            game.headers["Event"] = f"{name_a} vs {name_b}"
            game.headers["Round"] = str(game_index)
            games.append(game)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos", "opening", "white", "black", "result", "reason",
                    f"score_{name_a}", f"score_{name_b}"])
        w.writerows(rows)
    with open(a.pgn, "w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")
    print(f"wrote {len(rows)} game(s) to {a.out}", flush=True)


if __name__ == "__main__":
    main()
