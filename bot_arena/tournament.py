"""Plays engine A vs engine B over and over, alternating colors, until stopped.

Runs unattended: every finished game is appended to results.csv and games.pgn
and the standings file is rewritten, so the tournament can be read at any time
and nothing is lost if the process is killed mid-game.

Stop it by deleting nothing and simply creating a file named STOP in the
output directory, or by killing the process.

    python tournament.py <dirA> <dirB> --out <dir> [--ms 120000] [--inc 500]
"""
import argparse
import csv
import datetime as dt
import io
import os
import sys
import traceback

import chess.pgn

from chess_match import play


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--out", required=True)
    p.add_argument("--ms", type=int, default=120000)
    p.add_argument("--inc", type=int, default=500)
    p.add_argument("--max-games", type=int, default=0, help="0 = until stopped")
    a = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)

    name_a = os.path.basename(os.path.abspath(a.dir_a))
    name_b = os.path.basename(os.path.abspath(a.dir_b))
    results_csv = os.path.join(a.out, "results.csv")
    pgn_path = os.path.join(a.out, "games.pgn")
    standings_path = os.path.join(a.out, "standings.txt")
    stop_path = os.path.join(a.out, "STOP")

    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["game", "finished", "white", "black", "result", "reason",
                 "plies", "score_a", "score_b"])

    wins_a = wins_b = draws = 0
    game_no = 0

    print(f"Tournament {name_a} vs {name_b} — "
          f"{a.ms // 1000}s + {a.inc / 1000:g}s, started {now()}", flush=True)
    print(f"Output: {a.out}", flush=True)

    while True:
        if os.path.exists(stop_path):
            print(f"\nSTOP file found, finishing at {now()}.", flush=True)
            break
        if a.max_games and game_no >= a.max_games:
            break

        game_no += 1
        a_is_white = (game_no % 2 == 1)
        w_dir, b_dir = (a.dir_a, a.dir_b) if a_is_white else (a.dir_b, a.dir_a)
        w_name, b_name = (name_a, name_b) if a_is_white else (name_b, name_a)

        print(f"\n===== Game {game_no}: {w_name} (W) vs {b_name} (B) "
              f"— {now()} =====", flush=True)

        buf = io.StringIO()

        def log(msg=""):
            print(msg, flush=True)
            buf.write(str(msg) + "\n")

        try:
            result, reason, game = play(w_dir, b_dir, a.ms, a.inc, None, log)
        except Exception:
            traceback.print_exc()
            print(f"Game {game_no} aborted; continuing.", flush=True)
            continue

        # Score from engine A's point of view.
        if result == "1-0":
            sa, sb = (1, 0) if a_is_white else (0, 1)
        elif result == "0-1":
            sa, sb = (0, 1) if a_is_white else (1, 0)
        else:
            sa = sb = 0.5
        wins_a += sa == 1
        wins_b += sb == 1
        draws += sa == 0.5

        game.headers["Event"] = f"{name_a} vs {name_b}"
        game.headers["Round"] = str(game_no)
        game.headers["Date"] = dt.datetime.now().strftime("%Y.%m.%d")
        plies = len(list(game.mainline_moves()))

        with open(pgn_path, "a", encoding="utf-8") as f:
            print(game, file=f, end="\n\n")
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [game_no, now(), w_name, b_name, result, reason, plies, sa, sb])

        played = wins_a + wins_b + draws
        summary = (
            f"{name_a} vs {name_b}   {a.ms // 1000}s + {a.inc / 1000:g}s\n"
            f"updated {now()}\n\n"
            f"games played : {played}\n"
            f"{name_a:<13}: {wins_a} wins\n"
            f"{name_b:<13}: {wins_b} wins\n"
            f"draws        : {draws}\n"
            f"score        : {wins_a + draws / 2} - {wins_b + draws / 2}\n"
        )
        with open(standings_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print("\n" + summary, flush=True)

    print(f"\nTournament ended {now()}.", flush=True)


if __name__ == "__main__":
    main()
