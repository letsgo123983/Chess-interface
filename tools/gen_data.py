"""Training positions for the evaluation network, labelled by a reference engine.

Plays games against itself from randomised openings and records every quiet
position with the reference engine's search score and the game's final result.
The engine is only a teacher here: nothing from it ships in the bot, which
learns its own weights from these labels.

    python tools/gen_data.py <engine.exe> <out.txt> --games 100000 --seed 1

Output, one position per line:  fen | score_cp_side_to_move | result_white
where result_white is 1, 0.5 or 0.
"""
import argparse
import random
import sys

import chess
import chess.engine

MATE_CP = 3000


def score_cp(info, board):
    score = info["score"].pov(board.turn)
    if score.is_mate():
        return None
    return score.score()


def play_game(engine, rng, depth, out):
    board = chess.Board()
    # A random opening: the network has to judge ugly positions too, because
    # the search visits far more of them than good ones.
    for _ in range(rng.randint(6, 11)):
        moves = list(board.legal_moves)
        if not moves:
            return 0
        board.push(rng.choice(moves))
    if board.is_game_over():
        return 0

    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    first = score_cp(info, board)
    if first is None or abs(first) > 700:
        return 0

    rows = []
    result = None
    big = 0
    quiet_draw = 0
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result = 0.5 if outcome.winner is None else (1.0 if outcome.winner else 0.0)
            break
        if board.ply() > 400:
            result = 0.5
            break
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        pv = info.get("pv")
        if not pv:
            result = 0.5
            break
        move = pv[0]
        cp = score_cp(info, board)
        white_cp = None if cp is None else (cp if board.turn else -cp)

        # Adjudicate: a decided game teaches nothing more past the point it
        # was decided, and draws by shuffling teach nothing at all.
        if cp is None or abs(cp) > 2000:
            big += 1
            if big >= 4:
                if cp is None:
                    mate = info["score"].pov(chess.WHITE).mate()
                    result = 1.0 if mate > 0 else 0.0
                else:
                    result = 1.0 if white_cp > 0 else 0.0
                break
        else:
            big = 0
        if cp is not None and abs(cp) <= 8 and board.ply() > 100:
            quiet_draw += 1
            if quiet_draw >= 12:
                result = 0.5
                break
        else:
            quiet_draw = 0

        quiet = (cp is not None and not board.is_check()
                 and not board.is_capture(move) and move.promotion is None)
        if quiet:
            rows.append((board.fen(), max(-MATE_CP, min(MATE_CP, cp))))
        board.push(move)

    for fen, cp in rows:
        out.write(f"{fen} | {cp} | {result}\n")
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine")
    p.add_argument("out")
    p.add_argument("--games", type=int, default=100000)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = random.Random(a.seed)
    engine = chess.engine.SimpleEngine.popen_uci(a.engine)
    engine.configure({"Threads": 1, "Hash": 16})
    total = 0
    with open(a.out, "a", encoding="utf-8", buffering=1 << 16) as out:
        for game in range(a.games):
            total += play_game(engine, rng, a.depth, out)
            if game % 50 == 0:
                out.flush()
                print(f"game {game} positions {total}", file=sys.stderr, flush=True)
    engine.quit()


if __name__ == "__main__":
    main()
