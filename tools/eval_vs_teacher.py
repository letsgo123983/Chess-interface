"""Compare nets' centipawns with the teacher's on real game positions.

    python tools/eval_vs_teacher.py <game.pgn> <net.npz> [<net.npz> ...] \
        [--from 14 --to 22 --depth 14]

Prints each net's evaluation in engine centipawns (its logit times the
cpscale stored in the file, or 110 for a net without one) beside the
teacher's search score, all from the side to move. Then the median ratio of
each net to the teacher over the non-decisive positions, which is what the
search margins care about: a net that reads 3x the teacher near equality
makes every futility and razoring margin a third of what it was tuned as.
"""
import argparse
import os
import sys

import chess
import chess.engine
import chess.pgn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import featurize, quantised_logits  # noqa: E402

TEACHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "teacher.exe")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pgn")
    p.add_argument("nets", nargs="+")
    p.add_argument("--from", dest="start", type=int, default=0)
    p.add_argument("--to", type=int, default=80)
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--depth", type=int, default=12)
    a = p.parse_args()

    fens = []
    with open(a.pgn, encoding="utf-8") as f:
        for _ in range(a.games):
            g = chess.pgn.read_game(f)
            if g is None:
                break
            b = g.board()
            for i, mv in enumerate(g.mainline_moves()):
                b.push(mv)
                if a.start <= i <= a.to and not b.is_check():
                    fens.append(b.fen())
    st, ns, _, _ = featurize([f + " | 0 | 0.5" for f in fens])
    cols = []
    for path in a.nets:
        d = np.load(path)
        scale = int(d["cpscale"][0]) if "cpscale" in d.files else 110
        cols.append(quantised_logits(path, st, ns) * scale)

    eng = chess.engine.SimpleEngine.popen_uci(os.path.abspath(TEACHER))
    truth = []
    for fen in fens:
        board = chess.Board(fen)
        s = eng.analyse(board, chess.engine.Limit(depth=a.depth))["score"].pov(board.turn)
        truth.append(s.score(mate_score=3000))
    eng.quit()
    truth = np.array(truth, dtype=float)

    names = [os.path.basename(os.path.dirname(os.path.dirname(p))) or p for p in a.nets]
    print("teacher " + " ".join(f"{n[:10]:>10s}" for n in names))
    for i in range(0, len(fens), max(1, len(fens) // 15)):
        print(f"{truth[i]:7.0f} " + " ".join(f"{c[i]:10.0f}" for c in cols))
    mask = (np.abs(truth) >= 30) & (np.abs(truth) <= 400)
    for n, c in zip(names, cols):
        ratio = np.median(c[mask] / truth[mask]) if mask.any() else float("nan")
        corr = np.corrcoef(c, truth)[0, 1]
        print(f"{n}: median ratio to teacher (30-400cp) {ratio:.2f}, corr {corr:.3f}, "
              f"over {mask.sum()} of {len(fens)} positions")


if __name__ == "__main__":
    main()
