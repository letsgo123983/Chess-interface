"""Sanity checks for an engine directory before it plays a match.

    python tools/check_engine.py engines/me [--keys]

--keys  plays random games through make_move/unmake_move and checks the
        incrementally kept pawn key against a fresh from_fen at every ply
Always: times the import (numba warm-up, which counts against the 90 s init
budget) and searches a few positions, printing depth and nodes per second.
"""
import os
import random
import sys
import time

import chess

bot = os.path.abspath(sys.argv[1])
os.chdir(bot)
sys.path.insert(0, bot)

if "--keys" in sys.argv:
    import fastchess as fc
    history, hashes, hkey, moves = fc.new_buffers()
    rng = random.Random(1)
    checked = 0
    for game in range(40):
        ref = chess.Board()
        board, state = fc.from_fen(ref.fen())
        for ply in range(160):
            legal = list(ref.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            count = fc.generate(board, state, moves[0], 0)
            ours = None
            for i in range(count):
                m = int(moves[0, i])
                uci = (chess.square_name((fc.move_from(m) >> 4) * 8 + (fc.move_from(m) & 7))
                       + chess.square_name((fc.move_to(m) >> 4) * 8 + (fc.move_to(m) & 7)))
                if fc.move_flags(m) & fc.F_PROMO:
                    uci += ".pnbrqk"[fc.move_promo(m)]
                if uci == move.uci():
                    ours = m
            assert ours is not None, move
            assert fc.make_move(board, state, history, hashes, hkey, 0, ours)
            ref.push(move)
            _, fresh = fc.from_fen(ref.fen())
            assert state[fc.ST_PKEY] == fresh[fc.ST_PKEY], (game, ply, ref.fen())
            # And the undo restores it exactly.
            before = state.copy()
            fc.unmake_move(board, state, history, hashes, hkey, 0)
            assert fc.make_move(board, state, history, hashes, hkey, 0, ours)
            assert (state == before).all()
            checked += 1
    print(f"pawn keys consistent over {checked} plies")

t0 = time.time()
import agent  # noqa: E402
print(f"init {time.time() - t0:.1f}s")

FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 9",
    "2r2rk1/pp1bqpp1/2n1pn1p/3p4/2PP4/1PN1PN2/P1Q2PPP/2RR2K1 w - - 0 16",
    "8/5pk1/6p1/3R4/5P2/r5P1/6K1/8 b - - 3 45",
]
for fen in FENS:
    agent.ENGINE = agent.Engine()
    t0 = time.time()
    move = agent.get_move(fen, 60000)
    print(f"{move:6s} depth {agent.ENGINE.last_depth:2d} score {agent.ENGINE.last_score:5d} "
          f"{time.time() - t0:5.2f}s  nps {int(agent.ENGINE.nps)}")
