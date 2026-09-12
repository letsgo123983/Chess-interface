"""Compile time and search speed for one engine under given numba settings.

    python tools/compile_probe.py engines/base --core 2 --env NUMBA_OPT=2

Runs a fresh process pinned to one core: times the import (what the init
budget measures) and then searches fixed positions at a fixed node limit,
reporting depth reached and nodes per second. Compile time and playing
strength move in opposite directions here, so both are measured together.
"""
import argparse
import json
import os
import subprocess
import sys

import psutil

CHILD = r"""
import json, os, sys, time
t0 = time.monotonic()
import agent
import fastchess as fc
t1 = time.monotonic()
FENS = ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 9",
        "2r2rk1/pp1bqpp1/2n1pn1p/3p4/2PP4/1PN1PN2/P1Q2PPP/2RR2K1 w - - 0 16"]
out = []
for fen in FENS:
    agent.ENGINE = agent.Engine()
    board, state = fc.from_fen(fen)
    agent.ENGINE.hkey[0] = fc.hash_position(board, state)
    agent.ENGINE.counters[:] = 0
    agent.ENGINE.counters[fc.C_LIMIT] = 3_000_000
    t = time.monotonic()
    depth = 0
    for d in range(1, 40):
        n = fc.search_root(board, state, agent.ENGINE.history, agent.ENGINE.hashes,
            agent.ENGINE.hkey, agent.ENGINE.moves, agent.ENGINE.scores,
            agent.ENGINE.scratch, agent.ENGINE.killers, agent.ENGINE.history_heuristic,
            agent.ENGINE.cont_history, agent.ENGINE.cont_stack, agent.ENGINE.tt_key,
            agent.ENGINE.tt_data, agent.ENGINE.counters, agent.ENGINE.counter_moves,
            agent.ENGINE.game_keys, agent.ENGINE.game_count, agent.ENGINE.evals,
            agent.ENGINE.corr, agent.ENGINE.capt_hist, d, -fc.INFINITY, fc.INFINITY,
            agent.ENGINE.out)
        if agent.ENGINE.counters[fc.C_ABORT] != 0 or n == 0:
            break
        depth = d
    spent = time.monotonic() - t
    out.append({"depth": depth, "nodes": int(agent.ENGINE.counters[fc.C_NODES]),
                "nps": int(agent.ENGINE.counters[fc.C_NODES] / max(spent, 1e-6))})
print(json.dumps({"import": t1 - t0, "runs": out}))
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine")
    p.add_argument("--core", type=int, default=2)
    p.add_argument("--env", action="append", default=[])
    a = p.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHESS_")}
    env.update(NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for e in a.env:
        k, _, v = e.partition("=")
        env[k] = v
    proc = subprocess.Popen([sys.executable, "-c", CHILD], cwd=os.path.abspath(a.engine),
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    psutil.Process(proc.pid).cpu_affinity([a.core])
    so, se = proc.communicate(timeout=1200)
    try:
        d = json.loads(so.strip().splitlines()[-1])
    except Exception:
        print("FAILED", se[-800:])
        return
    tag = ",".join(a.env) or "default"
    nps = sum(r["nps"] for r in d["runs"]) // len(d["runs"])
    depths = "/".join(str(r["depth"]) for r in d["runs"])
    print(f"{a.engine} [{tag}] import {d['import']:5.1f}s  depth {depths}  nps {nps}")


if __name__ == "__main__":
    main()
