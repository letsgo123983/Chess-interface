"""Time a submission's import the way the competition counts it.

    python tools/init_check.py submissions/a.zip --runs 3 [--rival submissions/b.zip]

Each run unzips the submission afresh, starts a new Python pinned to one core
with numba single-threaded and no local overrides, and times:
  import   what the 90 s init budget measures
  move 1   the first get_move, which absorbs any compile left over when the
           import returned at its 78 s deadline
With --rival, the rival is imported at the same moment on the same core,
which is the worst case if the platform starts both agents together.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import psutil

CHILD = r"""
import json, os, sys, time
t0 = time.monotonic()
import agent
t1 = time.monotonic()
move = agent.get_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 120000)
t2 = time.monotonic()
print(json.dumps({"import": t1 - t0, "move1": t2 - t1, "move": move}))
"""


def launch(zip_path, core):
    d = tempfile.mkdtemp(prefix="initcheck_")
    zipfile.ZipFile(zip_path).extractall(d)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHESS_")}
    env.update(NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1")
    p = subprocess.Popen([sys.executable, "-c", CHILD], cwd=d, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    psutil.Process(p.pid).cpu_affinity([core])
    return p, d


def main():
    a = argparse.ArgumentParser()
    a.add_argument("zip")
    a.add_argument("--runs", type=int, default=3)
    a.add_argument("--rival")
    a.add_argument("--core", type=int, default=2)
    args = a.parse_args()
    worst = 0.0
    for r in range(args.runs):
        procs = [launch(args.zip, args.core)]
        if args.rival:
            procs.append(launch(args.rival, args.core))
        out = []
        for p, d in procs:
            so, se = p.communicate(timeout=900)
            shutil.rmtree(d, ignore_errors=True)
            try:
                out.append(json.loads(so.strip().splitlines()[-1]))
            except Exception:
                out.append({"error": se[-500:]})
        me = out[0]
        worst = max(worst, me.get("import", 999))
        line = f"run {r + 1}: import {me.get('import', 0):5.1f}s  move1 {me.get('move1', 0):5.1f}s"
        if args.rival:
            line += f"   | rival import {out[1].get('import', 0):5.1f}s"
        print(line, me.get("error", ""), flush=True)
    print(f"worst import {worst:.1f}s  ({'OK' if worst < 90 else 'OVER'} vs 90s)")


if __name__ == "__main__":
    main()
