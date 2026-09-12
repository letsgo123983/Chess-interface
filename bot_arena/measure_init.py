"""Measures an agent's init cost and memory under competition-like limits.

Run pinned to a single core with thread counts forced to 1; the caller is
responsible for the affinity mask. Reports the numbers that decide whether a
submission fits the 90s init budget and the 2 GB memory cap.
"""
import os
import sys
import time

for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import psutil

# One core, exclusively: match the competition machine before anything is
# imported, so numba compiles under the same limit it will run under.
psutil.Process().cpu_affinity([0])

bot_dir = os.path.abspath(sys.argv[1])
os.chdir(bot_dir)
sys.path.insert(0, bot_dir)

t0 = time.monotonic()
import agent  # noqa: E402  - the warm-up runs at import, as in competition
init_s = time.monotonic() - t0

peak_mb = psutil.Process().memory_info().peak_wset / (1024 * 1024)
rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

t0 = time.monotonic()
mv = agent.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 120000)
move_s = time.monotonic() - t0
after_mb = psutil.Process().memory_info().peak_wset / (1024 * 1024)

print(f"{os.path.basename(bot_dir)}: init {init_s:6.1f}s  "
      f"(budget 90s, headroom {90 - init_s:+.1f}s) | "
      f"first move {mv} in {move_s:.2f}s | "
      f"peak {peak_mb:.0f}MB -> {after_mb:.0f}MB, rss {rss_mb:.0f}MB (cap 2048MB)")
