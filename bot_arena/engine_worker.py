"""Hosts one chess agent in its own process, speaking JSON lines on stdio.

Protocol: parent writes one JSON object per line, worker answers with one.
  {"cmd": "init"}                          -> {"ok": true, "setup_ms": ...}
  {"cmd": "move", "fen": ..., "ms": ...}   -> {"ok": true, "move": "e2e4", "ms": ...}
  {"cmd": "newgame"}                       -> {"ok": true}
  {"cmd": "quit"}                          -> exits
Keeping the agent resident preserves its transposition table and its numba
warm-up between moves, which is how the competition harness runs it too.

`newgame` exists for local testing only: it replaces the agent's ENGINE with a
fresh one, so the next game starts from empty tables without paying another
minute of compilation. The competition always starts a new process.
"""
import json
import os
import sys
import time

bot_dir = os.path.abspath(sys.argv[1])
os.chdir(bot_dir)
sys.path.insert(0, bot_dir)

agent = None


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    cmd = req.get("cmd")

    if cmd == "quit":
        break

    if cmd == "init":
        # Import triggers the agent's own warm-up (numba compilation).
        t0 = time.monotonic()
        try:
            import agent as _agent
            agent = _agent
            reply({"ok": True, "setup_ms": int((time.monotonic() - t0) * 1000)})
        except Exception as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
        continue

    if cmd == "newgame":
        try:
            if agent is not None and hasattr(agent, "Engine"):
                agent.ENGINE = None
                agent.ENGINE = agent.Engine()
                reply({"ok": True})
            else:
                reply({"ok": False, "error": "agent has no Engine to reset"})
        except Exception as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
        continue

    if cmd == "move":
        t0 = time.monotonic()
        try:
            move = agent.get_move(req["fen"], int(req["ms"]))
            reply({"ok": True, "move": move,
                   "ms": int((time.monotonic() - t0) * 1000)})
        except Exception as e:
            reply({"ok": False, "error": f"{type(e).__name__}: {e}",
                   "ms": int((time.monotonic() - t0) * 1000)})
        continue

    reply({"ok": False, "error": f"unknown command {cmd!r}"})
