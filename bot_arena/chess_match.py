"""Referees a chess game between two agent.py submissions.

Each side runs in its own subprocess (engine_worker.py) with its own working
directory, so imports and tablebase paths resolve inside that submission.
python-chess is the referee: it decides legality and the result, not the bots.

    python chess_match.py <white_dir> <black_dir> [--ms 120000] [--inc 500]
                          [--pgn out.pgn]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import shutil

import chess
import chess.pgn

SETUP_TIMEOUT_S = 420  # warm-up is contended when games share a runner
MOVE_TIMEOUT_PAD_S = 15  # hard kill margin on top of a side's remaining clock


class Engine:
    def __init__(self, name, bot_dir, clock_ms, inc_ms=0):
        self.name = name
        self.dir = os.path.abspath(bot_dir)
        self.clock_ms = clock_ms
        self.inc_ms = inc_ms
        self.setup_ms = 0
        self.ready = False
        cmd = [sys.executable, "-u",
               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "engine_worker.py"), self.dir]
        # One core for the whole game. The two engines alternate -- only the
        # side to move thinks -- so sharing one pinned core gives each of them
        # a core to itself while it searches, which is what the competition
        # promises. Without this they drift onto whatever the scheduler has
        # free, and their own nodes-per-second calibration drifts with it.
        core = os.environ.get("CHESS_PIN_CORE")
        if core and os.name == "posix" and shutil.which("taskset"):
            cmd = ["taskset", "-c", core] + cmd
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", cwd=self.dir,
        )

    def _rpc(self, obj, timeout_s):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout_s
        line = self.proc.stdout.readline()  # worker is unbuffered; kill on overrun
        if not line:
            err = self.proc.stderr.read()[-2000:] if self.proc.stderr else ""
            raise RuntimeError(f"{self.name} died. stderr tail:\n{err}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{self.name} exceeded {timeout_s:.0f}s")
        return json.loads(line)

    def init(self):
        r = self._rpc({"cmd": "init"}, SETUP_TIMEOUT_S)
        if not r.get("ok"):
            raise RuntimeError(f"{self.name} failed to load: {r.get('error')}")
        self.setup_ms = r["setup_ms"]
        self.ready = True
        return self.setup_ms

    def alive(self):
        return self.proc.poll() is None

    def reset(self, name, clock_ms, inc_ms):
        """Local testing: start a new game in the same process."""
        self.name = name
        self.clock_ms = clock_ms
        self.inc_ms = inc_ms
        if self.ready:
            r = self._rpc({"cmd": "newgame"}, 60)
            if not r.get("ok"):
                raise RuntimeError(f"{self.name} could not reset: {r.get('error')}")

    def move(self, fen):
        budget = self.clock_ms / 1000.0 + MOVE_TIMEOUT_PAD_S
        r = self._rpc({"cmd": "move", "fen": fen, "ms": int(self.clock_ms)}, budget)
        if not r.get("ok"):
            raise RuntimeError(f"{self.name} raised: {r.get('error')}")
        self.clock_ms -= r["ms"]
        overstepped = self.clock_ms <= 0
        if not overstepped:
            self.clock_ms += self.inc_ms  # increment only if the flag held
        return r["move"], r["ms"]

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play(white_dir, black_dir, clock_ms, inc_ms=0, pgn_path=None, log=print,
         start_fen=None, opening=None, max_plies=600, reuse=None):
    """Play one game. `reuse=(white, black)` plays it on engines that are
    already running, and leaves them running afterwards."""
    if reuse is None:
        white = Engine("White", white_dir, clock_ms, inc_ms)
        black = Engine("Black", black_dir, clock_ms, inc_ms)
    else:
        white, black = reuse
        white.reset("White", clock_ms, inc_ms)
        black.reset("Black", clock_ms, inc_ms)
    engines = {chess.WHITE: white, chess.BLACK: black}

    board = chess.Board(start_fen) if start_fen else chess.Board()
    game = chess.pgn.Game()
    if start_fen:
        game.setup(board)
    if opening:
        game.headers["Opening"] = opening
    game.headers["White"] = os.path.basename(white.dir)
    game.headers["Black"] = os.path.basename(black.dir)
    game.headers["TimeControl"] = f"{clock_ms // 1000}+{inc_ms / 1000:g}"
    node = game
    result, reason = "*", ""

    try:
        for side, eng in engines.items():
            if not eng.ready:
                log(f"Loading {eng.name} ({os.path.basename(eng.dir)}) ...")
                log(f"  warm-up {eng.init() / 1000:.1f}s")

        log(f"\nTime control: {clock_ms / 1000:.0f}s per side, sudden death.\n")

        while True:
            if len(board.move_stack) >= max_plies:
                result, reason = "1/2-1/2", f"Drawn at the {max_plies}-ply cap"
                break
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result = outcome.result()
                reason = outcome.termination.name.replace("_", " ").title()
                break

            eng = engines[board.turn]
            mover = "White" if board.turn == chess.WHITE else "Black"
            fen = board.fen()

            try:
                uci, ms = eng.move(fen)
            except Exception as e:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = f"{mover} error: {e}"
                break

            if eng.clock_ms <= 0:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = f"{mover} forfeits on time ({-eng.clock_ms}ms over)"
                break

            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                move = None
            if move is None or move not in board.legal_moves:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = f"{mover} played illegal move {uci!r}"
                break

            san = board.san(move)
            num = board.fullmove_number
            prefix = f"{num}." if board.turn == chess.WHITE else f"{num}..."
            log(f"{prefix:>6} {san:<8} {ms:>6}ms   clock {eng.clock_ms / 1000:6.1f}s")

            board.push(move)
            node = node.add_variation(move)

        game.headers["Result"] = result
        if reason:
            game.headers["Termination"] = reason

        log(f"\nResult: {result}  ({reason})")
        log(f"Clocks left - White {white.clock_ms / 1000:.1f}s, "
            f"Black {black.clock_ms / 1000:.1f}s   after {board.fullmove_number - 1} moves")

        if pgn_path:
            with open(pgn_path, "w", encoding="utf-8") as f:
                print(game, file=f, end="\n\n")
            log(f"PGN written to {pgn_path}")

        return result, reason, game

    finally:
        if reuse is None:
            white.close()
            black.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("white_dir")
    p.add_argument("black_dir")
    p.add_argument("--ms", type=int, default=120000, help="clock per side, ms")
    p.add_argument("--inc", type=int, default=500, help="increment per move, ms")
    p.add_argument("--pgn")
    p.add_argument("--fen", help="start position (default: initial)")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    play(a.white_dir, a.black_dir, a.ms, a.inc, a.pgn, start_fen=a.fen)


if __name__ == "__main__":
    main()
