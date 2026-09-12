import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from flask import Flask, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit


def extract_zip(zip_file, extract_path):
    """Extracts a zip file safely into a given target directory."""
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Basic check against path traversal attacks (Zip Slip)
        for member in zip_ref.namelist():
            if member.startswith("/") or ".." in member:
                raise Exception("Unsafe ZIP file detected.")
        zip_ref.extractall(extract_path)

    # Ensure entry script exists
    if not os.path.exists(os.path.join(extract_path, "bot.py")):
        raise Exception("ZIP must contain a 'bot.py' entry file.")


def get_bot_move(bot_dir, game_state):
    """Runs a bot's script via subprocess and returns its move."""
    try:
        # Pass game state as a JSON string argument
        result = subprocess.run(
            [sys.executable, "bot.py", json.dumps(game_state)],
            cwd=bot_dir,
            capture_output=True,
            text=True,
            timeout=2,  # 2 second timeout per turn
        )
        if result.returncode != 0:
            return None, f"Runtime Error: {result.stderr.strip()}"

        return result.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "Timed out (> 2 seconds)."
    except Exception as e:
        return None, str(e)


def run_match_engine(bot1_dir, bot2_dir):
    """Runs a simple turn-based game loop between Bot 1 and Bot 2."""
    logs = []
    logs.append("=== MATCH STARTING ===")

    # Game initial state
    hp = {"Bot 1": 100, "Bot 2": 100}
    turn = 1
    max_turns = 20

    while turn <= max_turns and hp["Bot 1"] > 0 and hp["Bot 2"] > 0:
        logs.append(
            f"\n--- Turn {turn} [Bot 1 HP: {hp['Bot 1']} | Bot 2 HP: {hp['Bot 2']}] ---"
        )

        # 1. Bot 1 Turn
        state_for_bot1 = {
            "turn": turn,
            "your_hp": hp["Bot 1"],
            "opponent_hp": hp["Bot 2"],
        }
        move1, err1 = get_bot_move(bot1_dir, state_for_bot1)

        if err1:
            logs.append(f"Bot 1 failed: {err1}")
            logs.append("🏆 Bot 2 wins by default!")
            return "\n".join(logs)

        logs.append(f"Bot 1 choice: {move1}")

        # Simple game rules: 'attack' deals 20 damage, 'heal' recovers 10 HP
        if move1 == "attack":
            hp["Bot 2"] -= 20
        elif move1 == "heal":
            hp["Bot 1"] = min(100, hp["Bot 1"] + 10)

        # Check win condition
        if hp["Bot 2"] <= 0:
            logs.append("\n🏆 Bot 1 eliminated Bot 2 and won!")
            return "\n".join(logs)

        # 2. Bot 2 Turn
        state_for_bot2 = {
            "turn": turn,
            "your_hp": hp["Bot 2"],
            "opponent_hp": hp["Bot 1"],
        }
        move2, err2 = get_bot_move(bot2_dir, state_for_bot2)

        if err2:
            logs.append(f"Bot 2 failed: {err2}")
            logs.append("🏆 Bot 1 wins by default!")
            return "\n".join(logs)

        logs.append(f"Bot 2 choice: {move2}")

        if move2 == "attack":
            hp["Bot 1"] -= 20
        elif move2 == "heal":
            hp["Bot 2"] = min(100, hp["Bot 2"] + 10)

        if hp["Bot 1"] <= 0:
            logs.append("\n🏆 Bot 2 eliminated Bot 1 and won!")
            return "\n".join(logs)

        turn += 1

    logs.append("\n=== MATCH ENDED (MAX TURNS REACHED) ===")
    if hp["Bot 1"] > hp["Bot 2"]:
        logs.append("🏆 Bot 1 wins on higher remaining HP!")
    elif hp["Bot 2"] > hp["Bot 1"]:
        logs.append("🏆 Bot 2 wins on higher remaining HP!")
    else:
        logs.append("🤝 It's a draw!")

    return "\n".join(logs)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run-match", methods=["POST"])
def run_match():
    bot1_file = request.files.get("bot1")
    bot2_file = request.files.get("bot2")

    if not bot1_file or not bot2_file:
        return render_template("index.html", error="Please upload both zip files.")

    # Create an isolated temporary directory for this match
    match_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"match_{match_id}")

    bot1_dir = os.path.join(temp_dir, "bot1")
    bot2_dir = os.path.join(temp_dir, "bot2")

    try:
        os.makedirs(bot1_dir, exist_ok=True)
        os.makedirs(bot2_dir, exist_ok=True)

        # Extract submitted ZIP files
        extract_zip(bot1_file, bot1_dir)
        extract_zip(bot2_file, bot2_dir)

        # Run the game loop
        match_log = run_match_engine(bot1_dir, bot2_dir)

        return render_template("index.html", match_log=match_log)

    except Exception as e:
        return render_template("index.html", error=str(e))

    finally:
        # Cleanup temporary files after the match completes
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
