import json
import sys

game_state = json.loads(sys.argv[1])

# Defensive strategy: heal if HP < 40, else attack
if game_state["your_hp"] < 40:
    print("heal")
else:
    print("attack")
