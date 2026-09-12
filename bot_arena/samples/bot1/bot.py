import json
import sys

# Read input state from system arguments
game_state = json.loads(sys.argv[1])

# Aggressive strategy: always attack
print("attack")
