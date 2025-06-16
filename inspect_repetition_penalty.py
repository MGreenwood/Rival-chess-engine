import pickle
import sys
import os

# Allow running from anywhere in the project
sys.path.insert(0, os.path.abspath('python'))

# Path to the self-play data file
file_path = "checkpoints/rival_ai/self_play_data/games_epoch_88.pkl"

with open(file_path, "rb") as f:
    games = pickle.load(f)

print(f"Loaded {len(games)} games from {file_path}\n")

for i, game in enumerate(games[:10]):
    print(f"Game {i+1}:")
    print("  Result:", game.result)
    print("  Values:", game.values)
    print("  Num moves:", game.num_moves)
    print()

# Optionally, count how many games are repetition draws
num_repetition = sum(1 for g in games if str(g.result).endswith('REPETITION_DRAW'))
print(f"Total repetition draws in file: {num_repetition} / {len(games)}") 