#!/usr/bin/env python
"""
run_real_model.py

All-in-one NFL prediction pipeline:
1. Loads player stats and schedule
2. Builds features
3. Trains models
4. Predicts next game stats
5. Saves predictions.csv
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import timedelta

# --- Add repo root to sys.path so we can import src ---
repo_root = Path(__file__).parent.resolve()
sys.path.append(str(repo_root))

from src.data_loading import load_player_stats, load_schedule
from src.model_training import train_multi_targets
from src.model_predict import predict_next_games, load_models

# --------------------------
# CONFIG
# --------------------------
YEAR = 2023
TARGETS = ["rec_yards", "receptions", "targets"]
MODELS_DIR = repo_root / "models"
PREDICTIONS_OUT = repo_root / "predictions.csv"

# --------------------------
# Step 1: Load historical data
# --------------------------
print(f"Loading player stats for {YEAR}...")
player_stats = load_player_stats(YEAR)

print(f"Loading schedule for {YEAR}...")
schedule = load_schedule(YEAR)

# --------------------------
# Step 2: Train models
# --------------------------
print("Training models...")
MODELS_DIR.mkdir(exist_ok=True)
train_multi_targets(player_stats, TARGETS, team_def_df=None, games_df=schedule, out_dir=MODELS_DIR)

# --------------------------
# Step 3: Prepare next week's games automatically
# --------------------------
last_game_date = player_stats['game_date'].max()
future_games = schedule[schedule['game_date'] > last_game_date]

# Fallback: if no future games, use first week in schedule
if future_games.empty:
    future_games = schedule[schedule['game_date'] == schedule['game_date'].min()]

# Build next_games dataframe with players from last game
recent_players = player_stats[player_stats['game_date'] == last_game_date]
next_games = pd.DataFrame()

for _, game in future_games.iterrows():
    team_players = recent_players[recent_players['team'] == game['home_team']]
    if team_players.empty:
        team_players = recent_players[recent_players['team'] == game['away_team']]
    if team_players.empty:
        team_players = recent_players.sample(min(3, len(recent_players)))
    temp = team_players[['player_id', 'player_name']].copy()
    temp['game_id'] = game['game_id']
    temp['game_date'] = game['game_date']
    temp['team'] = game['home_team']
    temp['opp'] = game['away_team']
    next_games = pd.concat([next_games, temp], ignore_index=True)

print(f"Prepared {len(next_games)} next game rows for prediction.")

# --------------------------
# Step 4: Predict next games
# --------------------------
print("Loading trained models...")
models_dict = load_models(MODELS_DIR)

print("Predicting next game stats...")
predictions = predict_next_games(models_dict, player_stats, next_games, team_def_df=None, games_df=schedule)

# --------------------------
# Step 5: Save predictions
# --------------------------
predictions.to_csv(PREDICTIONS_OUT, index=False)
print(f"Predictions saved to {PREDICTIONS_OUT}")
print(predictions.head())
run_real_model.py
