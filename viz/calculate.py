#!/usr/bin/env python3
"""Unified calculate script.

Usage:
  python calculate.py --model elo
  python calculate.py --model trueskill
  python calculate.py --model trueskill_mov

If no argument provided, `DEFAULT_MODEL` will be used.
Outputs CSV to `viz/<model>_results.csv`.
"""

import argparse
import csv
import os
from collections import OrderedDict
from datetime import datetime, timedelta
import sys

# Ensure repo root is on sys.path so `data` and `models` packages import correctly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.data_loader import load_teams, load_games, get_all_players, parse_time_maybe
from models.elo import EloModel
from models.trueskill import TrueSkillModel
from models.trueskill_mov import TrueSkillMovModel
from models.bt_mov import BTMOVModel
from models.bt_mov_time_decay import BTMOVTimeDecayModel

# Default model name (can be changed)
DEFAULT_MODEL = 'elo'  # 'elo' | 'trueskill' | 'trueskill_mov'

TEAMS = os.path.join('data', 'Sports Elo - Teams.csv')
GAMES = os.path.join('data', 'Sports Elo - Games.csv')
VIZ_DIR = 'viz'

# Centralized hyperparameter configuration for all models
# Models own their defaults; calculate.py stays model-agnostic.


def _write_map(filepath, header_players, data_map):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Time'] + header_players)
        items = []
        for tstr in data_map.keys():
            parsed = parse_time_maybe(tstr)
            items.append((parsed, tstr))
        items.sort(key=lambda x: (x[0] is None, x[0]))
        for _, tstr in items:
            writer.writerow(data_map[tstr])


def run_model(model_name, team_to_players, games, all_players, output_file, force_recompute=False):
    """Unified runner for all models. Chooses instantiation and per-row handling
    based on `model_name` while preserving previous behavior.
    
    Args:
        model_name (str): Name of the model to run
        team_to_players (dict): Mapping of team names to player lists
        games (list): List of game data rows
        all_players (list): List of all player IDs
        output_file (str): Path to output CSV file
        force_recompute (bool): If True, ignore existing state and recompute from scratch
    """

    # Instantiate the chosen model with default hyperparameters (model owns defaults)
    model_classes = {
        'elo': EloModel,
        'trueskill': TrueSkillModel,
        'trueskill_mov': TrueSkillMovModel,
        'bt_mov': BTMOVModel,
        'bt_mov_time_decay': BTMOVTimeDecayModel,
    }
    ModelClass = model_classes.get(model_name)
    if ModelClass is None:
        raise SystemExit('Unknown model: ' + model_name)
    model = ModelClass()

    history_map = OrderedDict()
    last_processed_time = None

    # Attempt to resume from existing state unless force_recompute is set
    if not force_recompute and os.path.exists(output_file):
        print(f'Found existing results at {output_file}')
        if model.load_state(output_file, all_players):
            print('Successfully loaded previous state. Resuming from last game...')
            # Load existing history from CSV
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:
                        # Skip header, load all existing rows
                        for row in rows[1:]:
                            time_str = row[0]
                            history_map[time_str] = row
                        last_processed_time = rows[-1][0]
                print(f'Loaded {len(history_map)} existing game results.')
            except Exception as e:
                print(f'Warning: Could not load history from {output_file}: {e}')
                print('Starting fresh computation...')
                history_map = OrderedDict()
                last_processed_time = None
        else:
            print('Could not resume (model does not support resuming or file incompatible).')
            print('Starting fresh computation...')
    elif force_recompute:
        print('Force recompute enabled. Starting fresh computation...')
    else:
        print('No existing results found. Starting fresh computation...')

    # Prepend initial row 20 minutes before earliest game (only if starting fresh)
    if not last_processed_time and games:
        earliest_dt = None
        for row in games:
            time_str = row[0]
            try:
                dt = datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S')
            except Exception:
                continue
            if earliest_dt is None or dt < earliest_dt:
                earliest_dt = dt
        if earliest_dt is not None:
            initial_dt = earliest_dt - timedelta(minutes=20)
            initial_time_str = initial_dt.strftime('%m/%d/%Y %H:%M:%S')
            history_map[initial_time_str] = [initial_time_str] + model.expose(all_players)

    games_processed = 0
    num_existing_games = len(history_map) if last_processed_time else 0
    
    for row in games:
        # Treat every row uniformly: always pass the full row to the model.
        # `game_row` layout expected by models: [time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field]
        # Some rows may omit the 8th column; models should handle missing/None values.
        time = row[0] if len(row) > 0 else ''
        t1 = row[1] if len(row) > 1 else ''
        t2 = row[3] if len(row) > 3 else ''

        # Skip games we've already processed when resuming
        # Only skip if the timestamp matches exactly and we're resuming from a known state
        if last_processed_time:
            # When resuming, skip all games at or before the last processed timestamp
            # Use parse_time_maybe to handle various timestamp formats flexibly
            game_dt = parse_time_maybe(time)
            last_dt = parse_time_maybe(last_processed_time)
            if game_dt and last_dt and game_dt <= last_dt:
                continue

        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])

        # Call model's update with the whole row; model decides what to use.
        model.update(row, team1, team2)
        row_out = [time] + model.expose(all_players)
        history_map[time] = row_out
        games_processed += 1

    if last_processed_time:
        print(f'Processed {games_processed} new games.')
    else:
        print(f'Processed {games_processed} total games.')

    _write_map(output_file, all_players, history_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL, help='Model to run: elo, trueskill, trueskill_mov, bt_mov, bt_mov_time_decay')
    parser.add_argument('--force', '-f', action='store_true', help='Force recompute from scratch, ignoring existing results')
    args = parser.parse_args()

    model_name = args.model.lower()
    if model_name not in ('elo', 'trueskill', 'trueskill_mov', 'bt_mov', 'bt_mov_time_decay'):
        raise SystemExit('Unknown model: ' + model_name)

    team_to_players = load_teams(TEAMS)
    games = load_games(GAMES)
    
    # Sort games by timestamp to ensure chronological processing
    # This is critical for resume functionality to work correctly
    games_with_parsed_time = []
    for row in games:
        time_str = row[0] if len(row) > 0 else ''
        parsed_time = parse_time_maybe(time_str)
        games_with_parsed_time.append((parsed_time, row))
    
    # Sort by parsed time (None values go to the end)
    games_with_parsed_time.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else datetime.max))
    games = [row for _, row in games_with_parsed_time]
    
    all_players = get_all_players(team_to_players)

    output_file = os.path.join(VIZ_DIR, f"{model_name}_results.csv")

    run_model(model_name, team_to_players, games, all_players, output_file, force_recompute=args.force)

    print('Wrote', output_file)


if __name__ == '__main__':
    main()
