import csv
import json
from datetime import datetime


def load_teams(filename):
    """Load teams CSV. Accepts either an absolute path or a filename inside `data/`.
    Returns dict team_name -> list[player_names]
    """
    team_to_players = {}
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for team in reader.fieldnames:
            f.seek(0)
            next(reader)
            players = [row[team] for row in reader if row[team].strip()]
            team_to_players[team] = players
            f.seek(0)
            next(reader)
    return team_to_players


def load_games(filename, min_columns=5, pad_to=8):
    """Load games CSV rows. Returns list of rows (lists).
    Pads rows to `pad_to` length so downstream code can index safely.
    """
    games = []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= min_columns:
                while len(row) < pad_to:
                    row.append("")
                games.append(row)
    return games


def get_all_players(team_to_players):
    players = set()
    for plist in team_to_players.values():
        players.update([p for p in plist if p])
    return sorted(players)


def read_model_config(path):
    """Attempt to read a JSON model configuration file. Return dict or None."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def parse_time_maybe(ts_str):
    """Parse timestamp strings used in these scripts. Returns datetime or None."""
    for fmt in ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y'):
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    return None
