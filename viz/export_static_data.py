#!/usr/bin/env python3
"""
Export ELO data to JSON for static website.

Run this script whenever your data changes to regenerate the static site data:
    python viz/export_static_data.py

This creates static-site/data/elo_data.json containing all players, teams, 
games, and ELO history needed for the static website.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import from data/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the existing loader functions from dash_app_elo
import pandas as pd
import numpy as np
from data.data_loader import parse_time_maybe

# Constants
ELO_RESULTS = 'viz/bt_uncert_results.csv'
TEAMS_FILE = 'data/Sports Elo - Teams.csv'
GAMES_FILE = 'data/Sports Elo - Games.csv'
INITIAL_ELO = 1000
OUTPUT_DIR = 'static-site/data'
OUTPUT_FILE = 'elo_data.json'


def load_and_export():
    """Load all data and export to JSON."""
    
    # Load ELO results
    print("Loading ELO results...")
    df = pd.read_csv(ELO_RESULTS)
    # Ensure there is a 'Time' column and parse it robustly
    if 'Time' not in df.columns:
        df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
    # Use our helper to parse common timestamp formats
    df['Time'] = df['Time'].apply(lambda x: parse_time_maybe(x) if isinstance(x, str) else x)
    # Fallback: try pandas parsing for anything still unparsed
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    # Drop rows where time couldn't be parsed
    df = df[df['Time'].notna()].sort_values('Time')
    
    players = [c for c in df.columns if c != 'Time']
    elo_df = df.set_index('Time')
    
    # Export ELO history: timestamps and per-player arrays
    timestamps = [ts.isoformat() for ts in elo_df.index]
    elo_history = {
        'timestamps': timestamps,
        'players': {}
    }
    for player in players:
        elo_history['players'][player] = [
            None if pd.isna(v) else round(float(v), 2) 
            for v in elo_df[player].values
        ]
    
    # Load teams
    print("Loading teams...")
    team_to_players = {}
    with open(TEAMS_FILE, newline='', encoding='utf-8') as f:
        import csv
        reader = csv.DictReader(f)
        teams = reader.fieldnames
        team_to_players = {t: [] for t in teams}
        for row in reader:
            for t in teams:
                v = row.get(t, '')
                if v and v.strip():
                    team_to_players[t].append(v.strip())
    
    # Load games
    print("Loading games...")
    games = []
    games_with_scores = []
    with open(GAMES_FILE, newline='', encoding='utf-8') as f:
        import csv
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            time_str = row[0].strip()
            t1 = row[1].strip() if len(row) > 1 else ''
            s1 = row[2].strip() if len(row) > 2 else None
            t2 = row[3].strip() if len(row) > 3 else ''
            s2 = row[4].strip() if len(row) > 4 else None
            outcome_only = row[5].strip() if len(row) > 5 else ''
            tournament_flag = row[6].strip() if len(row) > 6 else ''
            is_outcome_only = outcome_only == '1'
            is_tournament = tournament_flag == '1'
            
            # Parse time using shared helper
            dt = parse_time_maybe(time_str)
            if dt is None:
                continue
            
            if dt:
                games.append({
                    'datetime': dt.isoformat(),
                    'team1': t1,
                    'team2': t2,
                    'outcome_only': is_outcome_only,
                    'tournament': is_tournament
                })
                try:
                    s1i = int(s1) if s1 not in (None, '') else None
                    s2i = int(s2) if s2 not in (None, '') else None
                    if s1i is not None and s2i is not None:
                        games_with_scores.append({
                            'datetime': dt.isoformat(),
                            'team1': t1,
                            'score1': s1i,
                            'team2': t2,
                            'score2': s2i,
                            'outcome_only': is_outcome_only,
                            'tournament': is_tournament
                        })
                except Exception:
                    pass
    
    # Compute player stats
    print("Computing player stats...")
    all_players = set(players)
    for members in team_to_players.values():
        all_players.update(members)
    
    player_stats = {}
    for p in all_players:
        stats = {
            'current_elo': None,
            'peak_elo': None,
            'min_elo': None,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'games': 0,
            'last_played': None,
            'teams': [t for t, members in team_to_players.items() if p in members]
        }
        
        if p in elo_df.columns:
            s = elo_df[p]
            try:
                stats['current_elo'] = round(float(s.iloc[-1]), 2)
                stats['peak_elo'] = round(float(s.max()), 2)
                stats['min_elo'] = round(float(s.min()), 2)
            except Exception:
                pass
        
        player_stats[p] = stats
    
    # Add game records to player stats
    for game in games_with_scores:
        dt = game['datetime']
        t1, s1, t2, s2 = game['team1'], game['score1'], game['team2'], game['score2']
        
        m1 = team_to_players.get(t1, [])
        m2 = team_to_players.get(t2, [])
        
        for p in m1:
            if p in player_stats:
                player_stats[p]['games'] += 1
                if s1 > s2:
                    player_stats[p]['wins'] += 1
                elif s1 < s2:
                    player_stats[p]['losses'] += 1
                else:
                    player_stats[p]['draws'] += 1
                
                if not player_stats[p]['last_played'] or dt > player_stats[p]['last_played']:
                    player_stats[p]['last_played'] = dt
        
        for p in m2:
            if p in player_stats:
                player_stats[p]['games'] += 1
                if s2 > s1:
                    player_stats[p]['wins'] += 1
                elif s2 < s1:
                    player_stats[p]['losses'] += 1
                else:
                    player_stats[p]['draws'] += 1
                
                if not player_stats[p]['last_played'] or dt > player_stats[p]['last_played']:
                    player_stats[p]['last_played'] = dt
    
    # Compute team stats
    print("Computing team stats...")
    team_stats = {}
    for team, members in team_to_players.items():
        valid_members = [m for m in members if m in elo_df.columns]
        
        # Compute team ELO history (average of members)
        if valid_members:
            team_elo_array = elo_df[valid_members].mean(axis=1).values
            team_elo_list = [
                None if pd.isna(v) else round(float(v), 2) 
                for v in team_elo_array
            ]
        else:
            team_elo_list = [INITIAL_ELO] * len(timestamps)
        
        stats = {
            'members': members,
            'elo_history': team_elo_list,
            'current_elo': team_elo_list[-1] if team_elo_list else None,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'games': 0,
            'first_game': None,
            'last_game': None
        }
        
        team_stats[team] = stats
    
    # Add game records to team stats
    for game in games_with_scores:
        dt = game['datetime']
        t1, s1, t2, s2 = game['team1'], game['score1'], game['team2'], game['score2']
        
        for t in [t1, t2]:
            if t in team_stats:
                team_stats[t]['games'] += 1
                if not team_stats[t]['first_game'] or dt < team_stats[t]['first_game']:
                    team_stats[t]['first_game'] = dt
                if not team_stats[t]['last_game'] or dt > team_stats[t]['last_game']:
                    team_stats[t]['last_game'] = dt
        
        if t1 in team_stats:
            if s1 > s2:
                team_stats[t1]['wins'] += 1
            elif s1 < s2:
                team_stats[t1]['losses'] += 1
            else:
                team_stats[t1]['draws'] += 1
        
        if t2 in team_stats:
            if s2 > s1:
                team_stats[t2]['wins'] += 1
            elif s2 < s1:
                team_stats[t2]['losses'] += 1
            else:
                team_stats[t2]['draws'] += 1
    
    # Assemble final data structure
    data = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'initial_elo': INITIAL_ELO,
            'num_players': len(all_players),
            'num_teams': len(team_to_players),
            'num_games': len(games_with_scores)
        },
        'elo_history': elo_history,
        'players': player_stats,
        'teams': team_stats,
        'games': games,
        'games_with_scores': games_with_scores
    }
    
    # Write to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"Writing data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    file_size = os.path.getsize(output_path)
    print(f"✓ Exported {file_size:,} bytes")
    print(f"✓ {len(all_players)} players, {len(team_to_players)} teams, {len(games_with_scores)} games")
    print(f"\nStatic site data ready at: {output_path}")
    print("\nNext steps:")
    print("  1. Open static-site/index.html in a browser to test locally")
    print("  2. Upload the static-site/ folder to your web host")
    print("  3. Re-run this script whenever your data updates")


if __name__ == '__main__':
    load_and_export()
