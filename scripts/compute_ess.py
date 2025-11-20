#!/usr/bin/env python3
"""Compute per-player Effective Sample Size (ESS) from games and teams CSVs.

Assumptions:
- Per-game rosters are not available in the dataset. We therefore treat membership in
  `Sports Elo - Teams.csv` as an indicator that a player may appear for that team in any game.
  This script builds a sequence of team outcomes for each player by including every game
  for the team(s) they appear on and marking win=1, loss=0, tie=0.5.

Outputs:
- `outputs/ess_by_player.csv` with columns: player, games_played, ess, ess_ratio
- `outputs/ess_hist.png` histogram of ESS distribution (if matplotlib available).
"""
from pathlib import Path
import argparse
import csv
import math
import statistics
import importlib.util
import os
from collections import defaultdict

import numpy as np


def load_data_loader(repo_root: Path):
    # dynamically load data/data_loader.py as module `data_loader`
    path = repo_root / "data" / "data_loader.py"
    spec = importlib.util.spec_from_file_location("data_loader", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def effective_sample_size(series, max_lag=50):
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n <= 1:
        return n
    x = x - x.mean()
    acov = np.array([np.sum(x[:n-k] * x[k:]) / (n - k) for k in range(0, min(max_lag, n-1) + 1)])
    acor = acov / acov[0]
    rho_sum = 0.0
    for k in range(1, len(acor)):
        if acor[k] <= 0:
            break
        rho_sum += acor[k]
    ess = n / (1 + 2 * rho_sum)
    return float(ess)


def norm_name(s):
    return (s or "").strip().lower()


def build_player_series(games_rows, teams_map, parse_time_fn):
    """Return dict player -> list[outcomes], where outcomes are 1=win,0=loss,0.5=tie.

    games_rows: list of lists as returned by data_loader.load_games
    teams_map: dict team_name -> list[player]
    parse_time_fn: function to parse time strings into datetimes (may return None)
    """
    # build normalized team lookup
    norm_to_team = {norm_name(k): k for k in teams_map.keys()}

    # Parse times and assemble structured games
    structured = []
    for row in games_rows:
        if len(row) < 5:
            continue
        time_str = (row[0] or "").strip()
        t1 = (row[1] or "").strip()
        s1 = (row[2] or "").strip()
        t2 = (row[3] or "").strip()
        s2 = (row[4] or "").strip()
        try:
            s1f = float(s1) if s1 != "" else math.nan
        except Exception:
            s1f = math.nan
        try:
            s2f = float(s2) if s2 != "" else math.nan
        except Exception:
            s2f = math.nan

        # skip games without numeric scores
        if math.isnan(s1f) or math.isnan(s2f):
            continue

        dt = None
        try:
            dt = parse_time_fn(time_str)
        except Exception:
            dt = None

        structured.append((dt, time_str, t1, s1f, t2, s2f))

    # sort by datetime when available, otherwise keep original order for unparsable times
    structured.sort(key=lambda x: (0, x[0]) if x[0] is not None else (1, x[1]))

    player_series = defaultdict(list)
    unmatched_teams = set()

    for dt, time_str, t1, s1f, t2, s2f in structured:
        n1 = norm_name(t1)
        n2 = norm_name(t2)
        team1_key = norm_to_team.get(n1)
        team2_key = norm_to_team.get(n2)
        if not team1_key:
            unmatched_teams.add(t1)
            continue
        if not team2_key:
            unmatched_teams.add(t2)
            continue

        if s1f > s2f:
            v1, v2 = 1.0, 0.0
        elif s1f < s2f:
            v1, v2 = 0.0, 1.0
        else:
            v1, v2 = 0.5, 0.5

        for p in teams_map.get(team1_key, []):
            player_series[p].append(v1)
        for p in teams_map.get(team2_key, []):
            player_series[p].append(v2)

    return player_series, unmatched_teams


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', default='data/Sports Elo - Games.csv')
    p.add_argument('--teams', default='data/Sports Elo - Teams.csv')
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--max-lag', type=int, default=50)
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_loader = load_data_loader(repo_root)

    games_path = Path(args.games)
    teams_path = Path(args.teams)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    games = data_loader.load_games(str(games_path))
    teams_map = data_loader.load_teams(str(teams_path))

    player_series, unmatched = build_player_series(games, teams_map, data_loader.parse_time_maybe)

    if unmatched:
        print(f"Warning: {len(unmatched)} game team names didn't match any team in teams CSV. First examples:\n  " + "\n  ".join(list(unmatched)[:10]))

    results = []
    ess_values = []
    games_counts = []
    for player, series in player_series.items():
        n = len(series)
        if n == 0:
            continue
        ess = effective_sample_size(series, max_lag=args.max_lag)
        ratio = ess / n if n > 0 else 0.0
        results.append((player, n, ess, ratio))
        ess_values.append(ess)
        games_counts.append(n)

    out_csv = outdir / 'ess_by_player.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['player', 'games_played', 'ess', 'ess_ratio'])
        for r in sorted(results, key=lambda x: x[1], reverse=True):
            writer.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}"])

    if results:
        median_ess = statistics.median(ess_values)
        median_games = statistics.median(games_counts)
        many_players = sum(1 for _,n,ess,_ in results if n >= 20)
        low_ess_many = sum(1 for _,n,ess,_ in results if n >= 20 and ess < 10)
        print(f"Wrote per-player ESS to: {out_csv}")
        print(f"Players with >=20 games: {many_players}; of those, ESS<10: {low_ess_many}")
        print(f"Median games per player: {median_games:.1f}; median ESS: {median_ess:.2f}")
    else:
        print("No players found (no team membership or no scored games).")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            plt.hist(ess_values, bins=40)
            plt.xlabel('ESS')
            plt.ylabel('Number of players')
            plt.title('Distribution of per-player ESS')
            imgp = outdir / 'ess_hist.png'
            plt.tight_layout()
            plt.savefig(imgp)
            print(f"Saved histogram to {imgp}")
        except Exception as e:
            print(f"Plot skipped: {e}")


if __name__ == '__main__':
    main()
