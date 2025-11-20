#!/usr/bin/env python3
"""Fit a regularized logistic regression with player and stint indicators.

Features:
- Player effects: +1 for each player on team A, -1 for each player on team B
- Stint effects: +1 for stintA, -1 for stintB where stint = team::year (uses game date)

This uses sparse matrices (scipy) and scikit-learn's LogisticRegression (saga solver)
with L2 regularization. Performs 5-fold stratified CV and writes coefficient summaries.

Assumptions / notes:
- Per-game rosters are not available; we use the roster from `data/Sports Elo - Teams.csv`
  as a proxy (same as `compute_ess.py`). If you have per-game rosters, adapt `get_players_for_game`.
- Tie games are skipped for binary logistic regression.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
from collections import defaultdict
import math

import numpy as np


def load_data_loader(repo_root: Path):
    path = repo_root / "data" / "data_loader.py"
    spec = importlib.util.spec_from_file_location("data_loader", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_name(s):
    return (s or "").strip().lower()


def build_feature_matrices(games_rows, teams_map, parse_time_fn):
    # normalize team map keys
    norm_to_team = {norm_name(k): k for k in teams_map.keys()}

    # build player index
    players = sorted({p for plist in teams_map.values() for p in plist if p})
    player_to_idx = {p: i for i, p in enumerate(players)}

    # build stint index on the fly
    stint_to_idx = {}

    rows_player = []  # list of (row_idx, col_idx, value)
    rows_stint = []
    y = []
    row_i = 0

    for row in games_rows:
        if len(row) < 5:
            continue
        time_str = (row[0] or "").strip()
        t1 = (row[1] or "").strip()
        s1 = (row[2] or "").strip()
        t2 = (row[3] or "").strip()
        s2 = (row[4] or "").strip()
        try:
            s1f = float(s1)
            s2f = float(s2)
        except Exception:
            continue

        # binary outcome: 1 if team1 wins, 0 if team2 wins. skip ties
        if math.isclose(s1f, s2f):
            continue
        outcome = 1 if s1f > s2f else 0

        n1 = norm_name(t1)
        n2 = norm_name(t2)
        team1_key = norm_to_team.get(n1)
        team2_key = norm_to_team.get(n2)
        if not team1_key or not team2_key:
            # skip games where teams are not in roster map
            continue

        # players for teams (use full roster as proxy)
        players_a = teams_map.get(team1_key, [])
        players_b = teams_map.get(team2_key, [])

        # player features: +1 for A, -1 for B
        for p in players_a:
            if p in player_to_idx:
                rows_player.append((row_i, player_to_idx[p], 1.0))
        for p in players_b:
            if p in player_to_idx:
                rows_player.append((row_i, player_to_idx[p], -1.0))

        # stint as team::year
        dt = None
        try:
            dt = parse_time_fn(time_str)
        except Exception:
            dt = None
        year = dt.year if dt is not None else 'NA'
        stint_a = f"{team1_key}::{year}"
        stint_b = f"{team2_key}::{year}"
        if stint_a not in stint_to_idx:
            stint_to_idx[stint_a] = len(stint_to_idx)
        if stint_b not in stint_to_idx:
            stint_to_idx[stint_b] = len(stint_to_idx)
        rows_stint.append((row_i, stint_to_idx[stint_a], 1.0))
        rows_stint.append((row_i, stint_to_idx[stint_b], -1.0))

        y.append(outcome)
        row_i += 1

    n_rows = row_i
    n_players = len(players)
    n_stints = len(stint_to_idx)

    # build sparse matrices
    from scipy import sparse

    if n_rows == 0:
        return None

    if rows_player:
        rpi, rpc, rpv = zip(*rows_player)
        Xp = sparse.csr_matrix((rpv, (rpi, rpc)), shape=(n_rows, n_players), dtype=float)
    else:
        Xp = sparse.csr_matrix((n_rows, n_players), dtype=float)

    if rows_stint:
        rsi, rsc, rsv = zip(*rows_stint)
        Xs = sparse.csr_matrix((rsv, (rsi, rsc)), shape=(n_rows, n_stints), dtype=float)
    else:
        Xs = sparse.csr_matrix((n_rows, n_stints), dtype=float)

    # combine
    X = sparse.hstack([Xp, Xs], format='csr')
    y = np.array(y, dtype=int)

    meta = {
        'players': players,
        'stints': [None] * n_stints
    }
    # invert stint_to_idx
    inv = {v: k for k, v in stint_to_idx.items()}
    for i in range(n_stints):
        meta['stints'][i] = inv.get(i)

    return X, y, meta


def fit_and_cv(X, y, C=1.0, n_splits=5, random_state=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    coefs = []
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr = X[train_idx]
        ytr = y[train_idx]
        Xte = X[test_idx]
        yte = y[test_idx]
        clf = LogisticRegression(penalty='l2', C=C, solver='saga', max_iter=2000, fit_intercept=False)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        acc = accuracy_score(yte, ypred)
        accs.append(acc)
        coefs.append(clf.coef_.ravel().copy())

    coefs = np.vstack(coefs)
    return coefs, accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', default='data/Sports Elo - Games.csv')
    p.add_argument('--teams', default='data/Sports Elo - Teams.csv')
    p.add_argument('--outdir', default='outputs')
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_loader = load_data_loader(repo_root)

    games = data_loader.load_games(args.games)
    teams = data_loader.load_teams(args.teams)

    print('Building feature matrices (players + stints)')
    res = build_feature_matrices(games, teams, data_loader.parse_time_maybe)
    if res is None:
        print('No training rows constructed. Exiting.')
        return
    X, y, meta = res
    print(f'Constructed X shape: {X.shape}; y len: {len(y)}')

    try:
        coefs, accs = fit_and_cv(X, y, C=args.C, n_splits=args.folds)
    except Exception as e:
        print('Error fitting model (scikit-learn required):', e)
        return

    mean_coef = coefs.mean(axis=0)
    std_coef = coefs.std(axis=0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_players = len(meta['players'])
    n_stints = len(meta['stints'])

    # write player coefficients
    with open(outdir / 'player_coefs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['player', 'coef_mean', 'coef_std'])
        for i, p in enumerate(meta['players']):
            w.writerow([p, f"{mean_coef[i]:.6f}", f"{std_coef[i]:.6f}"])

    # write stint coefficients
    with open(outdir / 'stint_coefs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stint', 'coef_mean', 'coef_std'])
        for j, s in enumerate(meta['stints']):
            idx = n_players + j
            w.writerow([s, f"{mean_coef[idx]:.6f}", f"{std_coef[idx]:.6f}"])

    print('CV accuracies:', accs)
    print(f'Mean CV accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'Wrote player_coefs.csv and stint_coefs.csv in {outdir}')


if __name__ == '__main__':
    import argparse
    main()
