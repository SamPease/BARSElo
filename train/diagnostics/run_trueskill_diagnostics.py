#!/usr/bin/env python3
"""Run diagnostics for the regular TrueSkill model.

Outputs (all under `train/diagnostics/`):
- `predictions.csv` : per-game predictions and NLLs
- `summary.json` : aggregated NLLs and logistic/regression summary
- `pred_prob_hist.png` : histogram of predicted win probabilities

This is designed to run without adding new Python deps beyond what's in `requirements.txt`.
"""
import os
import json
import csv
import math
from collections import defaultdict
from datetime import datetime

import sys
# make repo root importable so `from data import ...` and `from models import ...` work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from data import data_loader
from models.trueskill import TrueSkillModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DIAG_DIR = os.path.join(BASE_DIR, 'train', 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

GAMES_CSV = os.path.join(BASE_DIR, 'data', 'Sports Elo - Games.csv')
TEAMS_CSV = os.path.join(BASE_DIR, 'data', 'Sports Elo - Teams.csv')
BEST_PARAMS = os.path.join(BASE_DIR, 'train', 'trueskill_best_params.json')

# safety
EPS = 1e-9


def parse_time(s):
    t = data_loader.parse_time_maybe(s)
    return t


def find_team_players(team_to_players, team_name):
    # direct match
    if team_name in team_to_players:
        return team_to_players[team_name]
    # try case-insensitive
    lower = {k.lower(): k for k in team_to_players}
    if team_name.lower() in lower:
        return team_to_players[lower[team_name.lower()]]
    # try fuzzy: strip punctuation and whitespace
    key = ''.join(c.lower() for c in team_name if c.isalnum() or c.isspace()).strip()
    for k in team_to_players:
        k2 = ''.join(c.lower() for c in k if c.isalnum() or c.isspace()).strip()
        if k2 == key:
            return team_to_players[k]
    return []


def safe_logloss(p, y):
    p = max(EPS, min(1 - EPS, p))
    if y == 1:
        return -math.log(p)
    elif y == 0:
        return -math.log(1 - p)
    else:
        # draw treated as -log(0.5)
        return -math.log(0.5)


def auc_from_scores(scores, labels):
    # compute AUC using rank-sum method
    # labels: 1 for positive (team1 wins), 0 for negative
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    # rank all
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def fit_logistic_gd(x, y, lr=1e-3, n_iter=5000):
    # x: (n,) feature (skill_diff), y: {0,1}
    X = np.vstack([np.ones_like(x), x]).T  # intercept + x
    w = np.zeros(X.shape[1])
    for i in range(n_iter):
        z = X.dot(w)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T.dot(p - y) / len(y)
        w -= lr * grad
        if i % 500 == 0:
            ll = np.sum(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
    coef = float(w[1])
    intercept = float(w[0])
    # approximate se via observed Fisher information
    p = 1.0 / (1.0 + np.exp(-X.dot(w)))
    W = p * (1 - p)
    XtWX = (X.T * W).dot(X)
    try:
        cov = np.linalg.inv(XtWX / len(y))
        se = float(np.sqrt(cov[1, 1]))
    except Exception:
        se = None
    return {'coef': coef, 'intercept': intercept, 'se': se}


def main():
    # load games and teams
    games = data_loader.load_games(GAMES_CSV)
    team_to_players = data_loader.load_teams(TEAMS_CSV)

    # load best params
    try:
        with open(BEST_PARAMS, 'r', encoding='utf-8') as f:
            best = json.load(f)
            params = best.get('params', {})
    except Exception:
        params = {}

    mu = params.get('mu', 1000.0)
    sigma = params.get('sigma', None)
    beta = params.get('beta', None)
    tau = params.get('tau', None)
    draw_probability = params.get('draw_probability', None)

    model = TrueSkillModel(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability)

    # sort games by time
    def key_time(row):
        t = parse_time(row[0])
        return t or datetime.min

    games_sorted = sorted(games, key=key_time)

    # history counters for baselines
    team_counts = defaultdict(int)
    team_wins = defaultdict(int)

    rows_out = []
    probs = []
    skill_diffs = []
    labels = []
    nlls = []

    for row in games_sorted:
        # row layout: Time,Team 1,Score 1,Team 2,Score 2,Outcome Only,Tournament,Number of Players
        time = row[0]
        t1 = row[1].strip() if len(row) > 1 else ''
        s1 = row[2].strip() if len(row) > 2 else ''
        t2 = row[3].strip() if len(row) > 3 else ''
        s2 = row[4].strip() if len(row) > 4 else ''
        players_field = row[7].strip() if len(row) > 7 else ''

        try:
            s1i = int(s1)
            s2i = int(s2)
        except Exception:
            continue

        team1_players = find_team_players(team_to_players, t1)
        team2_players = find_team_players(team_to_players, t2)

        p_on = 8
        try:
            p_on = int(players_field) if players_field else 8
        except Exception:
            p_on = 8

        # compute current exposed skills
        team1_mu = sum((model.ratings[p].mu for p in team1_players if p in model.ratings)) if team1_players else 0.0
        team2_mu = sum((model.ratings[p].mu for p in team2_players if p in model.ratings)) if team2_players else 0.0
        skill_diff = team1_mu - team2_mu

        prob = model.predict_win_prob(team1_players, team2_players, players_on_court=p_on)

        # actual outcome
        if s1i > s2i:
            y = 1
        elif s1i < s2i:
            y = 0
        else:
            y = 0.5

        nll = safe_logloss(prob, y)

        rows_out.append({'Time': time, 'Team 1': t1, 'Team 2': t2, 'Score 1': s1i, 'Score 2': s2i, 'Players': p_on, 'Pred': prob, 'SkillDiff': skill_diff, 'NLL': nll, 'Outcome': y})

        probs.append(prob)
        skill_diffs.append(skill_diff)
        labels.append(1 if y == 1 else 0)
        nlls.append(nll)

        # update baselines
        team_counts[t1] += 1
        if y == 1:
            team_wins[t1] += 1

        # now update the model with the actual game
        model.update([time, t1, s1i, t2, s2i, None, None, p_on], team1_players, team2_players)

    # compute baselines
    total_games = len(labels)
    total_wins = sum(labels)
    global_empirical = float(total_wins) / total_games if total_games else 0.5

    baseline_nlls = {}
    # 50/50 baseline
    baseline_nlls['50_50_per_game_nll'] = -math.log(0.5)
    # global empirical baseline average NLL
    avg_global = 0.0
    for y in labels:
        if y == 1:
            avg_global += -math.log(max(EPS, global_empirical))
        else:
            avg_global += -math.log(max(EPS, 1 - global_empirical))
    avg_global = avg_global / total_games if total_games else None
    baseline_nlls['global_empirical_avg_nll'] = avg_global

    # per-team empirical baseline (use team1's empirical win rate if it has >=5 prior games)
    per_team_nll_total = 0.0
    per_team_count = 0
    team_history_counts = defaultdict(int)
    team_history_wins = defaultdict(int)

    for r in rows_out:
        t1 = r['Team 1']
        y = 1 if r['Outcome'] == 1 else 0
        if team_history_counts[t1] >= 5:
            p = team_history_wins[t1] / float(team_history_counts[t1])
            p = max(EPS, min(1 - EPS, p))
            per_team_nll_total += -math.log(p) if y == 1 else -math.log(1 - p)
            per_team_count += 1
        else:
            # fallback to global
            per_team_nll_total += -math.log(global_empirical) if y == 1 else -math.log(1 - global_empirical)
            per_team_count += 1
        team_history_counts[t1] += 1
        if y == 1:
            team_history_wins[t1] += 1

    baseline_nlls['per_team_empirical_avg_nll'] = per_team_nll_total / per_team_count if per_team_count else None

    # model average NLL
    model_avg_nll = sum(nlls) / len(nlls) if nlls else None

    # histogram of predicted probs
    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=20, range=(0, 1))
    plt.xlabel('Predicted win probability (team1)')
    plt.ylabel('Count')
    plt.title('Predicted probability histogram')
    hist_path = os.path.join(DIAG_DIR, 'pred_prob_hist.png')
    plt.savefig(hist_path)
    plt.close()

    # per-team match counts
    team_counts_list = sorted([(t, c, team_wins.get(t, 0)) for t, c in team_counts.items()], key=lambda x: -x[1])
    with open(os.path.join(DIAG_DIR, 'team_counts.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Team', 'Games', 'Wins'])
        w.writerows(team_counts_list)

    # predictions CSV
    with open(os.path.join(DIAG_DIR, 'predictions.csv'), 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Time', 'Team 1', 'Team 2', 'Score 1', 'Score 2', 'Players', 'Pred', 'SkillDiff', 'NLL', 'Outcome']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    # logistic regression fit of outcome ~ skill_diff
    x = np.array(skill_diffs)
    y = np.array(labels)
    if len(x) > 10 and len(np.unique(x)) > 1 and (y.sum() > 0) and (y.sum() < len(y)):
        logreg = fit_logistic_gd(x, y, lr=1e-3, n_iter=5000)
        auc = auc_from_scores(skill_diffs, labels)
    else:
        logreg = None
        auc = None

    summary = {
        'n_games': total_games,
        'model_avg_nll': model_avg_nll,
        'baseline_nlls': baseline_nlls,
        'best_params_used': params,
        'auc_skilldiff': auc,
        'logistic': logreg,
        'top_teams_by_games': team_counts_list[:30]
    }

    with open(os.path.join(DIAG_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Diagnostics complete')
    print('Saved:', os.path.join(DIAG_DIR, 'predictions.csv'))
    print('Saved:', os.path.join(DIAG_DIR, 'summary.json'))
    print('Saved:', hist_path)


if __name__ == '__main__':
    main()
