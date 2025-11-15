#!/usr/bin/env python3
"""
Compare TrueSkill and ELO using walk-forward evaluation and tuned hyperparameters.

Outputs:
 - `model_comparison_per_game.csv` : per-game predicted probabilities from both models and actual outcome
 - `model_comparison_summary.txt` : summary metrics (avg log-loss, Brier score, games evaluated)

Usage examples:
    python3 compare_models.py --trueskill-best best_trueskill_optuna_full.json \
        --elo-best best_elo_optuna_full.json

If best param files are missing, you can pass params manually.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime

import trueskill as ts

import train_trueskill_hyperparams as tts
import calculate_elo as ce


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clamp(p):
    return max(min(p, 1.0 - 1e-15), 1e-15)


def run_trueskill(mu, sigma, beta, teams_file, games_file):
    ts.setup(mu=mu, sigma=sigma, beta=beta)
    team_to_players = tts.load_teams(teams_file)
    games = tts.load_games(games_file)

    ratings = defaultdict(lambda: ts.Rating(mu=mu, sigma=sigma))

    per_game = []  # list of dicts
    total_logloss = 0.0
    total_brier = 0.0
    count = 0

    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = row[:8]
        s1_i, s2_i = tts.parse_score_fields(s1, s2, outcome_flag)
        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])
        if not team1 or not team2:
            continue
        team1_ratings = [ratings[p] for p in team1]
        team2_ratings = [ratings[p] for p in team2]
        p1 = tts.predicted_prob_team1_win(team1_ratings, team2_ratings, beta)
        # record only for non-tied games for log-loss/Brier
        if s1_i != s2_i:
            p1c = clamp(p1)
            actual = 1 if s1_i > s2_i else 0
            total_logloss += -math.log(p1c) if actual == 1 else -math.log(1 - p1c)
            total_brier += (p1 - actual) ** 2
            count += 1
        # update ratings
        try:
            players_total = int(players_field) if players_field.strip() else 8
        except Exception:
            players_total = 8
        w1 = tts.team_weight_list(team1, s1_i, players_total)
        w2 = tts.team_weight_list(team2, s2_i, players_total)
        if s1_i > s2_i:
            ranks = [0, 1]
        elif s1_i < s2_i:
            ranks = [1, 0]
        else:
            ranks = [0, 0]
        rated = ts.rate([team1_ratings, team2_ratings], ranks=ranks, weights=[w1, w2])
        for team_players, rated_team in ((team1, rated[0]), (team2, rated[1])):
            for p, new_rating in zip(team_players, rated_team):
                ratings[p] = new_rating
        per_game.append({'time': time, 't1': t1, 't2': t2, 's1': s1_i, 's2': s2_i, 'p_trueskill': p1})

    avg_logloss = float('nan') if count == 0 else (total_logloss / count)
    brier = float('nan') if count == 0 else (total_brier / count)
    return per_game, avg_logloss, brier, count


def run_elo(K, tournament_multiplier, teams_file, games_file):
    team_to_players = ce.load_teams(teams_file)
    games = ce.load_games(games_file)

    elos = defaultdict(lambda: ce.INITIAL_ELO)

    per_game = []
    total_logloss = 0.0
    total_brier = 0.0
    count = 0

    running_margin_sum = 0
    running_margin_count = 0

    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = row[:7]
        # outcome handling per calculate_elo.py
        if str(outcome_flag).strip():
            if running_margin_count > 0:
                margin = int(round(running_margin_sum / running_margin_count))
            else:
                margin = 1
            try:
                s1_i, s2_i = int(s1), int(s2)
            except Exception:
                s1_i, s2_i = 1, 0
            if s1_i > s2_i:
                s1_i, s2_i = margin, 0
            elif s2_i > s1_i:
                s1_i, s2_i = 0, margin
            else:
                s1_i = s2_i = 0
        else:
            try:
                s1_i, s2_i = int(s1), int(s2)
            except Exception:
                s1_i, s2_i = 0, 0
            margin = abs(s1_i - s2_i)
            if margin > 0:
                running_margin_sum += margin
                running_margin_count += 1

        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])
        if not team1 or not team2:
            continue
        avg1 = ce.average_elo(team1, elos)
        avg2 = ce.average_elo(team2, elos)
        p1 = 1.0 / (1.0 + 10 ** ((avg2 - avg1) / 400.0))
        if s1_i != s2_i:
            p1c = clamp(p1)
            actual = 1 if s1_i > s2_i else 0
            total_logloss += -math.log(p1c) if actual == 1 else -math.log(1 - p1c)
            total_brier += (p1 - actual) ** 2
            count += 1
        multiplier = float(tournament_multiplier) if str(tourney_flag).strip() else 1.0
        ce.update_elo_custom(team1, team2, s1_i, s2_i, elos, K=K, margin_override=margin, multiplier=multiplier)
        per_game.append({'time': time, 't1': t1, 't2': t2, 's1': s1_i, 's2': s2_i, 'p_elo': p1})

    avg_logloss = float('nan') if count == 0 else (total_logloss / count)
    brier = float('nan') if count == 0 else (total_brier / count)
    return per_game, avg_logloss, brier, count


def merge_and_write(per_ts, per_elo, out_csv='model_comparison_per_game.csv'):
    # Merge by chronological order (both lists are in games order). We'll align by index where both have entries for same game.
    # Safe merge: iterate over both lists and match by time+t1+t2; fallback to nearest index.
    rows = []
    # Build map from key to elo entry list (support duplicates by using list)
    elo_map = {}
    for e in per_elo:
        key = (e['time'], e['t1'], e['t2'])
        elo_map.setdefault(key, []).append(e)
    # Pop from lists as we match
    used_counts = {}
    for t in per_ts:
        key = (t['time'], t['t1'], t['t2'])
        elo_list = elo_map.get(key, [])
        idx = used_counts.get(key, 0)
        elo_entry = None
        if idx < len(elo_list):
            elo_entry = elo_list[idx]
            used_counts[key] = idx + 1
        else:
            # try reversed teams
            key2 = (t['time'], t['t2'], t['t1'])
            elo_list2 = elo_map.get(key2, [])
            idx2 = used_counts.get(key2, 0)
            if idx2 < len(elo_list2):
                elo_entry = elo_list2[idx2]
                used_counts[key2] = idx2 + 1
        row = {
            'time': t['time'],
            't1': t['t1'],
            't2': t['t2'],
            's1': t['s1'],
            's2': t['s2'],
            'p_trueskill': t.get('p_trueskill', ''),
            'p_elo': elo_entry.get('p_elo') if elo_entry else ''
        }
        rows.append(row)

    # Write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Time','Team1','Team2','Score1','Score2','P_TrueSkill(team1 wins)','P_ELO(team1 wins)'])
        for r in rows:
            w.writerow([r['time'], r['t1'], r['t2'], r['s1'], r['s2'], r['p_trueskill'], r['p_elo']])
    return rows


def main():
    parser = argparse.ArgumentParser(description='Compare TrueSkill and ELO on walk-forward log-loss and Brier')
    parser.add_argument('--teams', default=tts.DEFAULT_TEAMS)
    parser.add_argument('--games', default=tts.DEFAULT_GAMES)
    parser.add_argument('--trueskill-best', default='best_trueskill_optuna_full.json')
    parser.add_argument('--elo-best', default='best_elo_optuna_full.json')
    parser.add_argument('--out-csv', default='model_comparison_per_game.csv')
    parser.add_argument('--out-summary', default='model_comparison_summary.txt')
    args = parser.parse_args()

    # Load best params if available
    try:
        ts_best = load_json(args.trueskill_best)
        # ts_best may have .params or may be saved in different format; try flexibility
        if 'params' in ts_best:
            sigma = float(ts_best['params'].get('sigma'))
            beta = float(ts_best['params'].get('beta'))
            mu = float(ts_best.get('mu', 1000.0))
        else:
            # maybe direct dict
            sigma = float(ts_best.get('sigma', 333.3333))
            beta = float(ts_best.get('beta', 166.6667))
            mu = float(ts_best.get('mu', 1000.0))
    except Exception:
        # fallback defaults
        mu = 1000.0
        sigma = 333.3333333333333
        beta = 166.6666666666667

    try:
        elo_best = load_json(args.elo_best)
        if 'params' in elo_best:
            K = float(elo_best['params'].get('K'))
            tournament_multiplier = float(elo_best['params'].get('tournament_multiplier'))
        else:
            K = float(elo_best.get('K', ce.K_FACTOR))
            tournament_multiplier = float(elo_best.get('tournament_multiplier', ce.TOURNAMENT_MULTIPLIER))
    except Exception:
        K = ce.K_FACTOR
        tournament_multiplier = ce.TOURNAMENT_MULTIPLIER

    print('Running TrueSkill evaluation with:', {'mu': mu, 'sigma': sigma, 'beta': beta})
    per_ts, ts_logloss, ts_brier, ts_count = run_trueskill(mu, sigma, beta, args.teams, args.games)
    print('Running ELO evaluation with:', {'K': K, 'tournament_multiplier': tournament_multiplier})
    per_elo, elo_logloss, elo_brier, elo_count = run_elo(K, tournament_multiplier, args.teams, args.games)

    rows = merge_and_write(per_ts, per_elo, out_csv=args.out_csv)

    # Write summary
    with open(args.out_summary, 'w', encoding='utf-8') as f:
        f.write('Model comparison summary\n')
        f.write('-----------------------\n')
        f.write(f'TrueSkill params: mu={mu}, sigma={sigma}, beta={beta}\n')
        f.write(f'Evaluated games (TrueSkill): {ts_count}\n')
        f.write(f'TrueSkill avg log-loss: {ts_logloss}\n')
        f.write(f'TrueSkill Brier score: {ts_brier}\n')
        f.write('\n')
        f.write(f'ELO params: K={K}, tournament_multiplier={tournament_multiplier}\n')
        f.write(f'Evaluated games (ELO): {elo_count}\n')
        f.write(f'ELO avg log-loss: {elo_logloss}\n')
        f.write(f'ELO Brier score: {elo_brier}\n')
        f.write('\n')
        f.write('Notes: Log-loss and Brier computed over non-tied games only.\n')

    print('Wrote per-game CSV to', args.out_csv)
    print('Wrote summary to', args.out_summary)


if __name__ == '__main__':
    main()
