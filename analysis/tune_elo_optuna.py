#!/usr/bin/env python3
"""
Optuna tuner for ELO hyperparameters: `K_factor` and `tournament_multiplier`.

This script performs a walk-forward (online) evaluation that mirrors
`calculate_elo.py` logic: before each game it predicts team1 win probability
using current player ELOs, accumulates log-loss on non-tied games, then
updates ELOs using `update_elo_custom` with the candidate hyperparameters.

It writes a trials CSV and a best-params JSON.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
import optuna
import calculate_elo as ce


def evaluate_params(K, tournament_multiplier, teams_file, games_file, max_games=None):
    team_to_players = ce.load_teams(teams_file)
    games = ce.load_games(games_file)

    elos = defaultdict(lambda: ce.INITIAL_ELO)

    running_margin_sum = 0
    running_margin_count = 0

    total_logloss = 0.0
    count = 0

    for idx, row in enumerate(games):
        if max_games and idx >= max_games:
            break
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = row[:7]

        # Reproduce calculate_elo.py outcome handling and running margin logic
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

        # predict before update
        avg1 = ce.average_elo(team1, elos)
        avg2 = ce.average_elo(team2, elos)
        p1 = 1.0 / (1.0 + 10 ** ((avg2 - avg1) / 400.0))

        # accumulate log-loss for non-tied games
        if s1_i != s2_i:
            p1c = max(min(p1, 1.0 - 1e-15), 1e-15)
            if s1_i > s2_i:
                total_logloss += -math.log(p1c)
            else:
                total_logloss += -math.log(1.0 - p1c)
            count += 1

        # determine multiplier for this game
        multiplier = float(tournament_multiplier) if str(tourney_flag).strip() else 1.0

        # update elos using calculate_elo.update_elo_custom
        ce.update_elo_custom(team1, team2, s1_i, s2_i, elos, K=K, margin_override=margin, multiplier=multiplier)

    avg_logloss = float('nan') if count == 0 else (total_logloss / float(count))
    return avg_logloss, count


def objective_factory(teams_file, games_file, max_games, seed):
    def objective(trial):
        # Search ranges: K in [10, 500], tournament multiplier in [1.0, 4.0]
        K = trial.suggest_float('K', 10.0, 500.0)
        tm = trial.suggest_float('tournament_multiplier', 1.0, 4.0)
        avg_logloss, cnt = evaluate_params(K, tm, teams_file, games_file, max_games=max_games)
        if cnt == 0:
            return float('inf')
        return float(avg_logloss)
    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna tuner for ELO K and tournament multiplier')
    parser.add_argument('--teams', default=ce.TEAMS)
    parser.add_argument('--games', default=ce.GAMES)
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--max-games', type=int, default=0, help='If >0, limit number of games processed (for quick tests)')
    parser.add_argument('--out-csv', default='elo_optuna_trials.csv')
    parser.add_argument('--best-out', default='best_elo_optuna.json')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel trials for Optuna')
    args = parser.parse_args()

    max_games = args.max_games if args.max_games > 0 else None

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    objective = objective_factory(args.teams, args.games, max_games, args.seed)

    print(f"Running Optuna for ELO: {args.trials} trials (n_jobs={args.n_jobs})")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    # Save trials dataframe to CSV
    df = study.trials_dataframe()
    df.to_csv(args.out_csv, index=False)

    # Save best trial
    best = study.best_trial
    best_info = {'params': best.params, 'value': best.value, 'number': best.number}
    with open(args.best_out, 'w', encoding='utf-8') as f:
        json.dump(best_info, f, indent=2)

    print('Best trial:')
    print(json.dumps(best_info, indent=2))


if __name__ == '__main__':
    main()
