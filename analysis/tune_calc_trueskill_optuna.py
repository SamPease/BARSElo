#!/usr/bin/env python3
"""
Optuna wrapper tuned specifically for the `calculate_trueskill.py` defaults.
This computes search bounds as a percentage of the defaults defined in
`calculate_trueskill.py` and runs Optuna minimizing average log-loss.
"""
import argparse
import json
import optuna
import pandas as pd
from analysis.train_trueskill_hyperparams import evaluate_params
from calculate_trueskill import INITIAL_MU, INITIAL_SIGMA, INITIAL_BETA, INITIAL_TAU, DRAW_PROBABILITY


def objective_factory(mu, teams_file, games_file, max_games, seed,
                      sigma_min, sigma_max, beta_min, beta_max,
                      tau_min, tau_max, draw_min, draw_max):
    def objective(trial):
        sigma = trial.suggest_float('sigma', sigma_min, sigma_max)
        beta = trial.suggest_float('beta', beta_min, beta_max)
        tau = trial.suggest_float('tau', tau_min, tau_max)
        draw = trial.suggest_float('draw', draw_min, draw_max)
        avg_logloss, count = evaluate_params(mu, sigma, beta, teams_file, games_file,
                                            max_games=max_games, tau=tau, draw_probability=draw)
        if count == 0:
            return float('inf')
        return float(avg_logloss)
    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna tuner for calculate_trueskill defaults')
    parser.add_argument('--teams', default='Sports Elo - Teams.csv')
    parser.add_argument('--games', default='Sports Elo - Games.csv')
    parser.add_argument('--mu', type=float, default=INITIAL_MU)
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--max-games', type=int, default=0)
    parser.add_argument('--out-csv', default='analysis/trueskill_optuna_trials_calc.csv')
    parser.add_argument('--best-out', default='analysis/best_trueskill_optuna_calc.json')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--pct-low', type=float, default=0.6, help='Lower bound as fraction of default')
    parser.add_argument('--pct-high', type=float, default=1.4, help='Upper bound as fraction of default')
    args = parser.parse_args()

    max_games = args.max_games if args.max_games > 0 else None

    sigma_min = args.pct_low * INITIAL_SIGMA
    sigma_max = args.pct_high * INITIAL_SIGMA
    beta_min = args.pct_low * INITIAL_BETA
    beta_max = args.pct_high * INITIAL_BETA
    tau_min = args.pct_low * INITIAL_TAU
    tau_max = args.pct_high * INITIAL_TAU
    # draw probability must be in [0, 0.4999]
    draw_min = max(0.0, args.pct_low * DRAW_PROBABILITY)
    draw_max = min(0.4999, args.pct_high * DRAW_PROBABILITY)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    objective = objective_factory(args.mu, args.teams, args.games, max_games, args.seed,
                                 sigma_min, sigma_max, beta_min, beta_max,
                                 tau_min, tau_max, draw_min, draw_max)

    print(f"Running Optuna: {args.trials} trials (n_jobs={args.n_jobs})")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    df = study.trials_dataframe()
    best = study.best_trial
    best_info = {'params': best.params, 'value': best.value, 'number': best.number}

    df.to_csv(args.out_csv, index=False)
    with open(args.best_out, 'w', encoding='utf-8') as f:
        json.dump(best_info, f, indent=2)

    print('Best trial:')
    print(json.dumps(best_info, indent=2))


if __name__ == '__main__':
    main()
