#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuner for TrueSkill initial sigma and beta.

This script wraps the existing `evaluate_params` function from
`train_trueskill_hyperparams.py` and runs Optuna to minimize walk-forward
average log-loss.

It writes a trials CSV and the best parameters to `best_trueskill_optuna.json`.
"""
import argparse
import csv
import json
import optuna
import pandas as pd
from train_trueskill_hyperparams import evaluate_params


def objective_factory(mu, teams_file, games_file, max_games, seed, sigma_min, sigma_max, beta_min, beta_max):
    def objective(trial):
        sigma = trial.suggest_float('sigma', sigma_min, sigma_max)
        beta = trial.suggest_float('beta', beta_min, beta_max)
        avg_logloss, count = evaluate_params(mu, sigma, beta, teams_file, games_file, max_games=max_games)
        # if no games evaluated (e.g., all ties), return a large loss so it's avoided
        if count == 0:
            return float('inf')
        return float(avg_logloss)
    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna tuner for TrueSkill hyperparameters')
    parser.add_argument('--teams', default='Sports Elo - Teams.csv')
    parser.add_argument('--games', default='Sports Elo - Games.csv')
    parser.add_argument('--mu', type=float, default=1000.0)
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--max-games', type=int, default=0, help='If >0, limit number of games processed')
    parser.add_argument('--out-csv', default='trueskill_optuna_trials.csv')
    parser.add_argument('--best-out', default='best_trueskill_optuna.json')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs for Optuna (set 1 for serial)')
    parser.add_argument('--sigma-min', type=float, default=200.0, help='Lower bound for sigma')
    parser.add_argument('--sigma-max', type=float, default=450.0, help='Upper bound for sigma')
    parser.add_argument('--beta-min', type=float, default=60.0, help='Lower bound for beta')
    parser.add_argument('--beta-max', type=float, default=300.0, help='Upper bound for beta')
    parser.add_argument('--auto-expand', action='store_true', help='If true, auto-expand bounds and rerun when best is near bounds')
    parser.add_argument('--expand-factor', type=float, default=2.0, help='Factor to expand bounds by when auto-expanding')
    parser.add_argument('--bound-threshold', type=float, default=0.01, help='Fractional threshold to consider a best-value near a bound')
    args = parser.parse_args()

    max_games = args.max_games if args.max_games > 0 else None

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    objective = objective_factory(args.mu, args.teams, args.games, max_games, args.seed,
                                 args.sigma_min, args.sigma_max, args.beta_min, args.beta_max)

    print(f"Running Optuna: {args.trials} trials (n_jobs={args.n_jobs})")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    # save trials dataframe to CSV
    df1 = study.trials_dataframe()
    # save best params
    best = study.best_trial
    best_info = {
        'params': best.params,
        'value': best.value,
        'number': best.number,
    }

    # If requested, check if best params lie very close to any bound and optionally rerun with expanded bounds
    rerun_info = None
    if args.auto_expand:
        near_lower_sigma = best.params.get('sigma', 0) <= args.sigma_min * (1.0 + args.bound_threshold)
        near_upper_sigma = best.params.get('sigma', 0) >= args.sigma_max * (1.0 - args.bound_threshold)
        near_lower_beta = best.params.get('beta', 0) <= args.beta_min * (1.0 + args.bound_threshold)
        near_upper_beta = best.params.get('beta', 0) >= args.beta_max * (1.0 - args.bound_threshold)

        if any((near_lower_sigma, near_upper_sigma, near_lower_beta, near_upper_beta)):
            print('Best trial is near a bound; auto-expanding search space and rerunning once')
            # compute new bounds
            sigma_min2, sigma_max2 = args.sigma_min, args.sigma_max
            beta_min2, beta_max2 = args.beta_min, args.beta_max
            if near_lower_sigma:
                sigma_min2 = max(1.0, args.sigma_min / args.expand_factor)
            if near_upper_sigma:
                sigma_max2 = args.sigma_max * args.expand_factor
            if near_lower_beta:
                beta_min2 = max(1.0, args.beta_min / args.expand_factor)
            if near_upper_beta:
                beta_max2 = args.beta_max * args.expand_factor

            # create a fresh study and run
            sampler2 = optuna.samplers.TPESampler(seed=args.seed + 1)
            study2 = optuna.create_study(direction='minimize', sampler=sampler2)
            objective2 = objective_factory(args.mu, args.teams, args.games, max_games, args.seed + 1,
                                           sigma_min2, sigma_max2, beta_min2, beta_max2)
            study2.optimize(objective2, n_trials=args.trials, n_jobs=1)
            df2 = study2.trials_dataframe()
            # combine dfs
            try:
                df = pd.concat([df1, df2], ignore_index=True)
            except Exception:
                df = df1
            # choose best among both studies
            best_candidates = [best, study2.best_trial]
            best_overall = min(best_candidates, key=lambda t: t.value)
            best_info = {
                'params': best_overall.params,
                'value': best_overall.value,
                'number': best_overall.number,
            }
            rerun_info = {
                'expanded_bounds': {
                    'sigma_min': sigma_min2,
                    'sigma_max': sigma_max2,
                    'beta_min': beta_min2,
                    'beta_max': beta_max2,
                },
                'second_best': {
                    'params': study2.best_trial.params,
                    'value': study2.best_trial.value,
                    'number': study2.best_trial.number,
                }
            }
        else:
            df = df1
    else:
        df = df1

    df.to_csv(args.out_csv, index=False)

    with open(args.best_out, 'w', encoding='utf-8') as f:
        json.dump(best_info, f, indent=2)

    print('Best trial:')
    print(json.dumps(best_info, indent=2))
    if rerun_info:
        print('\nAuto-expand run info:')
        print(json.dumps(rerun_info, indent=2))


if __name__ == '__main__':
    main()
