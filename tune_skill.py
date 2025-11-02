#!/usr/bin/env python3
"""Tune LEARNING_RATE, TOURNAMENT_MULTIPLIER, and MARGIN_BETA for calculate_skill.py

Uses time-ordered one-step-ahead evaluation and optimizes log loss (probabilistic
prediction of match winner). Uses scikit-optimize if available, else falls back to
random search. Saves results to `tuning_results.csv` and best params to
`best_params.json` and prints the best params to stdout.
"""
import csv
import json
import math
import random
from datetime import datetime
from statistics import mean
import argparse

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False

import calculate_skill as cs


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def team_strength(team, ratings):
    # mirror calculate_skill.team_strength behaviour
    if not team:
        return cs.INITIAL_RATING
    return sum(ratings.get(p, cs.INITIAL_RATING) for p in team) / len(team)


def update_ratings_bt_local(team1, team2, score1, score2, ratings, lr, multiplier, margin_beta):
    """Local copy of the update function using explicit params."""
    margin = abs(score1 - score2)
    s1_strength = team_strength(team1, ratings)
    s2_strength = team_strength(team2, ratings)
    diff = (s1_strength - s2_strength) / 400.0
    diff += margin_beta * margin

    p1_win = logistic(diff)
    actual1 = 1 if score1 > score2 else (0.5 if score1 == score2 else 0)

    error = (actual1 - p1_win)

    # update each player's rating in place
    for p in team1:
        ratings[p] = ratings.get(p, cs.INITIAL_RATING) + lr * error * multiplier
    for p in team2:
        ratings[p] = ratings.get(p, cs.INITIAL_RATING) - lr * error * multiplier

    return ratings


def eval_params(lr, mult, mbeta, games, team_to_players, all_players):
    # sequential one-step-ahead evaluation
    ratings = {p: cs.INITIAL_RATING for p in all_players}
    eps = 1e-15
    losses = []
    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = row[:7]
        try:
            s1i = int(s1); s2i = int(s2)
        except Exception:
            continue
        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])

        # predict probability of team1 win using current ratings
        s1_strength = team_strength(team1, ratings)
        s2_strength = team_strength(team2, ratings)
        margin = abs(s1i - s2i)
        diff = (s1_strength - s2_strength) / 400.0 + mbeta * margin
        p1 = logistic(diff)
        # clamp
        p1 = min(max(p1, eps), 1 - eps)

        actual = 1.0 if s1i > s2i else (0.5 if s1i == s2i else 0.0)
        # log loss
        loss = -(actual * math.log(p1) + (1 - actual) * math.log(1 - p1))
        losses.append(loss)

        # update ratings using local update (respect tournament flag)
        multiplier = mult if tourney_flag.strip() else 1.0
        ratings = update_ratings_bt_local(team1, team2, s1i, s2i, ratings, lr=lr, multiplier=multiplier, margin_beta=mbeta)

    return mean(losses) if losses else float('inf')


def load_data():
    team_to_players = cs.load_teams(cs.TEAMS)
    games = cs.load_games(cs.GAMES)
    all_players = cs.get_all_players(team_to_players)
    return games, team_to_players, all_players


def random_search(n_trials, games, team_to_players, all_players, bounds):
    results = []
    for i in range(n_trials):
        lr = random.uniform(bounds['lr'][0], bounds['lr'][1])
        mult = random.uniform(bounds['mult'][0], bounds['mult'][1])
        mbeta = random.uniform(bounds['mbeta'][0], bounds['mbeta'][1])
        score = eval_params(lr, mult, mbeta, games, team_to_players, all_players)
        results.append({'lr': lr, 'mult': mult, 'mbeta': mbeta, 'score': score})
        if i % 10 == 0:
            print(f"Trial {i}/{n_trials}: lr={lr:.3f} mult={mult:.3f} mbeta={mbeta:.3f} score={score:.5f}")
    return results


def skopt_search(n_calls, games, team_to_players, all_players, bounds):
    space = [Real(bounds['lr'][0], bounds['lr'][1], name='lr'),
             Real(bounds['mult'][0], bounds['mult'][1], name='mult'),
             Real(bounds['mbeta'][0], bounds['mbeta'][1], name='mbeta')]

    def objective(x):
        lr, mult, mbeta = x
        return eval_params(lr, mult, mbeta, games, team_to_players, all_players)

    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    results = []
    for x, val in zip(res.x_iters, res.func_vals):
        results.append({'lr': float(x[0]), 'mult': float(x[1]), 'mbeta': float(x[2]), 'score': float(val)})
    return results, res.x, res.fun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=60, help='Number of trials / calls (default 60)')
    parser.add_argument('--out', default='tuning_results.csv')
    parser.add_argument('--best', default='best_params.json')
    args = parser.parse_args()

    games, team_to_players, all_players = load_data()

    bounds = {'lr': (1.0, 100.0), 'mult': (0.5, 3.0), 'mbeta': (0.0, 2.0)}

    if SKOPT_AVAILABLE:
        print('Running Bayesian optimization (scikit-optimize gp_minimize)')
        results, best_x, best_val = skopt_search(args.trials, games, team_to_players, all_players, bounds)
        best = {'lr': float(best_x[0]), 'mult': float(best_x[1]), 'mbeta': float(best_x[2]), 'score': float(best_val)}
    else:
        print('scikit-optimize not available — falling back to random search')
        results = random_search(args.trials, games, team_to_players, all_players, bounds)
        best = min(results, key=lambda r: r['score'])

    # write results CSV
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['lr', 'mult', 'mbeta', 'score'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # write best params
    with open(args.best, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2)

    print('\nBest params (minimized log loss):')
    print(f"LEARNING_RATE={best['lr']:.6f}")
    print(f"TOURNAMENT_MULTIPLIER={best['mult']:.6f}")
    print(f"MARGIN_BETA={best['mbeta']:.6f}")
    print(f"log loss={best['score']:.6f}")


if __name__ == '__main__':
    main()
