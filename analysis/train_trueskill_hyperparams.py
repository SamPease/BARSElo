#!/usr/bin/env python3
import csv
import math
import argparse
from collections import defaultdict
from datetime import datetime
import trueskill as ts

# This script performs a grid search over initial_sigma and initial_beta
# It evaluates each pair by running a walk-forward rating update over the
# games and computing average log-loss of predicted winners.

DEFAULT_TEAMS = 'Sports Elo - Teams.csv'
DEFAULT_GAMES = 'Sports Elo - Games.csv'


def load_teams(filename):
    team_to_players = {}
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for team in reader.fieldnames:
            f.seek(0)
            next(reader)  # skip header
            players = [row[team] for row in reader if row[team].strip()]
            team_to_players[team] = players
            f.seek(0)
            next(reader)
    return team_to_players


def load_games(filename):
    games = []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 5:
                # ensure at least 8 columns (added: number of players)
                while len(row) < 8:
                    row.append("")
                games.append(row)
    return games


def parse_score_fields(s1, s2, outcome_flag):
    # preserve current default behavior: if outcome_flag is truthy
    # default to 2:0 for the winner, or 0:0 for a tie
    if str(outcome_flag).strip():
        try:
            s1_i, s2_i = int(s1), int(s2)
        except Exception:
            s1_i, s2_i = 2, 0
        if s1_i > s2_i:
            return 2, 0
        elif s2_i > s1_i:
            return 0, 2
        else:
            return 0, 0
    else:
        try:
            return int(s1), int(s2)
        except Exception:
            return 0, 0


def team_weight_list(team_players, score, players_total):
    size = len(team_players)
    if size == 0:
        return []
    factor = float(score) / 5.0
    mult = float(players_total) / float(size)
    weight = factor * mult
    return [weight for _ in team_players]


def predicted_prob_team1_win(team1_ratings, team2_ratings, beta):
    # Use team-average performance so teams with equal per-player mu tie
    # regardless of team size. Team mean mu is the average of member mus.
    if not team1_ratings or not team2_ratings:
        return 0.5

    n1 = len(team1_ratings)
    n2 = len(team2_ratings)

    mu1 = sum(r.mu for r in team1_ratings) / float(n1)
    mu2 = sum(r.mu for r in team2_ratings) / float(n2)

    # Variance of the team mean: sum(individual variance + beta^2) / n^2
    var1 = sum((r.sigma ** 2) + (beta ** 2) for r in team1_ratings) / (n1 ** 2)
    var2 = sum((r.sigma ** 2) + (beta ** 2) for r in team2_ratings) / (n2 ** 2)

    var = var1 + var2
    if var <= 0:
        return 0.5
    z = (mu1 - mu2) / math.sqrt(var)
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def evaluate_params(mu, sigma, beta, teams_file, games_file, max_games=None):
    # Setup trueskill env
    ts.setup(mu=mu, sigma=sigma, beta=beta)

    team_to_players = load_teams(teams_file)
    games = load_games(games_file)

    # Initialize ratings
    ratings = defaultdict(lambda: ts.Rating(mu=mu, sigma=sigma))

    total_logloss = 0.0
    count = 0

    for idx, row in enumerate(games):
        if max_games and idx >= max_games:
            break
        time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = row[:8]
        s1_i, s2_i = parse_score_fields(s1, s2, outcome_flag)

        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])

        if not team1 or not team2:
            continue

        # compute predicted prob before updating
        team1_ratings = [ratings[p] for p in team1]
        team2_ratings = [ratings[p] for p in team2]

        p1 = predicted_prob_team1_win(team1_ratings, team2_ratings, beta)

        # Evaluate only non-tied games (ties are ambiguous for win prob)
        if s1_i != s2_i:
            # clamp p1 to avoid log(0)
            p1 = max(min(p1, 1.0 - 1e-15), 1e-15)
            if s1_i > s2_i:
                total_logloss += -math.log(p1)
            else:
                total_logloss += -math.log(1.0 - p1)
            count += 1

        # now update ratings using partial-play weights
        try:
            players_total = int(players_field) if players_field.strip() else 8
        except Exception:
            players_total = 8

        w1 = team_weight_list(team1, s1_i, players_total)
        w2 = team_weight_list(team2, s2_i, players_total)

        # Determine ranks for trueskill.rate
        if s1_i > s2_i:
            ranks = [0, 1]
        elif s1_i < s2_i:
            ranks = [1, 0]
        else:
            ranks = [0, 0]

        # Use ts.rate to update ratings
        rated = ts.rate([team1_ratings, team2_ratings], ranks=ranks, weights=[w1, w2])
        for team_players, rated_team in ((team1, rated[0]), (team2, rated[1])):
            for p, new_rating in zip(team_players, rated_team):
                ratings[p] = new_rating

    avg_logloss = float('nan') if count == 0 else (total_logloss / float(count))
    return avg_logloss, count


def make_range(start, stop, steps):
    if steps == 1:
        return [start]
    step = (stop - start) / float(steps - 1)
    return [start + i * step for i in range(steps)]


def main():
    parser = argparse.ArgumentParser(description='Grid-search initial_sigma and initial_beta for TrueSkill')
    parser.add_argument('--teams', default=DEFAULT_TEAMS)
    parser.add_argument('--games', default=DEFAULT_GAMES)
    parser.add_argument('--mu', type=float, default=1000.0)
    parser.add_argument('--sigma-min', type=float, default=200.0)
    parser.add_argument('--sigma-max', type=float, default=400.0)
    parser.add_argument('--sigma-steps', type=int, default=5)
    parser.add_argument('--beta-min', type=float, default=80.0)
    parser.add_argument('--beta-max', type=float, default=200.0)
    parser.add_argument('--beta-steps', type=int, default=5)
    parser.add_argument('--max-games', type=int, default=0, help='If >0, limit number of games processed (for quick tests)')
    parser.add_argument('--out-csv', default='trueskill_hyperparam_results.csv')
    args = parser.parse_args()

    sigma_list = make_range(args.sigma_min, args.sigma_max, max(1, args.sigma_steps))
    beta_list = make_range(args.beta_min, args.beta_max, max(1, args.beta_steps))

    results = []
    max_games = args.max_games if args.max_games > 0 else None

    print(f"Running grid: {len(sigma_list)} sigma x {len(beta_list)} beta = {len(sigma_list) * len(beta_list)} runs")

    for sigma in sigma_list:
        for beta in beta_list:
            avg_logloss, count = evaluate_params(args.mu, sigma, beta, args.teams, args.games, max_games=max_games)
            print(f"sigma={sigma:.3f}, beta={beta:.3f} -> avg_logloss={avg_logloss}, games_evaluated={count}")
            results.append((sigma, beta, avg_logloss, count))

    # write results
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sigma', 'beta', 'avg_logloss', 'games_evaluated'])
        for row in results:
            writer.writerow(row)

    # print best
    valid_results = [r for r in results if not math.isnan(r[2])]
    if valid_results:
        best = min(valid_results, key=lambda x: x[2])
        print(f"Best params: sigma={best[0]:.3f}, beta={best[1]:.3f}, avg_logloss={best[2]:.6f}, games_evaluated={best[3]}")
    else:
        print("No valid evaluation runs (no non-tied games were evaluated?).")


if __name__ == '__main__':
    main()
