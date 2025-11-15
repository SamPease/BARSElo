#!/usr/bin/env python3
"""
Evaluator for TrueSkill hyperparameters for the MOV (margin-of-victory)
calculation. This performs `sim_count` randomized simulations per hyperparameter
setting where repeated `ts.rate` calls (representing MOV) are applied in a
random interleaving order, matching the behavior in
`calculate_trueskill_mov.py` (weights by team sizes).

Returns average log-loss (negative log-likelihood) across simulations.
"""
import csv
import math
import random
import argparse
from collections import defaultdict
import trueskill as ts


DEFAULT_TEAMS = 'Sports Elo - Teams.csv'
DEFAULT_GAMES = 'Sports Elo - Games.csv'


def load_teams(filename):
    team_to_players = {}
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for team in reader.fieldnames:
            f.seek(0)
            next(reader)
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
                while len(row) < 8:
                    row.append("")
                games.append(row)
    return games


def parse_score_fields(s1, s2, outcome_flag):
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


def predicted_prob_team1_win(team1_ratings, team2_ratings, beta):
    if not team1_ratings or not team2_ratings:
        return 0.5

    n1 = len(team1_ratings)
    n2 = len(team2_ratings)

    mu1 = sum(r.mu for r in team1_ratings) / float(n1)
    mu2 = sum(r.mu for r in team2_ratings) / float(n2)

    var1 = sum((r.sigma ** 2) + (beta ** 2) for r in team1_ratings) / (n1 ** 2)
    var2 = sum((r.sigma ** 2) + (beta ** 2) for r in team2_ratings) / (n2 ** 2)

    var = var1 + var2
    if var <= 0:
        return 0.5
    z = (mu1 - mu2) / math.sqrt(var)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def evaluate_params_mov(mu, sigma, beta, teams_file, games_file, max_games=None, sim_count=5, seed=1234, tau=None, draw_probability=None):
    """
    Run `sim_count` independent simulations of updating ratings using
    the repeated-rate MOV approach from `calculate_trueskill_mov.py`.

    Returns (avg_logloss, count) where `avg_logloss` is averaged across
    simulations and `count` is the number of non-tied games per simulation.
    """
    random.seed(seed)

    team_to_players = load_teams(teams_file)
    games = load_games(games_file)

    total_logloss = 0.0
    total_count = 0

    # run multiple simulations and average
    for sim in range(sim_count):
        # Default tau/draw if not provided
        tau_run = tau if tau is not None else float(sigma) / 100.0
        draw_run = draw_probability if draw_probability is not None else 0.123
        ts.setup(mu=mu, sigma=sigma, beta=beta, tau=tau_run, draw_probability=draw_run)
        ratings = defaultdict(lambda: ts.Rating(mu=mu, sigma=sigma))

        sim_logloss = 0.0
        sim_count_games = 0

        for idx, row in enumerate(games):
            if max_games and idx >= max_games:
                break
            time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = row[:8]
            s1_i, s2_i = parse_score_fields(s1, s2, outcome_flag)

            team1 = team_to_players.get(t1, [])
            team2 = team_to_players.get(t2, [])

            if not team1 or not team2:
                continue

            team1_ratings = [ratings[p] for p in team1]
            team2_ratings = [ratings[p] for p in team2]

            # predicted probability before applying updates for this game
            p1 = predicted_prob_team1_win(team1_ratings, team2_ratings, beta)

            # Evaluate only non-tied games
            if s1_i != s2_i:
                p1 = max(min(p1, 1.0 - 1e-15), 1e-15)
                if s1_i > s2_i:
                    sim_logloss += -math.log(p1)
                else:
                    sim_logloss += -math.log(1.0 - p1)
                sim_count_games += 1

            # apply MOV updates by repeating ts.rate calls
            try:
                players_total = int(players_field) if players_field.strip() else 8
            except Exception:
                players_total = 8

            def team_weights_by_players(team_players):
                size = len(team_players)
                if size == 0:
                    return []
                mult = max(1.0, float(players_total) / float(size))
                return [mult for _ in team_players]

            weights = [team_weights_by_players(team1), team_weights_by_players(team2)]

            try:
                calls_a = max(0, int(s1_i))
            except Exception:
                calls_a = 0
            try:
                calls_b = max(0, int(s2_i))
            except Exception:
                calls_b = 0

            # create a randomized interleaving of win events
            events = [0] * calls_a + [1] * calls_b
            random.shuffle(events)

            for ev in events:
                # refresh rating objects for team members
                team1_ratings = [ratings[p] for p in team1]
                team2_ratings = [ratings[p] for p in team2]
                if ev == 0:
                    rated = ts.rate([team1_ratings, team2_ratings], ranks=[0, 1], weights=weights)
                else:
                    rated = ts.rate([team1_ratings, team2_ratings], ranks=[1, 0], weights=weights)
                for team_players, rated_team in ((team1, rated[0]), (team2, rated[1])):
                    for p, new_rating in zip(team_players, rated_team):
                        ratings[p] = new_rating

        total_logloss += sim_logloss
        total_count += sim_count_games

    # average across simulations
    if total_count == 0:
        return float('nan'), 0
    avg_logloss = total_logloss / float(total_count)
    # return count per single sim to match convention
    return avg_logloss, (total_count // max(1, sim_count))


def main():
    parser = argparse.ArgumentParser(description='Evaluate TrueSkill params with MOV simulations')
    parser.add_argument('--teams', default=DEFAULT_TEAMS)
    parser.add_argument('--games', default=DEFAULT_GAMES)
    parser.add_argument('--mu', type=float, default=1000.0)
    parser.add_argument('--sigma', type=float, default=333.3333333)
    parser.add_argument('--beta', type=float, default=372.6779962)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--draw', type=float, default=None)
    parser.add_argument('--sim-count', type=int, default=5)
    parser.add_argument('--max-games', type=int, default=0)
    args = parser.parse_args()

    avg_logloss, count = evaluate_params_mov(args.mu, args.sigma, args.beta, args.teams, args.games,
                                            max_games=(args.max_games if args.max_games > 0 else None),
                                            sim_count=args.sim_count, tau=args.tau, draw_probability=args.draw)
    print(f"avg_logloss={avg_logloss}, games_evaluated={count}")


if __name__ == '__main__':
    main()
