"""
rolling_cv_evaluate.py

Run a rolling (online) cross-validation over choices of K (K-factor)
and tournament multiplier to evaluate their effect on game prediction.

This script re-uses the loader and ELO update logic from calculate_elo.py.

Usage examples:
    python3 rolling_cv_evaluate.py --ks 10,20,40 --tms 1.0,1.5,2.0

Output: CSV with columns: K,tournament_multiplier,games,accuracy,brier,logloss
"""

import argparse
import csv
import math
import statistics
from itertools import product

from calculate_elo import load_teams, load_games, get_all_players, average_elo, update_elo_custom, INITIAL_ELO


def expected_prob(avg1, avg2):
    # Elo expected probability of team1 winning
    return 1.0 / (1.0 + 10 ** ((avg2 - avg1) / 400.0))


def safe_parse_score(s):
    try:
        return int(s)
    except Exception:
        return None


def evaluate_params(games, team_to_players, K, tournament_mult):
    # initialize player elos
    all_players = get_all_players(team_to_players)
    elos = {p: float(INITIAL_ELO) for p in all_players}

    total = 0
    correct = 0
    brier_sum = 0.0
    logloss_sum = 0.0

    for row in games:
        # each row has at least 5 cols, up to 7 in calculate_elo format
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = (row + [""] * 7)[:7]

        s1_val = safe_parse_score(s1)
        s2_val = safe_parse_score(s2)
        if s1_val is None or s2_val is None:
            # skip games with non-numeric scores
            continue

        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])

        avg1 = average_elo(team1, elos)
        avg2 = average_elo(team2, elos)

        p1 = expected_prob(avg1, avg2)

        # define actual outcome numeric: 1 = team1 win, 0 = team2 win, 0.5 = tie
        if s1_val > s2_val:
            actual = 1.0
        elif s1_val < s2_val:
            actual = 0.0
        else:
            actual = 0.5

        # prediction accuracy: choose winner by p>0.5 (ties not counted as correct unless p==0.5)
        pred = 1.0 if p1 > 0.5 else (0.0 if p1 < 0.5 else 0.5)
        if pred == actual:
            correct += 1
        total += 1

        # Brier score
        brier_sum += (p1 - actual) ** 2

        # log loss: guard p1 in (eps,1-eps)
        eps = 1e-15
        p1c = min(max(p1, eps), 1 - eps)
        # for ties treat as 0.5 -> compute logistic mixture logloss
        # use generic formula: -[actual*log(p) + (1-actual)*log(1-p)]
        logloss_sum += -(actual * math.log(p1c) + (1 - actual) * math.log(1 - p1c))

        # apply elo update (use K and tournament multiplier)
        margin = abs(s1_val - s2_val)
        update_elo_custom(team1, team2, s1_val, s2_val, elos, K=K, margin_override=margin, multiplier=tournament_mult)

    if total == 0:
        return None

    accuracy = correct / total
    brier = brier_sum / total
    logloss = logloss_sum / total
    return {
        'K': K,
        'tournament_multiplier': tournament_mult,
        'games': total,
        'accuracy': accuracy,
        'brier': brier,
        'logloss': logloss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ks', default='10,20,40,80', help='Comma-separated K-factor values to try')
    parser.add_argument('--tms', default='1.0,1.5,2.0', help='Comma-separated tournament multipliers to try')
    parser.add_argument('--metric', choices=['accuracy', 'brier', 'logloss'], default='logloss',
                        help='Metric to optimize (accuracy higher is better; brier/logloss lower is better)')
    parser.add_argument('--top', type=int, default=1, help='Show top N results (sorted by chosen metric)')
    parser.add_argument('--mode', choices=['grid', 'adaptive'], default='adaptive',
                        help='Search mode: full grid (grid) or multiplicative adaptive zoom (adaptive)')
    parser.add_argument('--rounds', type=int, default=4, help='Number of adaptive zoom rounds')
    parser.add_argument('--grid-size', type=int, default=7, help='Grid size per axis for each adaptive round')
    parser.add_argument('--span', type=float, default=2.0, help='Initial multiplicative span around center (low=center/span, high=center*span)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Minimum improvement in metric to continue adapting')
    parser.add_argument('--games-file', default=None, help='Optional path to games CSV (overrides default in calculate_elo)')
    parser.add_argument('--teams-file', default=None, help='Optional path to teams CSV (overrides default in calculate_elo)')
    args = parser.parse_args()

    ks = [float(x) for x in args.ks.split(',') if x.strip()]
    tms = [float(x) for x in args.tms.split(',') if x.strip()]

    # load data
    teams_file = args.teams_file if args.teams_file else None
    games_file = args.games_file if args.games_file else None

    # calculate_elo.load_teams/load_games are bound to filenames inside calculate_elo; to override, pass path args to those functions if needed.
    # For simplicity we'll call them without args and assume files are in cwd under their expected names.
    team_to_players = load_teams(teams_file) if teams_file else load_teams('Sports Elo - Teams.csv')
    games = load_games(games_file) if games_file else load_games('Sports Elo - Games.csv')

    results = []
    def evaluate_grid(ks_list, tms_list):
        out = []
        for K, tm in product(ks_list, tms_list):
            print(f'Evaluating K={K}, tournament_multiplier={tm} ...')
            metrics = evaluate_params(games, team_to_players, K=K, tournament_mult=tm)
            if metrics:
                out.append(metrics)
                print(f"  games={metrics['games']} accuracy={metrics['accuracy']:.3f} brier={metrics['brier']:.4f} logloss={metrics['logloss']:.4f}")
            else:
                print('  no games processed for this configuration')
        return out

    def adaptive_search(initial_ks, initial_tms, rounds=4, grid_size=7, span=2.0, tol=1e-6):
        # start center at geometric mean of initial lists
        def geom_mean(vals):
            vals = [float(v) for v in vals]
            prod = 1.0
            for v in vals:
                prod *= v
            return prod ** (1.0 / len(vals))

        center_k = geom_mean(initial_ks)
        center_tm = geom_mean(initial_tms)
        current_span = span

        best_overall = None
        for r in range(1, rounds + 1):
            # build multiplicative geometric grid around center
            def geom_space(center, span_val, n):
                low = center / span_val
                high = center * span_val
                if n == 1:
                    return [center]
                vals = []
                for i in range(n):
                    frac = i / (n - 1)
                    vals.append(low * ((high / low) ** frac))
                return vals

            ks_round = geom_space(center_k, current_span, grid_size)
            tms_round = geom_space(center_tm, current_span, grid_size)

            print(f'Adaptive round {r}: center_k={center_k:.6g} center_tm={center_tm:.6g} span={current_span:.6g} grid={grid_size}x{grid_size}')
            round_results = evaluate_grid(ks_round, tms_round)
            if not round_results:
                break

            # pick best by logloss (min)
            round_best = min(round_results, key=lambda x: x['logloss'])
            if best_overall is None or round_best['logloss'] < best_overall['logloss']:
                best_overall = round_best

            improvement = 0.0
            if best_overall and round_best:
                if 'logloss' in best_overall and 'logloss' in round_best:
                    # improvement = previous_best - new_best (positive means improvement)
                    # when best_overall equals round_best this will be 0
                    improvement = (best_overall['logloss'] - round_best['logloss'])

            print(f"  round best: K={round_best['K']} tm={round_best['tournament_multiplier']} logloss={round_best['logloss']:.6g} improvement={improvement:.6g}")

            # stop if improvement is tiny
            if improvement is not None and abs(improvement) < tol:
                print('Improvement below tol; stopping adaptive search')
                break

            # set new center to round_best and shrink span
            center_k = float(round_best['K'])
            center_tm = float(round_best['tournament_multiplier'])
            # shrink multiplicative span towards 1 (no change) by sqrt
            current_span = max(1.01, current_span ** 0.5)

        return best_overall

    # choose mode
    if args.mode == 'grid':
        results = evaluate_grid(ks, tms)
    else:
        # adaptive mode
        print('Starting adaptive multiplicative search...')
        best = adaptive_search(ks, tms, rounds=args.rounds, grid_size=args.grid_size, span=args.span, tol=args.tol)
        results = [best] if best else []
    if not results:
        print('No results to report.')
        return

    metric = args.metric
    # accuracy should be sorted descending (higher is better); brier/logloss ascending (lower is better)
    reverse = True if metric == 'accuracy' else False
    sorted_results = sorted(results, key=lambda r: r[metric], reverse=reverse)

    top_n = max(1, args.top)
    print('\nTop {} results by {}:'.format(top_n, metric))
    print('K\ttm\tgames\taccuracy\tbrier\tlogloss')
    for r in sorted_results[:top_n]:
        print(f"{int(r['K']) if float(r['K']).is_integer() else r['K']}\t{r['tournament_multiplier']}\t{r['games']}\t{r['accuracy']:.4f}\t{r['brier']:.5f}\t{r['logloss']:.5f}")

    best = sorted_results[0]
    print('\nBest parameters:')
    print(f"  K={best['K']}  tournament_multiplier={best['tournament_multiplier']}  ({metric}={best[metric]})")

    print('\nDone.')


if __name__ == '__main__':
    main()
