import csv
from collections import defaultdict, OrderedDict
import os
from datetime import datetime, timedelta
import trueskill as ts
import math

# Trueskill settings
INITIAL_MU = 1000.0
# Scale sigma proportionally to the library defaults (default mu=25, sigma=8.3333)
DEFAULT_MU = 25.0
DEFAULT_SIGMA = 8.333333333333334
INITIAL_SIGMA = INITIAL_MU * (DEFAULT_SIGMA / DEFAULT_MU)
# trueskill default beta for (mu=25, sigma=8.3333) is ~4.1667; scale similarly
DEFAULT_BETA = 4.166666666666667
INITIAL_BETA = INITIAL_SIGMA * (DEFAULT_BETA / DEFAULT_SIGMA)

# MOV-to-multiplier hyperparameters
MAX_EXTRA = 2.0  # allow factor up to 1 + MAX_EXTRA (e.g., up to ~3x)
C = 1.2  # sensitivity constant for tanh mapping
MAX_FACTOR_CAP = 1.0 + MAX_EXTRA

TOURNAMENT_MULTIPLIER = 1.5

TEAMS = 'Sports Elo - Teams.csv'
GAMES = 'Sports Elo - Games.csv'


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
                while len(row) < 7:
                    row.append("")
                games.append(row)
    return games


def get_all_players(team_to_players):
    players = set()
    for plist in team_to_players.values():
        players.update([p for p in plist if p])
    return sorted(players)


def average_mu(players, ratings):
    return sum(ratings[p].mu for p in players) / len(players) if players else INITIAL_MU


def mov_factor(goal_diff):
    if goal_diff <= 0:  # tie
        return 1
    elif goal_diff == 1:
        return 1
    elif goal_diff == 2:
        return 1.5
    else:  # goal_diff >= 3
        return (11 + goal_diff) / 8


def team_mu_and_var(team_players, ratings, beta=INITIAL_BETA):
    """Return (mu_sum, var_sum) for a team using trueskill variances.

    var_sum = sum(sigma_i^2) + n * beta^2
    """
    mus = [ratings[p].mu for p in team_players]
    sig2 = [ratings[p].sigma ** 2 for p in team_players]
    mu_sum = sum(mus)
    var_sum = sum(sig2) + len(team_players) * (beta ** 2)
    return mu_sum, var_sum


def mov_factor_from_margin(teamA, teamB, ratings, margin, is_tourney=False,
                           max_extra=MAX_EXTRA, c=C, beta=INITIAL_BETA):
    """Compute MOV multiplier from observed margin using Gaussian assumptions.

    Returns a multiplier >= 1.0. Uses tanh mapping to smoothly cap large z-scores.
    """
    muA, varA = team_mu_and_var(teamA, ratings, beta=beta)
    muB, varB = team_mu_and_var(teamB, ratings, beta=beta)
    mu_diff = muA - muB
    var_diff = varA + varB
    if var_diff <= 0:
        var_diff = 1e-6
    # standardize: how surprising is the observed margin relative to current belief
    z = (margin - mu_diff) / math.sqrt(var_diff)
    z_abs = abs(z)
    extra = max_extra * math.tanh(z_abs / c)
    factor = 1.0 + extra
    # cap
    if factor > 1.0 + max_extra:
        factor = 1.0 + max_extra
    if is_tourney:
        factor *= TOURNAMENT_MULTIPLIER
    # final safety cap
    factor = min(factor, MAX_FACTOR_CAP * (TOURNAMENT_MULTIPLIER if is_tourney else 1.0))
    return factor


def main():
    # configure trueskill environment with scaled mu/sigma/beta
    ts.setup(mu=INITIAL_MU, sigma=INITIAL_SIGMA, beta=INITIAL_BETA)

    team_to_players = load_teams(TEAMS)
    games = load_games(GAMES)
    all_players = get_all_players(team_to_players)

    # Prepare rating objects for all players
    ratings = defaultdict(lambda: ts.Rating(mu=INITIAL_MU, sigma=INITIAL_SIGMA))

    # Output maps: timestamp -> row
    mu_history_map = OrderedDict()
    mu_minus_3sigma_history_map = OrderedDict()

    # Prepend initial row 20 minutes before earliest game
    if games:
        earliest_dt = None
        for row in games:
            time_str = row[0]
            try:
                dt = datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S')
            except Exception:
                continue
            if earliest_dt is None or dt < earliest_dt:
                earliest_dt = dt
        if earliest_dt is not None:
            initial_dt = earliest_dt - timedelta(minutes=20)
            initial_time_str = initial_dt.strftime('%m/%d/%Y %H:%M:%S')
            mu_row = [initial_time_str] + [round(ratings[p].mu, 2) for p in all_players]
            mu_minus_3sigma_row = [initial_time_str] + [round(ratings[p].mu - 3 * ratings[p].sigma, 2) for p in all_players]
            mu_history_map[initial_time_str] = mu_row
            mu_minus_3sigma_history_map[initial_time_str] = mu_minus_3sigma_row

    # Running margin logic copied from calculate_elo
    running_margin_sum = 0
    running_margin_count = 0

    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = row[:7]

        if outcome_flag.strip():
            if running_margin_count > 0:
                margin = int(round(running_margin_sum / running_margin_count))
            else:
                margin = 1
            try:
                s1, s2 = int(s1), int(s2)
            except Exception:
                s1, s2 = 1, 0
            if s1 > s2:
                s1, s2 = margin, 0
            elif s2 > s1:
                s1, s2 = 0, margin
            else:
                s1 = s2 = 0
        else:
            s1, s2 = int(s1), int(s2)
            margin = abs(s1 - s2)
            if margin > 0:
                running_margin_sum += margin
                running_margin_count += 1

        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])

        # Team mu before
        team1_mu_before = average_mu(team1, ratings)
        team2_mu_before = average_mu(team2, ratings)

        # Determine ranks for trueskill.rate (lower is better)
        if s1 > s2:
            ranks = [0, 1]
        elif s1 < s2:
            ranks = [1, 0]
        else:
            # tie -> same rank to denote draw
            ranks = [0, 0]

        multiplier = TOURNAMENT_MULTIPLIER if tourney_flag.strip() else 1.0

        # Build team rating lists (list of lists) in the same shape trueskill expects
        team1_ratings = [ratings[p] for p in team1]
        team2_ratings = [ratings[p] for p in team2]

        # If a team has no players, skip rate and only record history (no update)
        if not team1_ratings or not team2_ratings:
            # still record current values at this timestamp
            row_mu = [time] + [round(ratings[p].mu, 2) for p in all_players]
            row_mu_3 = [time] + [round(ratings[p].mu - 3 * ratings[p].sigma, 2) for p in all_players]
            mu_history_map[time] = row_mu
            mu_minus_3sigma_history_map[time] = row_mu_3
            continue

        # Use trueskill to get tentative new ratings
        rated_teams = ts.rate([team1_ratings, team2_ratings], ranks=ranks)

        # rated_teams is a list matching input teams; compute MOV multiplier
        G = mov_factor_from_margin(team1, team2, ratings, margin, is_tourney=bool(tourney_flag.strip()))

        # Apply scaled deltas to each player's rating
        # rated_teams[0] corresponds to team1, rated_teams[1] to team2
        for team_players, rated_team in ((team1, rated_teams[0]), (team2, rated_teams[1])):
            for p, new_rating in zip(team_players, rated_team):
                old = ratings[p]
                delta_mu = new_rating.mu - old.mu
                delta_sigma = new_rating.sigma - old.sigma
                # Scale both mu and sigma deltas by MOV and tournament multiplier
                scaled_mu = old.mu + delta_mu * G * multiplier
                scaled_sigma = max(1e-6, old.sigma + delta_sigma * G * multiplier)
                ratings[p] = ts.Rating(mu=scaled_mu, sigma=scaled_sigma)

        # After applying updates, save history for this timestamp
        row_mu = [time] + [round(ratings[p].mu, 2) for p in all_players]
        row_mu_3 = [time] + [round(ratings[p].mu - 3 * ratings[p].sigma, 2) for p in all_players]
        mu_history_map[time] = row_mu
        mu_minus_3sigma_history_map[time] = row_mu_3

    # Write CSV outputs (sorted by parsed timestamp)
    def _parse_time(ts_str):
        try:
            return datetime.strptime(ts_str, '%m/%d/%Y %H:%M:%S')
        except Exception:
            try:
                return datetime.strptime(ts_str, '%m/%d/%Y')
            except Exception:
                return None

    def _write_map(filename, header_players, data_map):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'] + header_players)
            items = []
            for tstr in data_map.keys():
                parsed = _parse_time(tstr)
                items.append((parsed, tstr))
            items.sort(key=lambda x: (x[0] is None, x[0]))
            for _, tstr in items:
                writer.writerow(data_map[tstr])

    _write_map('trueskill_mu_results.csv', all_players, mu_history_map)
    _write_map('trueskill_mu_minus_3sigma_results.csv', all_players, mu_minus_3sigma_history_map)


if __name__ == '__main__':
    main()
