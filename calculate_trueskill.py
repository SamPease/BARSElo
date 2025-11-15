import csv
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import trueskill as ts
# Trueskill settings
INITIAL_MU = 1000.0
INITIAL_SIGMA = INITIAL_MU / 3.0
INITIAL_BETA = INITIAL_SIGMA / 2.0
INITIAL_TAU = (INITIAL_SIGMA / 100.0)
DRAW_PROBABILITY = 0.123

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
                # ensure at least 8 columns (added: number of players)
                while len(row) < 8:
                    row.append("")
                games.append(row)
    return games


def get_all_players(team_to_players):
    players = set()
    for plist in team_to_players.values():
        players.update([p for p in plist if p])
    return sorted(players)


def main():
    # configure trueskill environment with scaled mu/sigma/beta
    ts.setup(mu=INITIAL_MU, sigma=INITIAL_SIGMA, beta=INITIAL_BETA, tau=INITIAL_TAU, draw_probability=DRAW_PROBABILITY)

    team_to_players = load_teams(TEAMS)
    games = load_games(GAMES)
    all_players = get_all_players(team_to_players)

    # Prepare rating objects for all players
    ratings = defaultdict(lambda: ts.Rating(mu=INITIAL_MU, sigma=INITIAL_SIGMA))

    # Output map: timestamp -> expose row
    expose_history_map = OrderedDict()

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
            expose_row = [initial_time_str] + [round(ts.expose(ratings[p]), 2) for p in all_players]
            expose_history_map[initial_time_str] = expose_row

    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = row[:8]

        # preserve previous outcome_flag behavior for missing detailed scores
        if outcome_flag.strip():
            # when only outcome is known, default to 2-0 for the winning team, or 0-0 for tie
            try:
                s1, s2 = int(s1), int(s2)
            except Exception:
                s1, s2 = 2, 0
            if s1 > s2:
                s1, s2 = 2, 0
            elif s2 > s1:
                s1, s2 = 0, 2
            else:
                s1 = s2 = 0
        else:
            try:
                s1, s2 = int(s1), int(s2)
            except Exception:
                s1, s2 = 0, 0

        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])


        # Determine ranks for trueskill.rate (lower is better)
        if s1 > s2:
            ranks = [0, 1]
        elif s1 < s2:
            ranks = [1, 0]
        else:
            # tie -> same rank to denote draw
            ranks = [0, 0]

        # Build team rating lists (list of lists) in the same shape trueskill expects
        team1_ratings = [ratings[p] for p in team1]
        team2_ratings = [ratings[p] for p in team2]

        # If a team has no players, skip rate and only record history (no update)
        if not team1_ratings or not team2_ratings:
            # still record current values at this timestamp
            expose_row = [time] + [round(ts.expose(ratings[p]), 2) for p in all_players]
            expose_history_map[time] = expose_row
            continue


        # Compute per-team weights based on players available versus team size.
        # Keep weights proportional to (players_total / team_size) but do not
        # include any margin-of-victory / score-based factor.
        try:
            players_total = int(players_field) if players_field.strip() else 8
        except Exception:
            players_total = 8

        def team_weights_by_players(team_players):
            size = len(team_players)
            if size == 0:
                return []
            mult = float(players_total) / float(size)
            return [mult for _ in team_players]

        weights = [team_weights_by_players(team1), team_weights_by_players(team2)]

        # Call trueskill.rate with per-player weights derived from player counts.
        rated_teams = ts.rate([team1_ratings, team2_ratings], ranks=ranks, weights=weights)

        # Apply new ratings directly (no MOV/tournament scaling)
        for team_players, rated_team in ((team1, rated_teams[0]), (team2, rated_teams[1])):
            for p, new_rating in zip(team_players, rated_team):
                ratings[p] = new_rating

        # After applying updates, save expose values for this timestamp
        expose_row = [time] + [round(ts.expose(ratings[p]), 2) for p in all_players]
        expose_history_map[time] = expose_row

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

    _write_map('trueskill_results.csv', all_players, expose_history_map)


if __name__ == '__main__':
    main()
