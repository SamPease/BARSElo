
import csv
from collections import defaultdict
import os
from datetime import datetime, timedelta

INITIAL_ELO = 1000
K_FACTOR = 250
TOURNAMENT_MULTIPLIER = 1.5  # Set as needed


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
        header = next(reader)  # skip header
        for row in reader:
            # Accept rows with at least 5 columns, up to 7
            if len(row) >= 5:
                # Pad to 7 columns
                while len(row) < 7:
                    row.append("")
                games.append(row)
    return games

def get_all_players(team_to_players):
    players = set()
    for plist in team_to_players.values():
        players.update([p for p in plist if p])
    return sorted(players)

def average_elo(players, elos):
    return sum(elos[p] for p in players) / len(players) if players else INITIAL_ELO


def mov_factor(goal_diff):
    if goal_diff <= 0:  # tie
        return 1
    elif goal_diff == 1:
        return 1
    elif goal_diff == 2:
        return 1.5
    else:  # goal_diff >= 3
        return (11 + goal_diff) / 8

def update_elo_custom(team1, team2, score1, score2, elos, K=20, margin_override=None, multiplier=1.0):
    avg1 = average_elo(team1, elos)
    avg2 = average_elo(team2, elos)
    expected1 = 1 / (1 + 10 ** ((avg2 - avg1) / 400))
    expected2 = 1 - expected1
    if score1 > score2:
        actual1, actual2 = 1, 0
    elif score1 < score2:
        actual1, actual2 = 0, 1
    else:
        actual1, actual2 = 0.5, 0.5
    margin = abs(score1 - score2) if margin_override is None else margin_override
    G = mov_factor(margin)
    change1 = K * G * (actual1 - expected1) * multiplier
    change2 = K * G * (actual2 - expected2) * multiplier
    for p in team1:
        elos[p] += change1 / len(team1)
    for p in team2:
        elos[p] += change2 / len(team2)
    return elos


def update_elo(team1, team2, score1, score2, elos, K=20):
    avg1 = average_elo(team1, elos)
    avg2 = average_elo(team2, elos)

    expected1 = 1 / (1 + 10 ** ((avg2 - avg1) / 400))
    expected2 = 1 - expected1

    # actual scores
    if score1 > score2:
        actual1, actual2 = 1, 0
    elif score1 < score2:
        actual1, actual2 = 0, 1
    else:
        actual1, actual2 = 0.5, 0.5

    # margin of victory multiplier
    margin = abs(score1 - score2)
    G = mov_factor(margin)

    change1 = K * G * (actual1 - expected1)
    change2 = K * G * (actual2 - expected2)

    # Update each player in team, scaled by team size
    for p in team1:
        elos[p] += change1 / len(team1)
    for p in team2:
        elos[p] += change2 / len(team2)

    return elos



def update_elo_MoV(team1, team2, score1, score2, elos):
    if len(team1) == 0 or len(team2) == 0:
        raise Exception("Teams must have at least one player each" + str(team1) + str(team2))
    avg1 = average_elo(team1, elos)
    avg2 = average_elo(team2, elos)
    expected = avg1 - avg2
    change1 = (score1 - score2 - expected) / len(team1)
    change2 = (score2 - score1 + expected) / len(team2)
    for p in team1:
        elos[p] += change1
    for p in team2:
        elos[p] += change2

def main():

    team_to_players = load_teams(TEAMS)
    games = load_games(GAMES)
    all_players = get_all_players(team_to_players)

    output_file = 'elo_results.csv'
    # Use an ordered mapping time_str -> row so multiple games sharing the same
    # timestamp collapse into a single CSV row. When a game at an existing
    # timestamp is processed, we overwrite the entry with the latest player ELOs
    # (i.e. the ELOs after applying that game). This yields one row per
    # timestamp in the output CSV.
    from collections import OrderedDict
    elo_history_map = OrderedDict()
    # Prepend a row 20 minutes before the earliest game with everyone at INITIAL_ELO
    if games:
        earliest_dt = None
        for row in games:
            time_str = row[0]
            try:
                dt = datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S')
            except Exception:
                # if parsing fails, skip this row
                continue
            if earliest_dt is None or dt < earliest_dt:
                earliest_dt = dt
        if earliest_dt is not None:
            initial_dt = earliest_dt - timedelta(minutes=20)
            initial_time_str = initial_dt.strftime('%m/%d/%Y %H:%M:%S')
            # create row with INITIAL_ELO for all players in consistent order
            row_out = [initial_time_str] + [round(INITIAL_ELO, 2) for p in all_players]
            elo_history_map[initial_time_str] = row_out
    elos = defaultdict(lambda: INITIAL_ELO)
    # Track team ELOs after each game
    team_elo_history = {team: [] for team in team_to_players}
    team_elo_change = {team: [] for team in team_to_players}
    # Track the team's starting ELO (ELO immediately before their first game)
    beginning_team_elos = {team: None for team in team_to_players}
    all_game_elo_changes = []  # (abs_change, time, t1, t2, t1_elo_before, t1_elo_after, t2_elo_before, t2_elo_after, score, t1_change, t2_change)
    # Track record high/low for teams and players
    player_high = {p: (INITIAL_ELO, None) for p in all_players}  # (elo, time)
    player_low = {p: (INITIAL_ELO, None) for p in all_players}
    team_high = {team: (INITIAL_ELO, None) for team in team_to_players}
    team_low = {team: (INITIAL_ELO, None) for team in team_to_players}

    running_margin_sum = 0
    running_margin_count = 0
    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = row[:7]
        # If outcome_flag (col 6) is set, use running average margin (rounded)
        if outcome_flag.strip():
            if running_margin_count > 0:
                margin = int(round(running_margin_sum / running_margin_count))
            else:
                margin = 1  # fallback if no games yet
            # Assign winner/loser based on s1/s2 (should be 1/0 or 0/1)
            try:
                s1, s2 = int(s1), int(s2)
            except Exception:
                s1, s2 = 1, 0
            if s1 > s2:
                s1, s2 = margin, 0
            elif s2 > s1:
                s1, s2 = 0, margin
            else:
                s1 = s2 = 0  # tie, but margin=0
        else:
            s1, s2 = int(s1), int(s2)
            margin = abs(s1 - s2)
            if margin > 0:
                running_margin_sum += margin
                running_margin_count += 1
        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])
        # Get team ELOs before game
        team1_elo_before = average_elo(team1, elos)
        team2_elo_before = average_elo(team2, elos)
        # Record starting ELO for teams the first time they appear in a game
        if t1 in beginning_team_elos and beginning_team_elos[t1] is None:
            beginning_team_elos[t1] = team1_elo_before
        if t2 in beginning_team_elos and beginning_team_elos[t2] is None:
            beginning_team_elos[t2] = team2_elo_before
        # Tournament multiplier
        multiplier = TOURNAMENT_MULTIPLIER if tourney_flag.strip() else 1.0
        # Use custom update_elo
        update_elo_custom(team1, team2, s1, s2, elos, K=K_FACTOR, margin_override=margin, multiplier=multiplier)
        # Get team ELOs after game
        team1_elo_after = average_elo(team1, elos)
        team2_elo_after = average_elo(team2, elos)
        # Track ELO history and change
        for team, before, after in [ (t1, team1_elo_before, team1_elo_after), (t2, team2_elo_before, team2_elo_after) ]:
            team_elo_history[team].append( (time, after) )
            team_elo_change[team].append( (time, after - before) )
        # Track all game ELO changes for both teams (for top 10 swings)
        score_str = f"{t1} {s1} - {s2} {t2}"
        t1_change = team1_elo_after - team1_elo_before
        t2_change = team2_elo_after - team2_elo_before
        abs_change = max(abs(t1_change), abs(t2_change))
        all_game_elo_changes.append({
            'abs_change': abs_change,
            'time': time,
            't1': t1,
            't2': t2,
            't1_elo_before': team1_elo_before,
            't1_elo_after': team1_elo_after,
            't2_elo_before': team2_elo_before,
            't2_elo_after': team2_elo_after,
            'score': score_str,
            't1_change': t1_change,
            't2_change': t2_change
        })
        # Track record high/low for teams
        for team, after in [(t1, team1_elo_after), (t2, team2_elo_after)]:
            if team_high[team][1] is None or after > team_high[team][0]:
                team_high[team] = (after, time)
            if team_low[team][1] is None or after < team_low[team][0]:
                team_low[team] = (after, time)
        # Track record high/low for players
        for p in all_players:
            p_elo = elos[p]
            if player_high[p][1] is None or p_elo > player_high[p][0]:
                player_high[p] = (p_elo, time)
            if player_low[p][1] is None or p_elo < player_low[p][0]:
                player_low[p] = (p_elo, time)
        # Save ELOs after this game for all players
        # After this game, record the current ELOs for all players at this
        # game's timestamp. If another game occurs at the same timestamp later,
        # this entry will be overwritten so the final CSV will contain the ELOs
        # after all games at that timestamp have been applied.
        row_out = [time] + [round(elos[p], 2) for p in all_players]
        elo_history_map[time] = row_out
    # Get current team ELOs
    current_team_elos = {}
    last_game_team_elos = {}
    for team, players in team_to_players.items():
        if players:
            # Current ELO
            valid_players = [p for p in players if p in elos]
            if valid_players:
                current_team_elos[team] = sum(elos[p] for p in valid_players) / len(valid_players)
            
            # Last game ELO
            if team_elo_history[team]:  # If the team has played any games
                last_game_team_elos[team] = team_elo_history[team][-1][1]  # Get the ELO from their last game

    # Fill in beginning ELOs for teams that never played or had no players
    filled_beginning_team_elos = {}
    for team, players in team_to_players.items():
        if beginning_team_elos.get(team) is not None:
            filled_beginning_team_elos[team] = beginning_team_elos[team]
            continue
        # If team never played, treat their "beginning" as their current ELO if available,
        # otherwise use INITIAL_ELO (teams with no players)
        if team in current_team_elos:
            filled_beginning_team_elos[team] = current_team_elos[team]
        else:
            filled_beginning_team_elos[team] = float(INITIAL_ELO)

    # (Summary prints removed.)

    # Write full ELO history
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Time'] + all_players)
        # Write rows in chronological order. Parse timestamps to sort reliably
        # even if the input CSV had them out of order.
        def _parse_time(ts):
            try:
                return datetime.strptime(ts, '%m/%d/%Y %H:%M:%S')
            except Exception:
                try:
                    return datetime.strptime(ts, '%m/%d/%Y')
                except Exception:
                    return None

        # Build list of (parsed_dt, time_str) and sort by parsed_dt (None go last)
        items = []
        for tstr in elo_history_map.keys():
            parsed = _parse_time(tstr)
            items.append((parsed, tstr))
        items.sort(key=lambda x: (x[0] is None, x[0]))
        for _, tstr in items:
            writer.writerow(elo_history_map[tstr])

    # (Player/team summary prints removed.)


if __name__ == '__main__':
    main()
