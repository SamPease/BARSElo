import csv
import math
from collections import defaultdict
import os
from datetime import datetime, timedelta

INITIAL_RATING = 1000.0
LEARNING_RATE = 80  # can tune
TOURNAMENT_MULTIPLIER = 3
MARGIN_BETA = 0.025 # influence of score differential

TEAMS = 'Sports Elo - Teams.csv'
GAMES = 'Sports Elo - Games.csv'
OUTPUT = 'elo_results.csv'


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
            while len(row) < 7:
                row.append("")
            games.append(row)
    return games


def get_all_players(team_to_players):
    players = set()
    for plist in team_to_players.values():
        players.update([p for p in plist if p])
    return sorted(players)


def team_strength(team, ratings):
    """Sum of player ratings (Bradley–Terry parameter)."""
    if not team:
        return INITIAL_RATING
    return sum(ratings[p] for p in team) / len(team)


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def update_ratings_bt(team1, team2, score1, score2, ratings, lr=LEARNING_RATE, multiplier=1.0):
    margin = abs(score1 - score2)
    s1_strength = team_strength(team1, ratings)
    s2_strength = team_strength(team2, ratings)
    diff = (s1_strength - s2_strength) / 400.0  # normalize like Elo
    diff += MARGIN_BETA * margin

    p1_win = logistic(diff)
    actual1 = 1 if score1 > score2 else (0.5 if score1 == score2 else 0)

    # Gradient of log likelihood wrt each player's rating
    error = (actual1 - p1_win)

    for p in team1:
        ratings[p] += lr * error * multiplier
    for p in team2:
        ratings[p] -= lr * error * multiplier

    return ratings


def main():
    team_to_players = load_teams(TEAMS)
    games = load_games(GAMES)
    all_players = get_all_players(team_to_players)

    ratings = defaultdict(lambda: INITIAL_RATING)
    history = []

    # Track team histories and changes similar to calculate_elo.py
    team_elo_history = {team: [] for team in team_to_players}
    team_elo_change = {team: [] for team in team_to_players}
    beginning_team_elos = {team: None for team in team_to_players}
    all_game_elo_changes = []

    # Track record high/low for players
    player_high = {p: (INITIAL_RATING, None) for p in all_players}
    player_low = {p: (INITIAL_RATING, None) for p in all_players}

    # initialize
    if games:
        earliest_dt = None
        for row in games:
            try:
                dt = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S')
                if earliest_dt is None or dt < earliest_dt:
                    earliest_dt = dt
            except Exception:
                continue
        if earliest_dt:
            t0 = earliest_dt - timedelta(minutes=20)
            history.append([t0.strftime('%m/%d/%Y %H:%M:%S')] + [INITIAL_RATING for _ in all_players])

    for row in games:
        time, t1, s1, t2, s2, outcome_flag, tourney_flag = row[:7]
        try:
            s1, s2 = int(s1), int(s2)
        except Exception:
            continue
        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])

        # compute margin and team strengths before update
        margin = abs(s1 - s2)
        team1_elo_before = team_strength(team1, ratings)
        team2_elo_before = team_strength(team2, ratings)

        # Record starting ELO for teams the first time they appear in a game
        if t1 in beginning_team_elos and beginning_team_elos[t1] is None:
            beginning_team_elos[t1] = team1_elo_before
        if t2 in beginning_team_elos and beginning_team_elos[t2] is None:
            beginning_team_elos[t2] = team2_elo_before

        multiplier = TOURNAMENT_MULTIPLIER if tourney_flag.strip() else 1.0
        ratings = update_ratings_bt(team1, team2, s1, s2, ratings, multiplier=multiplier)

        # Get team ELOs after game
        team1_elo_after = team_strength(team1, ratings)
        team2_elo_after = team_strength(team2, ratings)

        # Track ELO history and change
        for team, before, after in [ (t1, team1_elo_before, team1_elo_after), (t2, team2_elo_before, team2_elo_after) ]:
            team_elo_history[team].append( (time, after) )
            team_elo_change[team].append( (time, after - before) )

        # Track all game changes for reporting
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

        # Track record high/low for players
        for p in all_players:
            p_rating = ratings[p]
            if player_high[p][1] is None or p_rating > player_high[p][0]:
                player_high[p] = (p_rating, time)
            if player_low[p][1] is None or p_rating < player_low[p][0]:
                player_low[p] = (p_rating, time)

        # Save ratings after this game for all players
        history.append([time] + [round(ratings[p], 2) for p in all_players])

    # Output results
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time"] + all_players)
        writer.writerows(history)

    # Prepare team summaries
    current_team_elos = {}
    last_game_team_elos = {}
    for team, players in team_to_players.items():
        if players:
            valid_players = [p for p in players if p in ratings]
            if valid_players:
                current_team_elos[team] = sum(ratings[p] for p in valid_players) / len(valid_players)
            if team_elo_history[team]:
                last_game_team_elos[team] = team_elo_history[team][-1][1]

    # Fill in beginning ELOs for teams that never played or had no players
    filled_beginning_team_elos = {}
    for team, players in team_to_players.items():
        if beginning_team_elos.get(team) is not None:
            filled_beginning_team_elos[team] = beginning_team_elos[team]
            continue
        if team in current_team_elos:
            filled_beginning_team_elos[team] = current_team_elos[team]
        else:
            filled_beginning_team_elos[team] = float(INITIAL_RATING)

    # Print team/player summaries (mirroring calculate_elo.py)
    print("\nTop 10 Teams by Current Rating:")
    for team, elo in sorted(current_team_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{team}: {elo:.2f}")

    print("\nBottom 10 Teams by Current Rating:")
    for team, elo in sorted(current_team_elos.items(), key=lambda x: x[1])[:10]:
        print(f"{team}: {elo:.2f}")

    print("\nTop 10 Teams by Beginning Rating (immediately before first game):")
    for team, elo in sorted(filled_beginning_team_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{team}: {elo:.2f}")

    print("\nBottom 10 Teams by Beginning Rating (immediately before first game):")
    for team, elo in sorted(filled_beginning_team_elos.items(), key=lambda x: x[1])[:10]:
        print(f"{team}: {elo:.2f}")

    print("\nTop 10 Teams by Last Game Rating:")
    for team, elo in sorted(last_game_team_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        last_game_time = team_elo_history[team][-1][0]
        print(f"{team}: {elo:.2f} (last played {last_game_time})")

    print("\nBottom 10 Teams by Last Game Rating:")
    for team, elo in sorted(last_game_team_elos.items(), key=lambda x: x[1])[:10]:
        last_game_time = team_elo_history[team][-1][0]
        print(f"{team}: {elo:.2f} (last played {last_game_time})")

    # Record high/low individual ratings
    print("\nRecord High Individual Ratings:")
    for p, (val, t) in sorted(player_high.items(), key=lambda x: -x[1][0])[:10]:
        print(f"{p}: {val:.2f} at {t}")
    print("\nRecord Low Individual Ratings:")
    for p, (val, t) in sorted(player_low.items(), key=lambda x: x[1][0])[:10]:
        print(f"{p}: {val:.2f} at {t}")

    # Top 10 biggest team changes
    print("\nTop 10 Biggest Rating Changes (by game):")
    for entry in sorted(all_game_elo_changes, key=lambda x: -x['abs_change'])[:10]:
        print(f"{entry['time']}: {entry['score']} | {entry['t1']} Δ{entry['t1_change']:+.2f} ({entry['t1_elo_before']:.2f}->{entry['t1_elo_after']:.2f}) | {entry['t2']} Δ{entry['t2_change']:+.2f} ({entry['t2_elo_before']:.2f}->{entry['t2_elo_after']:.2f})")

    # Print top/bottom players by rating
    sorted_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Players by Rating:")
    for player, elo in sorted_players[:10]:
        print(f"{player}: {round(elo,2)}")
    print("\nBottom 10 Players by Rating:")
    for player, elo in sorted_players[-1:-11:-1]:
        print(f"{player}: {round(elo,2)}")

    # Print all teams with current ratings, sorted by last game time
    print("\nAll Team Ratings (Current), sorted by last game time (least recent first, then most recent). Teams with no games are listed last in CSV order:")
    original_order = {team: idx for idx, team in enumerate(team_to_players.keys())}
    team_rows = []
    for team in team_to_players.keys():
        players = team_to_players[team]
        valid_players = [p for p in players if p in ratings]
        current_elo = sum(ratings[p] for p in valid_players) / len(valid_players) if valid_players else None
        last_dt = None
        if team_elo_history.get(team):
            last_time_str = team_elo_history[team][-1][0]
            try:
                last_dt = datetime.strptime(last_time_str, '%m/%d/%Y %H:%M:%S')
            except Exception:
                last_dt = None
        team_rows.append((team, current_elo, last_dt, original_order.get(team, 10**9)))

    def sort_key(item):
        team, current_elo, last_dt, orig_idx = item
        if last_dt is not None:
            return (0, last_dt.timestamp(), orig_idx)
        else:
            return (1, orig_idx)

    for team, current_elo, last_dt, _ in sorted(team_rows, key=sort_key):
        if not team_to_players[team]:
            print(f"{team}: No players")
            continue
        if current_elo is None:
            current_str = "N/A"
        else:
            current_str = f"{current_elo:.2f}"
        print(f"{team}: {current_str}")


if __name__ == "__main__":
    main()
