
import csv
from collections import defaultdict
import os

INITIAL_ELO = 1000
K_FACTOR = 250
TOURNAMENT_MULTIPLIER = 2.0  # Set as needed


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
    elo_history = []
    elos = defaultdict(lambda: INITIAL_ELO)
    # Track team ELOs after each game
    team_elo_history = {team: [] for team in team_to_players}
    team_elo_change = {team: [] for team in team_to_players}
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
        row_out = [time] + [round(elos[p], 2) for p in all_players]
        elo_history.append(row_out)
    # Print record high/low team ELOs
    print("\nRecord High Team ELOs:")
    for team, (elo, t) in sorted(team_high.items(), key=lambda x: -x[1][0]):
        print(f"{team}: {elo:.2f} at {t}")
    print("\nRecord Low Team ELOs:")
    for team, (elo, t) in sorted(team_low.items(), key=lambda x: x[1][0]):
        print(f"{team}: {elo:.2f} at {t}")

    # Print record high/low individual ELOs
    print("\nRecord High Individual ELOs:")
    for p, (elo, t) in sorted(player_high.items(), key=lambda x: -x[1][0])[:10]:
        print(f"{p}: {elo:.2f} at {t}")
    print("\nRecord Low Individual ELOs:")
    for p, (elo, t) in sorted(player_low.items(), key=lambda x: x[1][0])[:10]:
        print(f"{p}: {elo:.2f} at {t}")
    # Print top 10 biggest ELO changes across all teams/games
    print("\nTop 10 Biggest ELO Changes (by game):")
    for entry in sorted(all_game_elo_changes, key=lambda x: -x['abs_change'])[:10]:
        print(f"{entry['time']}: {entry['score']} | {entry['t1']} Δ{entry['t1_change']:+.2f} ({entry['t1_elo_before']:.2f}->{entry['t1_elo_after']:.2f}) | {entry['t2']} Δ{entry['t2_change']:+.2f} ({entry['t2_elo_before']:.2f}->{entry['t2_elo_after']:.2f})")

    # Write full ELO history
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Time'] + all_players)
        for row in elo_history:
            writer.writerow(row)

    # Print top 10 and bottom 10 players by ELO
    sorted_players = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Players by ELO:")
    for player, elo in sorted_players[:10]:
        print(f"{player}: {round(elo,2)}")
    print("\nBottom 10 Players by ELO:")
    for player, elo in sorted_players[-1:-11:-1]:
        print(f"{player}: {round(elo,2)}")


    # Print each team's ELO (average of its players)
    print("\nTeam ELOs:")
    for team, players in team_to_players.items():
        if players:
            avg = sum(elos[p] for p in players if p in elos) / len([p for p in players if p in elos])
            print(f"{team}: {round(avg,2)}")
        else:
            print(f"{team}: No players")


if __name__ == '__main__':
    main()
