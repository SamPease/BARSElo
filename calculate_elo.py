
import csv
from collections import defaultdict
import os

INITIAL_ELO = 1000
 # ELO K-factor

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
        for row in reader:
            if len(row) == 5:
                games.append(row)
    return games

def get_all_players(team_to_players):
    players = set()
    for plist in team_to_players.values():
        players.update([p for p in plist if p])
    return sorted(players)

def average_elo(players, elos):
    return sum(elos[p] for p in players) / len(players) if players else INITIAL_ELO

def update_elo(team1, team2, score1, score2, elos, K=1000, expected_margin=2, gamma=0.5):
    avg1 = average_elo(team1, elos)
    avg2 = average_elo(team2, elos)
    
    expected1 = 1 / (1 + 10 ** ((avg2 - avg1) / 400))
    expected2 = 1 / (1 + 10 ** ((avg1 - avg2) / 400))
    
    if score1 > score2:
        actual1, actual2 = 1, 0
    elif score1 < score2:
        actual1, actual2 = 0, 1
    else:
        actual1 = actual2 = 0.5

    # Margin of victory factor
    margin = abs(score1 - score2)
    mov_factor = (margin / expected_margin) ** gamma

    change1 = (K * (actual1 - expected1) * mov_factor) / len(team1)
    change2 = (K * (actual2 - expected2) * mov_factor) / len(team2)

    for p in team1:
        elos[p] += change1
    for p in team2:
        elos[p] += change2


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

    team_to_players = load_teams('Teams.csv')
    games = load_games('Games.csv')
    all_players = get_all_players(team_to_players)

    output_file = 'elo_results.csv'
    elo_history = []
    elos = defaultdict(lambda: INITIAL_ELO)
    for row in games:
        time, t1, s1, t2, s2 = row
        s1, s2 = int(s1), int(s2)
        team1 = team_to_players.get(t1, [])
        team2 = team_to_players.get(t2, [])
        update_elo(team1, team2, s1, s2, elos)
        # Save ELOs after this game
        row_out = [time] + [round(elos[p], 2) for p in all_players]
        elo_history.append(row_out)

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
    for player, elo in sorted_players[-10:]:
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
