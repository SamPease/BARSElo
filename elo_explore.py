import csv

ELO_RESULTS = 'elo_results.csv'
TEAMS_FILE = 'Sports Elo - Teams.csv'

def load_elo_results(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        last_row = None
        for row in reader:
            last_row = row
        if last_row is None:
            raise ValueError('No data in ELO results')
        # Map player name to ELO
        return dict(zip(header[1:], map(float, last_row[1:])))

def load_teams(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        team_to_players = {team: [] for team in reader.fieldnames}
        for row in reader:
            for team in reader.fieldnames:
                player = row[team].strip()
                if player:
                    team_to_players[team].append(player)
    return team_to_players

def invert_teams(team_to_players):
    player_to_teams = {}
    for team, players in team_to_players.items():
        for p in players:
            player_to_teams.setdefault(p, []).append(team)
    return player_to_teams

def main():
    elos = load_elo_results(ELO_RESULTS)
    team_to_players = load_teams(TEAMS_FILE)
    player_to_teams = invert_teams(team_to_players)
    print("Type 'team TEAMNAME' to see ELOs of all members, or 'player PLAYERNAME' to see all teams and ELOs for that player. Type 'exit' to quit.")
    while True:
        cmd = input('> ').strip()
        if cmd.lower() == 'exit':
            break
        if cmd.lower().startswith('team '):
            team = cmd[5:].strip()
            if team not in team_to_players:
                print(f"Team '{team}' not found.")
                continue
            # Print team ELO (average of members)
            members = team_to_players[team]
            member_elos = [elos.get(p) for p in members if p in elos]
            if member_elos:
                avg_elo = sum(member_elos) / len(member_elos)
                print(f"Team {team} average ELO: {avg_elo:.2f}")
            else:
                print(f"Team {team} has no ELO data.")
            print(f"ELOs for team {team}:")
            for p in members:
                print(f"  {p}: {elos.get(p, 'N/A')}")
        elif cmd.lower().startswith('player '):
            player = cmd[7:].strip()
            if player not in player_to_teams:
                print(f"Player '{player}' not found.")
                continue
            print(f"Player {player} ELO: {elos.get(player, 'N/A')}")
            print(f"Teams for player {player}:")
            for t in player_to_teams[player]:
                members = team_to_players[t]
                member_elos = [elos.get(p) for p in members if p in elos]
                if member_elos:
                    avg_elo = sum(member_elos) / len(member_elos)
                    print(f"  {t}: avg ELO {avg_elo:.2f}")
                else:
                    print(f"  {t}: avg ELO N/A")
        else:
            print("Unknown command. Use 'team TEAMNAME' or 'player PLAYERNAME'.")

if __name__ == '__main__':
    main()
