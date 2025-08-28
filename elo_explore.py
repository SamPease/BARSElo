import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.patheffects as pe
import numpy as np

ELO_RESULTS = 'elo_results.csv'
TEAMS_FILE = 'Sports Elo - Teams.csv'
GAMES_FILE = 'Sports Elo - Games.csv'

def load_games(filename):
    games = []  # List of game records
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) >= 5:  # Ensure row has enough columns
                try:
                    time = row[0].strip()
                    team1 = row[1].strip()
                    team2 = row[3].strip()
                    if time and team1 and team2:
                        date = datetime.strptime(time, '%m/%d/%Y %H:%M:%S')
                        games.append([date, team1, team2])
                except (ValueError, IndexError) as e:
                    print(f"Error parsing game row: {row}")
                    continue
    return sorted(games)  # Sort by date

def load_elo_results(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # First row is header
        dates = []
        elo_data = {}
        
        # Initialize elo_data with empty lists for each player
        for player in header[1:]:
            elo_data[player] = []
            
        # Read each row and store data
        for row in reader:
            date = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S')
            dates.append(date)
            for i, elo in enumerate(row[1:]):
                player = header[i+1]
                elo_data[player].append(float(elo))
                
        if not dates:
            raise ValueError('No data in ELO results')
            
        return header[1:], dates, elo_data

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

INITIAL_ELO = 1000  # Same as in calculate_elo.py

def plot_player_elo(player, dates, elo_data, team_to_players=None, games=None):
    # Make room on the right for an external legend (match team plot layout)
    plt.figure(figsize=(15, 6))
    
    # Create main plot area (use 60% width so legend can sit on the right)
    plt.axes([0.1, 0.1, 0.6, 0.8])
    
    # Plot player line
    plt.plot(dates, elo_data[player], 'b-', label='ELO')
    # mark player's first and last ELO with dots (no legend entries)
    plt.scatter(dates[0], elo_data[player][0], color='b', s=50, zorder=4, label='_nolegend_',
                edgecolors='white', linewidths=0.8)
    plt.scatter(dates[-1], elo_data[player][-1], color='b', s=50, zorder=4, label='_nolegend_',
                edgecolors='white', linewidths=0.8)
    plt.axhline(y=1000, color='r', linestyle='--', label='Starting ELO')
    
    plt.title(f'{player} ELO History')
    plt.xlabel('Date')
    plt.ylabel('ELO Rating')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    date_formatter = DateFormatter("%m/%d/%Y")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    
    # Add some padding to y-axis
    plt.margins(y=0.1)
    
    # If team and game info provided, plot team-average ELOs for teams the player belonged to
    if team_to_players and games:
        # find teams the player is/was on
        player_teams = [t for t, members in team_to_players.items() if player in members]
        cmap = plt.get_cmap('tab10')
        for ti, team in enumerate(player_teams):
            # find team's games
            team_games = [g for g in games if g[1] == team or g[2] == team]
            if not team_games:
                continue
            first_game = min(g[0] for g in team_games)
            last_game = max(g[0] for g in team_games)

            # determine indices for date range
            try:
                start_idx = next(i for i, d in enumerate(dates) if d >= first_game)
            except StopIteration:
                start_idx = 0
            try:
                end_idx = max(i for i, d in enumerate(dates) if d <= last_game)
            except ValueError:
                end_idx = len(dates) - 1

            # compute team average over that window
            members = [p for p in team_to_players.get(team, []) if p in elo_data]
            if not members:
                continue
            team_segment = []
            for i in range(start_idx, end_idx + 1):
                team_segment.append(sum(elo_data[p][i] for p in members) / len(members))

            seg_dates = dates[start_idx:end_idx + 1]
            color = cmap(ti % 10)
            # plot team segment without adding to legend
            plt.plot(seg_dates, team_segment, color=color, linestyle='--', linewidth=2, label='_nolegend_')
            # mark the segment start and end and annotate the team name inline for clarity
            y_first_seg = team_segment[0]
            y_last_seg = team_segment[-1]
            plt.scatter(seg_dates[0], y_first_seg, color=color, s=40, zorder=3, label='_nolegend_',
                        edgecolors='white', linewidths=0.8)
            plt.scatter(seg_dates[-1], y_last_seg, color=color, s=40, zorder=3, label='_nolegend_',
                        edgecolors='white', linewidths=0.8)
            plt.gca().annotate(
                team,
                xy=(seg_dates[-1], y_last_seg),
                xytext=(6, 0),
                textcoords='offset points',
                va='center',
                fontsize='small',
                color=color,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')]
            )

    # Place legend outside of the plot on the right (includes player, starting ELO, and team segments)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')

    plt.show()

def plot_team_elo(team, dates, elo_data, team_to_players, games):
    # Create figure with extra width for legend
    plt.figure(figsize=(15, 6))
    
    # Create the main plot area with adjusted size to accommodate legend
    main_plot = plt.axes([0.1, 0.1, 0.6, 0.8])
    
    # Calculate and plot individual player ELOs
    players = team_to_players[team]
    valid_players = [p for p in players if p in elo_data]
    
    # Generate a color map for players
    colors = plt.cm.rainbow(np.linspace(0, 1, len(valid_players)))
    
    # Plot individual player lines (no legend entries) and add inline labels at the right edge
    label_positions = []  # list of (y, player, color)
    for player, color in zip(valid_players, colors):
        # plot the line without adding it to the legend
        line, = plt.plot(dates, elo_data[player], color=color, alpha=0.8, linewidth=1.5, label='_nolegend_')
        # mark the first and last points
        y_first = elo_data[player][0]
        y_last = elo_data[player][-1]
        plt.scatter(dates[0], y_first, color=color, s=30, zorder=3, label='_nolegend_', edgecolors='white', linewidths=0.6)
        plt.scatter(dates[-1], y_last, color=color, s=30, zorder=3, label='_nolegend_', edgecolors='white', linewidths=0.6)
        # record label position for later collision avoidance
        label_positions.append((y_last, player, color))

    # Simple collision avoidance for right-edge labels: sort by y and nudge if too close
    if label_positions:
        label_positions.sort(key=lambda x: x[0])
        adjusted_positions = []
        min_sep = max(6, (plt.ylim()[1] - plt.ylim()[0]) * 0.02)  # minimal separation in data units
        for y, player, color in label_positions:
            if not adjusted_positions:
                adjusted_positions.append([y, player, color])
            else:
                prev_y = adjusted_positions[-1][0]
                if y - prev_y < min_sep:
                    y = prev_y + min_sep
                adjusted_positions.append([y, player, color])

        # place annotations using adjusted positions
        for y, player, color in adjusted_positions:
            plt.gca().annotate(
                player,
                xy=(dates[-1], y),
                xytext=(6, 0),
                textcoords='offset points',
                va='center',
                fontsize='small',
                color=color,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')]
            )
    
    # Calculate and plot team average ELO
    team_elos = []
    for i, date in enumerate(dates):
        if valid_players:
            avg_elo = sum(elo_data[p][i] for p in valid_players) / len(valid_players)
            team_elos.append(avg_elo)
        else:
            team_elos.append(1000)  # Default ELO if no valid players
    
    # Plot team average with thick line
    plt.plot(dates, team_elos, 'k-', linewidth=2.5, label=f'{team} (Team Average)')
    # Mark the start and end of the team average with dots
    plt.scatter(dates[0], team_elos[0], color='k', s=50, zorder=4, label='_nolegend_', edgecolors='white', linewidths=0.8)
    plt.scatter(dates[-1], team_elos[-1], color='k', s=50, zorder=4, label='_nolegend_', edgecolors='white', linewidths=0.8)
    
    # Find team's first and last game
    team_games = [game for game in games if game[1] == team or game[2] == team]
    if team_games:
        first_game = min(game[0] for game in team_games)
        last_game = max(game[0] for game in team_games)
        
        # Plot background colors
        plt.axvspan(dates[0], first_game, color='gray', alpha=0.15, label='Before First Game')
        plt.axvspan(first_game, last_game, color='lightblue', alpha=0.3, label='Active Period')
        plt.axvspan(last_game, dates[-1], color='gray', alpha=0.15, label='After Last Game')
    
    plt.axhline(y=1000, color='r', linestyle='--', label='Starting ELO')
    
    plt.title(f'{team} ELO History')
    plt.xlabel('Date')
    plt.ylabel('Team ELO Rating')
    plt.grid(True, alpha=0.3)
    
    # Place legend outside of the plot on the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
              ncol=1, fontsize='small')
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    date_formatter = DateFormatter("%m/%d/%Y")
    plt.gca().xaxis.set_major_formatter(date_formatter)
    
    # Add some padding to y-axis
    plt.margins(y=0.1)
    
    plt.show()

def main():
    header, dates, elo_data = load_elo_results(ELO_RESULTS)
    team_to_players = load_teams(TEAMS_FILE)
    player_to_teams = invert_teams(team_to_players)
    games = load_games(GAMES_FILE)
    
    # Get current ELOs (from last row)
    current_elos = {player: elo_data[player][-1] for player in header}
    
    print(f"Found {len(dates)} ELO records and {len(games)} games in history")
    
    # Create date -> elos mapping for historical lookups
    date_to_elos = {}
    for i, date in enumerate(dates):
        date_to_elos[date.date()] = {player: elo_data[player][i] for player in header}
    
    # Find last game for each team
    team_last_game = {}  # {team: (date, opponent)}
    for game_date, team1, team2 in reversed(games):  # Go backwards to find most recent first
        if team1 not in team_last_game:
            team_last_game[team1] = (game_date, team2)
        if team2 not in team_last_game:
            team_last_game[team2] = (game_date, team1)
    
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
            
            # Show last game ELO if available
            if team in team_last_game:
                last_game_date, opponent = team_last_game[team]
                # Find ELOs from that date
                # Convert last_game_date to just the date part for lookup
                lookup_date = last_game_date.date()
                if lookup_date in date_to_elos:
                    historical_elos = date_to_elos[lookup_date]
                    # Calculate team ELO for that game
                    historical_players = [p for p in team_to_players[team] if p in historical_elos]
                    if historical_players:
                        last_game_elo = sum(historical_elos[p] for p in historical_players) / len(historical_players)
                        print(f"\nLast Game: vs {opponent} on {last_game_date.strftime('%m/%d/%Y %I:%M %p')}")
                        print(f"Team ELO in that game: {last_game_elo:.2f}")
                        print("Players in that game:")
                        for p in sorted(historical_players):
                            print(f"  {p}: {historical_elos[p]:.2f}")
            
            # Print current team ELO (average of current members)
            members = team_to_players[team]
            member_elos = [current_elos.get(p) for p in members if p in current_elos]
            if member_elos:
                avg_elo = sum(member_elos) / len(member_elos)
                print(f"Current Team Average ELO: {avg_elo:.2f}")
            else:
                print("No ELO data available for current roster")
            
            print("\nCurrent Roster:")
            for p in sorted(members):
                print(f"  {p}: {current_elos.get(p, 'N/A')}")

            # Show ELO history graph
            plot_team_elo(team, dates, elo_data, team_to_players, games)
            
        elif cmd.lower().startswith('player '):
            player = cmd[7:].strip()
            if player not in player_to_teams:
                print(f"Player '{player}' not found.")
                continue

            player_elo = current_elos.get(player)
            if player_elo is not None:
                # Get all ELOs and sort them
                all_elos = [(p, elo) for p, elo in current_elos.items()]
                all_elos.sort(key=lambda x: x[1], reverse=True)
                
                # Find rank
                rank = next(i + 1 for i, (p, _) in enumerate(all_elos) if p == player)
                total_players = len(all_elos)
                percentile = 100 * (total_players - rank + 1) / total_players
                
                print(f"\nPlayer {player}:")
                print(f"ELO: {player_elo:.2f}")
                print(f"Rank: #{rank} out of {total_players} ({percentile:.1f}th percentile)")
                

            else:
                print(f"\nPlayer {player} ELO: N/A")
            
            print(f"\nTeams:")
            for t in sorted(player_to_teams[player]):
                print(f"\n{t}:")
                
                # Calculate current team ELO
                members = team_to_players[t]
                current_member_elos = [current_elos.get(p) for p in members if p in current_elos]
                if current_member_elos:
                    current_avg = sum(current_member_elos) / len(current_member_elos)
                    print(f"  Current Team ELO: {current_avg:.2f}")
                else:
                    print("  Current Team ELO: N/A")
                
                # Show last game information
                if t in team_last_game:
                    last_game_date, opponent = team_last_game[t]
                    lookup_date = last_game_date.date()
                    if lookup_date in date_to_elos:
                        historical_elos = date_to_elos[lookup_date]
                        # Calculate historical team ELO
                        historical_players = [p for p in team_to_players[t] if p in historical_elos]
                        if historical_players:
                            last_game_elo = sum(historical_elos[p] for p in historical_players) / len(historical_players)
                            print(f"  Last Game ELO ({last_game_date.strftime('%m/%d/%Y')}): {last_game_elo:.2f}")
                else:
                    print("  No previous game data")

                        # Show ELO history graph
            if player in elo_data:
                plot_player_elo(player, dates, elo_data, team_to_players=team_to_players, games=games)
        else:
            print("Unknown command. Use 'team TEAMNAME' or 'player PLAYERNAME'.")

if __name__ == '__main__':
    main()
