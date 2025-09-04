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

def plot_player_elo(player, dates, elo_data, team_to_players=None, games=None, team_first_last=None):
    # Make room on the right for an external legend (match team plot layout)
    plt.figure(figsize=(15, 6))
    
    # Create main plot area (use 60% width so legend can sit on the right)
    plt.axes([0.1, 0.1, 0.6, 0.8])
    
    # Determine player's active window (based on games for teams they belong to)
    start_idx = 0
    end_idx = len(dates) - 1
    if team_to_players and games:
        player_teams = [t for t, members in team_to_players.items() if player in members]
        player_team_games = [g for g in games if g[1] in player_teams or g[2] in player_teams]
        if player_team_games:
            first_game = min(g[0] for g in player_team_games)
            last_game = max(g[0] for g in player_team_games)
            try:
                start_idx = next(i for i, d in enumerate(dates) if d >= first_game)
            except StopIteration:
                start_idx = 0
            if start_idx > 0:
                start_idx -= 1
            try:
                end_idx = max(i for i, d in enumerate(dates) if d <= last_game)
            except ValueError:
                end_idx = len(dates) - 1
            if end_idx < len(dates) - 1:
                end_idx += 1

    # Slice dates and player ELOs to active window
    plot_dates = dates[start_idx:end_idx + 1]
    plot_elos = elo_data[player][start_idx:end_idx + 1]

    # Plot player line for the active window only
    plt.plot(plot_dates, plot_elos, 'b-', label='ELO')
    # mark player's first and last ELO within the window with dots (no legend entries)
    plt.scatter(plot_dates[0], plot_elos[0], color='b', s=50, zorder=4, label='_nolegend_',
                edgecolors='white', linewidths=0.8)
    plt.scatter(plot_dates[-1], plot_elos[-1], color='b', s=50, zorder=4, label='_nolegend_',
                edgecolors='white', linewidths=0.8)
    # annotate the ending ELO value
    try:
        plt.gca().annotate(f"{plot_elos[-1]:.0f}", xy=(plot_dates[-1], plot_elos[-1]), xytext=(-6, 4),
                           textcoords='offset points', ha='right', va='bottom', fontsize='small', color='b',
                           clip_on=False, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    except Exception:
        pass
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
            # use precomputed first/last game for the team if available
            fl = None
            if team_first_last is not None:
                fl = team_first_last.get(team)
            # fallback to scanning games if mapping not provided
            if fl is None and games is not None:
                team_games = [g for g in games if g[1] == team or g[2] == team]
                if team_games:
                    fl = (min(g[0] for g in team_games), max(g[0] for g in team_games))

            if not fl:
                continue
            first_game, last_game = fl

            # determine indices for team's date range: start = immediately before first_game, end = at last_game
            try:
                team_start_idx = next(i for i, d in enumerate(dates) if d >= first_game)
            except StopIteration:
                team_start_idx = 0
            if team_start_idx > 0:
                team_start_idx -= 1
            try:
                team_end_idx = max(i for i, d in enumerate(dates) if d <= last_game)
            except ValueError:
                team_end_idx = len(dates) - 1

            # clip team's window to player's active window
            clip_start = max(team_start_idx, start_idx)
            clip_end = min(team_end_idx, end_idx)
            if clip_start > clip_end:
                continue  # no overlap with player's window

            # compute team average over the clipped window
            members = [p for p in team_to_players.get(team, []) if p in elo_data]
            if not members:
                continue
            team_segment = [sum(elo_data[p][i] for p in members) / len(members) for i in range(clip_start, clip_end + 1)]

            seg_dates = dates[clip_start:clip_end + 1]
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
            # annotate segment endpoints with ELO values
            try:
                plt.gca().annotate(f"{y_first_seg:.0f}", xy=(seg_dates[0], y_first_seg), xytext=(-6, -4),
                                   textcoords='offset points', ha='right', va='top', fontsize='x-small', color=color,
                                   clip_on=False, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
                plt.gca().annotate(f"{y_last_seg:.0f}", xy=(seg_dates[-1], y_last_seg), xytext=(6, -4),
                                   textcoords='offset points', ha='left', va='top', fontsize='x-small', color=color,
                                   clip_on=False, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
            except Exception:
                pass
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

    # Restrict x-axis to the player's active window
    plt.xlim(plot_dates[0], plot_dates[-1])

    # Place legend outside of the plot on the right (includes player, starting ELO, and team segments)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize='small')

    plt.show()

def plot_team_elo(team, dates, elo_data, team_to_players, games, team_first_last=None):
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
        # annotate the scatter points with ELO values
        try:
            plt.gca().annotate(f"{y_first:.0f}", xy=(dates[0], y_first), xytext=(-6, -2), textcoords='offset points',
                               ha='right', va='top', fontsize='x-small', color=color, clip_on=False,
                               path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        except Exception:
            pass
        # record label position and elo for later collision avoidance (one entry per player)
        label_positions.append((y_last, player, color, y_last))

    # Simple collision avoidance for right-edge labels: sort by y and nudge if too close
    if label_positions:
        label_positions.sort(key=lambda x: x[0])
        adjusted_positions = []
        min_sep = max(6, (plt.ylim()[1] - plt.ylim()[0]) * 0.02)  # minimal separation in data units
        for y, player, color, elo_val in label_positions:
            if not adjusted_positions:
                adjusted_positions.append([y, player, color, elo_val])
            else:
                prev_y = adjusted_positions[-1][0]
                if y - prev_y < min_sep:
                    y = prev_y + min_sep
                adjusted_positions.append([y, player, color, elo_val])

        # place annotations using adjusted positions
        for y, player, color, elo_val in adjusted_positions:
            # combine player name and elo into one inline label
            label_text = f"{player} {elo_val:.0f}"
            plt.gca().annotate(
                label_text,
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
    
    # Find team's first and last game and compute the team's active window
    fl = None
    if team_first_last is not None:
        fl = team_first_last.get(team)
    if fl is None:
        team_games = [game for game in games if game[1] == team or game[2] == team]
        if team_games:
            fl = (min(game[0] for game in team_games), max(game[0] for game in team_games))

    team_start_idx = 0
    team_end_idx = len(dates) - 1
    if fl:
        first_game, last_game = fl

        # determine indices for team's date range (expand one step before/after if available)
        try:
            team_start_idx = next(i for i, d in enumerate(dates) if d >= first_game)
        except StopIteration:
            team_start_idx = 0
        if team_start_idx > 0:
            team_start_idx -= 1
        try:
            team_end_idx = max(i for i, d in enumerate(dates) if d <= last_game)
        except ValueError:
            team_end_idx = len(dates) - 1

    # Plot team average with thick line
    plt.plot(dates, team_elos, 'k-', linewidth=2.5, label=f'{team} (Team Average)')

    # If we know first/last game, mark the team ELO at the team's window boundaries and shade background
    if fl:
        # Use the computed start/end indices for consistent windowing behavior
        before_idx = max(0, min(team_start_idx, len(team_elos) - 1))
        after_idx = max(0, min(team_end_idx, len(team_elos) - 1))

        # plot internal markers corresponding to the background/window boundaries
        plt.scatter(dates[before_idx], team_elos[before_idx], color='k', s=70, zorder=5, label='_nolegend_',
                    edgecolors='white', linewidths=0.8)
        plt.scatter(dates[after_idx], team_elos[after_idx], color='k', s=70, zorder=5, label='_nolegend_',
                    edgecolors='white', linewidths=0.8)
        # annotate the internal boundary markers with ELO values
        try:
            plt.gca().annotate(f"{team_elos[before_idx]:.0f}", xy=(dates[before_idx], team_elos[before_idx]),
                               xytext=(-6, -6), textcoords='offset points', ha='right', va='top', fontsize='small',
                               color='k', clip_on=False, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
            plt.gca().annotate(f"{team_elos[after_idx]:.0f}", xy=(dates[after_idx], team_elos[after_idx]),
                               xytext=(6, -6), textcoords='offset points', ha='left', va='top', fontsize='small',
                               color='k', clip_on=False, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        except Exception:
            pass

        # Plot background colors: shade exactly from the team's first game to last game
        # Use a lighter grey before the active window and a slightly darker grey after it
        plt.axvspan(dates[0], first_game, color='#f5f5f5', alpha=0.4, label='Before First Game')
        plt.axvspan(first_game, last_game, color='lightblue', alpha=0.3, label='Active Period')
        plt.axvspan(last_game, dates[-1], color='#d0d0d0', alpha=0.25, label='After Last Game')

    # Mark the start and end of the team average with dots
    plt.scatter(dates[0], team_elos[0], color='k', s=50, zorder=4, label='_nolegend_', edgecolors='white', linewidths=0.8)
    plt.scatter(dates[-1], team_elos[-1], color='k', s=50, zorder=4, label='_nolegend_', edgecolors='white', linewidths=0.8)
    # annotate the global start/end team-average markers
    try:
        plt.gca().annotate(f"{team_elos[0]:.0f}", xy=(dates[0], team_elos[0]), xytext=(6, 2), textcoords='offset points',
                           ha='left', va='bottom', fontsize='small', color='k', clip_on=False,
                           path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        plt.gca().annotate(f"{team_elos[-1]:.0f}", xy=(dates[-1], team_elos[-1]), xytext=(-6, 2), textcoords='offset points',
                           ha='right', va='bottom', fontsize='small', color='k', clip_on=False,
                           path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    except Exception:
        pass
    
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

def plot_all_teams(dates, elo_data, team_to_players, games, team_first_last=None):
    """Plot all teams' average ELO during their active windows on the same graph.
    Each team's segment runs from the date immediately before its first game to the date
    immediately after its last game (clamped to available dates). Endpoints are marked and
    team names are annotated inline.
    """
    plt.figure(figsize=(15, 8))
    plt.axes([0.1, 0.1, 0.6, 0.8])

    teams = sorted(team_to_players.keys())
    cmap = plt.get_cmap('tab20')

    label_items = []  # collect annotation targets grouped by end-date

    for ti, team in enumerate(teams):
        # get canonical first/last game for the team if provided
        fl = None
        if team_first_last is not None:
            fl = team_first_last.get(team)
        # fallback to scanning games
        if fl is None:
            team_games = [g for g in games if g[1] == team or g[2] == team]
            if not team_games:
                continue
            fl = (min(g[0] for g in team_games), max(g[0] for g in team_games))

        if not fl:
            continue
        first_game, last_game = fl

        # find start index as the date immediately before first_game (if available)
        try:
            start_idx = next(i for i, d in enumerate(dates) if d >= first_game)
        except StopIteration:
            start_idx = 0
        if start_idx > 0:
            start_idx = start_idx - 1
        # end index is at the last_game (do NOT expand after last game)
        try:
            end_idx = max(i for i, d in enumerate(dates) if d <= last_game)
        except ValueError:
            end_idx = len(dates) - 1

        members = [p for p in team_to_players.get(team, []) if p in elo_data]
        if not members:
            continue

        team_segment = [sum(elo_data[p][i] for p in members) / len(members) for i in range(start_idx, end_idx + 1)]
        seg_dates = dates[start_idx:end_idx + 1]
        color = cmap(ti % 20)

        plt.plot(seg_dates, team_segment, color=color, linestyle='-', linewidth=2, label='_nolegend_')
        # endpoint markers
        plt.scatter(seg_dates[0], team_segment[0], color=color, s=50, zorder=4, label='_nolegend_', edgecolors='white', linewidths=0.8)
        plt.scatter(seg_dates[-1], team_segment[-1], color=color, s=50, zorder=4, label='_nolegend_', edgecolors='white', linewidths=0.8)

        # annotate endpoints with values
        try:
            plt.gca().annotate(f"{team_segment[0]:.0f}", xy=(seg_dates[0], team_segment[0]), xytext=(6, 2),
                               textcoords='offset points', ha='left', va='bottom', fontsize='x-small', color=color,
                               clip_on=False, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        except Exception:
            pass

        # queue annotation at the segment end; group by end-date (include the value so label shows name + value)
        label_items.append({'x': seg_dates[-1], 'y': team_segment[-1], 'team': team, 'color': color, 'val': team_segment[-1]})

    # Simple collision avoidance per end-date group: group by x and nudge y if too close
    if label_items:
        # group by end-date
        groups = {}
        for it in label_items:
            groups.setdefault(it['x'], []).append(it)

        for x, group in groups.items():
            # sort by y and nudge
            group.sort(key=lambda g: g['y'])
            adjusted = []
            min_sep = max(6, (plt.ylim()[1] - plt.ylim()[0]) * 0.02)
            for item in group:
                y = item['y']
                if not adjusted:
                    adjusted.append([y, item])
                else:
                    prev_y = adjusted[-1][0]
                    if y - prev_y < min_sep:
                        y = prev_y + min_sep
                    adjusted.append([y, item])

            # annotate adjusted positions
            for y_adj, item in adjusted:
                label_text = f"{item['team']} {item.get('val', 0):.0f}"
                plt.gca().annotate(
                    label_text,
                    xy=(item['x'], y_adj),
                    xytext=(6, 0),
                    textcoords='offset points',
                    va='center',
                    fontsize='small',
                    color=item['color'],
                    clip_on=False,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
                )

    plt.title('All Teams: Active Windows')
    plt.xlabel('Date')
    plt.ylabel('Team Average ELO')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
    plt.margins(y=0.1)
    plt.show()


def plot_top_players(dates, elo_data, players):
    """Plot the given players' ELO across the entire date span.
    players: iterable of player names (must exist in elo_data)
    """
    plt.figure(figsize=(15, 8))
    plt.axes([0.1, 0.1, 0.6, 0.8])

    players = [p for p in players if p in elo_data]
    if not players:
        print('No players to plot')
        return

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(players))]

    label_items = []
    for i, player in enumerate(players):
        y = elo_data[player]
        plt.plot(dates, y, color=colors[i], linewidth=1.8, label='_nolegend_')
        # mark endpoints
        plt.scatter(dates[0], y[0], color=colors[i], s=40, zorder=4, edgecolors='white', linewidths=0.7)
        plt.scatter(dates[-1], y[-1], color=colors[i], s=40, zorder=4, edgecolors='white', linewidths=0.7)
        # queue inline annotation at right edge
        label_items.append({'x': dates[-1], 'y': y[-1], 'player': player, 'color': colors[i], 'elo': y[-1]})

    # collision avoidance for right-edge labels
    if label_items:
        groups = {}
        for it in label_items:
            groups.setdefault(it['x'], []).append(it)

        for x, group in groups.items():
            group.sort(key=lambda g: g['y'])
            adjusted = []
            min_sep = max(6, (plt.ylim()[1] - plt.ylim()[0]) * 0.02)
            for item in group:
                y = item['y']
                if not adjusted:
                    adjusted.append([y, item])
                else:
                    prev_y = adjusted[-1][0]
                    if y - prev_y < min_sep:
                        y = prev_y + min_sep
                    adjusted.append([y, item])

            for y_adj, item in adjusted:
                text = f"{item['player']} {item['elo']:.0f}"
                plt.gca().annotate(
                    text,
                    xy=(item['x'], y_adj),
                    xytext=(6, 0),
                    textcoords='offset points',
                    va='center',
                    fontsize='small',
                    color=item['color'],
                    clip_on=False,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')]
                )

    plt.title('Players who were highest ELO at any point')
    plt.xlabel('Date')
    plt.ylabel('ELO')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
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

    # Compute canonical first and last game datetimes for each team (used by plotting)
    team_first_last = {}
    for game_date, team1, team2 in games:
        team_first_last.setdefault(team1, [game_date, game_date])
        team_first_last.setdefault(team2, [game_date, game_date])
        # update first/last
        if game_date < team_first_last[team1][0]:
            team_first_last[team1][0] = game_date
        if game_date > team_first_last[team1][1]:
            team_first_last[team1][1] = game_date
        if game_date < team_first_last[team2][0]:
            team_first_last[team2][0] = game_date
        if game_date > team_first_last[team2][1]:
            team_first_last[team2][1] = game_date
    # convert lists to tuples for immutability
    for t, fl in list(team_first_last.items()):
        team_first_last[t] = (fl[0], fl[1])
    
    print("Type 'team TEAMNAME' to see ELOs of all members, 'player PLAYERNAME' to see all teams and ELOs for that player, or 'teams' to plot all teams. Type 'exit' to quit.")
    while True:
        cmd = input('> ').strip()
        if cmd.lower() == 'exit':
            break
        if cmd.lower() == 'players':
            # find the uniquely-highest player at each timestamp (ignore ties)
            top_players = set()
            for i, d in enumerate(dates):
                # build list of (player, elo) for this date
                row = [(p, elo_data[p][i]) for p in elo_data.keys()]
                # find max elo
                max_elo = max(v for _, v in row)
                # find all players with max_elo
                leaders = [p for p, v in row if v == max_elo]
                # if unique leader, record
                if len(leaders) == 1:
                    top_players.add(leaders[0])
            if not top_players:
                print('No unique top players found (all ties)')
            else:
                print(f"Found {len(top_players)} players who were uniquely top at some point")
                plot_top_players(dates, elo_data, sorted(top_players))
            continue
        if cmd.lower() == 'teams':
            plot_all_teams(dates, elo_data, team_to_players, games, team_first_last=team_first_last)
            continue
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
            plot_team_elo(team, dates, elo_data, team_to_players, games, team_first_last=team_first_last)
            
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
                plot_player_elo(player, dates, elo_data, team_to_players=team_to_players, games=games, team_first_last=team_first_last)
        else:
            print("Unknown command. Use 'team TEAMNAME' or 'player PLAYERNAME'.")

if __name__ == '__main__':
    main()
