
import csv
from collections import defaultdict
import os
from datetime import datetime, timedelta

try:
    from trueskill import TrueSkill, Rating
except Exception:
    raise ImportError("trueskill package not found. Install with: pip install trueskill")

# We'll map TrueSkill mu to a 1000-style scale so existing plots/CSV remain compatible.
# Default TrueSkill mu ~= 25. We'll map 25 -> 1000 (scale factor = 1000 / 25 = 40)
TS_MU = 25.0
SCALE_FACTOR = 1000.0 / TS_MU

# TrueSkill environment (defaults). We keep defaults but expose env if needed.
TS_ENV = TrueSkill()

# Multiplier for tournament games (applied to rating deltas)
TOURNAMENT_MULTIPLIER = 2.0

# Files
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

def rating_to_score(rating: Rating) -> float:
    """Map a TrueSkill Rating to the 1000-style score (mu scaled)."""
    return rating.mu * SCALE_FACTOR


def average_score(players, elos):
    """Average scaled mu for a team (returns 1000-style value)."""
    if not players:
        return SCALE_FACTOR * TS_MU
    return sum(rating_to_score(elos[p]) for p in players) / len(players)


def mov_factor(goal_diff):
    # keep roughly the same shape as before; returns a multiplier >= 1
    if goal_diff <= 0:
        return 1.0
    elif goal_diff == 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11.0 + goal_diff) / 8.0

def apply_trueskill_update(team1, team2, score1, score2, elos, margin_override=None, tournament_mult=1.0):
    """
    Rate teams using TrueSkill and apply a simple multiplier for margin/tournament by scaling
    the rating deltas (mu and sigma) before storing them back.
    """
    # Prepare team rating lists
    team1_ratings = [elos[p] for p in team1 if p in elos]
    team2_ratings = [elos[p] for p in team2 if p in elos]
    if not team1_ratings or not team2_ratings:
        return elos

    teams = [team1_ratings, team2_ratings]

    # Determine ranks for trueskill.rate: lower rank == better placement
    if score1 > score2:
        ranks = [0, 1]
    elif score1 < score2:
        ranks = [1, 0]
    else:
        ranks = [0, 0]

    # compute margin multiplier
    margin = abs(score1 - score2) if margin_override is None else margin_override
    mov_mult = mov_factor(margin)
    combined_mult = mov_mult * (tournament_mult if tournament_mult is not None else 1.0)

    # Rate using TrueSkill
    new_teams = TS_ENV.rate(teams, ranks=ranks)

    # Apply scaled deltas back to individual players
    for old_list, new_list, players in ((team1_ratings, new_teams[0], team1), (team2_ratings, new_teams[1], team2)):
        for i, old_rating in enumerate(old_list):
            new_rating = new_list[i]
            # scale deltas
            delta_mu = new_rating.mu - old_rating.mu
            delta_sigma = new_rating.sigma - old_rating.sigma
            scaled_mu = old_rating.mu + combined_mult * delta_mu
            scaled_sigma = max(1e-6, old_rating.sigma + combined_mult * delta_sigma)
            elos[players[i]] = Rating(scaled_mu, scaled_sigma)

    return elos


# legacy ELO functions removed - we use TrueSkill updates instead



# update_elo_MoV removed - TrueSkill handles update logic

def main():

    team_to_players = load_teams(TEAMS)
    games = load_games(GAMES)
    all_players = get_all_players(team_to_players)

    output_file = 'trueskill_results.csv'
    elo_history = []
    # Prepend a row 20 minutes before the earliest game with everyone at the scaled initial rating
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
        # create row with scaled initial rating for all players in consistent order
        initial_score = SCALE_FACTOR * TS_MU
        row_out = [initial_time_str] + [round(initial_score, 2) for p in all_players]
        elo_history.append(row_out)

    # Initialize player ratings as TrueSkill Ratings with mu=TS_MU, sigma=TS_ENV.sigma
    elos = defaultdict(lambda: Rating(TS_MU, TS_ENV.sigma))
    # Track team ELOs after each game
    team_elo_history = {team: [] for team in team_to_players}
    team_elo_change = {team: [] for team in team_to_players}
    all_game_elo_changes = []  # (abs_change, time, t1, t2, t1_elo_before, t1_elo_after, t2_elo_before, t2_elo_after, score, t1_change, t2_change)
    # Track record high/low for teams and players (store scaled scores)
    initial_score = SCALE_FACTOR * TS_MU
    player_high = {p: (initial_score, None) for p in all_players}  # (score, time)
    player_low = {p: (initial_score, None) for p in all_players}
    team_high = {team: (initial_score, None) for team in team_to_players}
    team_low = {team: (initial_score, None) for team in team_to_players}

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

        # Get team scores before game (scaled mu)
        team1_elo_before = average_score(team1, elos)
        team2_elo_before = average_score(team2, elos)

        # Tournament multiplier
        t_mult = TOURNAMENT_MULTIPLIER if tourney_flag.strip() else 1.0

        # Apply TrueSkill update (with margin/tournament scaling)
        apply_trueskill_update(team1, team2, s1, s2, elos, margin_override=margin, tournament_mult=t_mult)

        # Get team scores after game
        team1_elo_after = average_score(team1, elos)
        team2_elo_after = average_score(team2, elos)

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
            p_score = rating_to_score(elos[p])
            if player_high[p][1] is None or p_score > player_high[p][0]:
                player_high[p] = (p_score, time)
            if player_low[p][1] is None or p_score < player_low[p][0]:
                player_low[p] = (p_score, time)

        # Save scaled mu values after this game for all players
        row_out = [time] + [round(rating_to_score(elos[p]), 2) for p in all_players]
        elo_history.append(row_out)
    # Get current team ELOs
    current_team_elos = {}
    last_game_team_elos = {}
    for team, players in team_to_players.items():
        if players:
            # Current ELO
            valid_players = [p for p in players if p in elos]
            if valid_players:
                current_team_elos[team] = sum(rating_to_score(elos[p]) for p in valid_players) / len(valid_players)
            
            # Last game ELO
            if team_elo_history[team]:  # If the team has played any games
                # team_elo_history stores scaled scores already
                last_game_team_elos[team] = team_elo_history[team][-1][1]

    # Print top 10 teams by current ELO
    print("\nTop 10 Teams by Current ELO:")
    for team, elo in sorted(current_team_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{team}: {elo:.2f}")
    
    # Print bottom 10 teams by current ELO
    print("\nBottom 10 Teams by Current ELO:")
    for team, elo in sorted(current_team_elos.items(), key=lambda x: x[1])[:10]:
        print(f"{team}: {elo:.2f}")

    # Print top 10 teams by last game ELO
    print("\nTop 10 Teams by Last Game ELO:")
    for team, elo in sorted(last_game_team_elos.items(), key=lambda x: x[1], reverse=True)[:10]:
        last_game_time = team_elo_history[team][-1][0]  # Get the time of their last game
        print(f"{team}: {elo:.2f} (last played {last_game_time})")
    
    # Print bottom 10 teams by last game ELO
    print("\nBottom 10 Teams by Last Game ELO:")
    for team, elo in sorted(last_game_team_elos.items(), key=lambda x: x[1])[:10]:
        last_game_time = team_elo_history[team][-1][0]  # Get the time of their last game
        print(f"{team}: {elo:.2f} (last played {last_game_time})")

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
    # Sorted players by scaled score
    sorted_players = sorted(((p, rating_to_score(elos[p])) for p in all_players), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Players by Score:")
    for player, score in sorted_players[:10]:
        print(f"{player}: {round(score,2)}")
    print("\nBottom 10 Players by Score:")
    for player, score in sorted_players[-1:-11:-1]:
        print(f"{player}: {round(score,2)}")


    # Print all teams with both current and last game ELOs
    print("\nAll Team ELOs (Current / Last Game):")
    # Use the original order from the CSV file
    for team in team_to_players.keys():  # This maintains the order from the Teams CSV
        players = team_to_players[team]
        if not players:
            print(f"{team}: No players")
            continue
            
        # Get current ELO (scaled)
        valid_players = [p for p in players if p in elos]
        if valid_players:
            current_elo = sum(rating_to_score(elos[p]) for p in valid_players) / len(valid_players)
        else:
            current_elo = None
            
        # Get last game ELO
        last_game_elo = None
        last_game_time = None
        if team_elo_history[team]:
            last_game_time, last_game_elo = team_elo_history[team][-1]
            
        # Format the output
        current_str = f"{current_elo:.2f}" if current_elo is not None else "N/A"
        if last_game_elo is not None:
            last_game_str = f"{last_game_elo:.2f} (last played {last_game_time})"
        else:
            last_game_str = "No games played"
            
        print(f"{team}:")
        print(f"  Current ELO: {current_str}")
        print(f"  Last Game:   {last_game_str}")


if __name__ == '__main__':
    main()
