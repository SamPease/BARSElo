"""
Dash app to explore ELO results and teams.

- Loads `elo_results.csv`, `Sports Elo - Teams.csv`, and `Sports Elo - Games.csv`.
- Shows a Teams tab with a sortable DataTable containing: Team, Current ELO, Last Game ELO,
  Beginning ELO, Win%, Games. Columns are sortable and you can click a team name.
- Clicking a team updates the URL to /team/<teamname> and shows a detail view with:
  - The matplotlib plot (same look as the existing scripts) rendered to PNG and embedded
  - Team roster, current elo, last-game elo, beginning elo, win/loss/draw, and ranks for the
            # label the team name at the right edge of the plot so it's visible even
            # when the team's segment doesn't extend to the plot edge. Use xref='x domain'.
            try:
                fig.add_annotation(
                    x=0.98,
                    xref='x domain',
                    y=team_series.values[-1],
                    yref='y',
                    text=team,
                    showarrow=False,
                    xanchor='left',
                    font=dict(color=color, size=12),
                    bgcolor='rgba(255,255,255,0)'
                )
            except Exception:
                pass
    columns shown in the table.

Notes:
- This app uses pandas, dash, and matplotlib. If you prefer a fully interactive Plotly chart
  instead of a PNG, I can convert the plotting code.
- Save this file as `dash_app.py` and run with: `python dash_app.py` (it will start the Dash dev server).

"""

import base64
import io
import urllib.parse
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import csv

# Constants - file names (same as other scripts)
ELO_RESULTS = 'trueskill_results.csv'
TEAMS_FILE = 'Sports Elo - Teams.csv'
GAMES_FILE = 'Sports Elo - Games.csv'
INITIAL_ELO = 1000

# Utility loaders (simplified, robust to missing/empty rows)

def load_elo_results(filename=ELO_RESULTS):
    # returns (players, dates, elo_data_df)
    df = pd.read_csv(filename, parse_dates=['Time'])
    # Ensure Time column exists
    if 'Time' not in df.columns:
        # fall back to first column
        df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'])
    players = [c for c in df.columns if c != 'Time']
    dates = list(df['Time'])
    elo_df = df.set_index('Time')
    return players, dates, elo_df


def load_teams(filename=TEAMS_FILE):
    # Teams file is CSV where columns are teams and rows are players (cells empty if none)
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # initialize keys
        teams = reader.fieldnames
        team_to_players = {t: [] for t in teams}
        for row in reader:
            for t in teams:
                v = row.get(t, '')
                if v and v.strip():
                    team_to_players[t].append(v.strip())
    return team_to_players


def load_games(filename=GAMES_FILE):
    # Returns list of (datetime, team1, team2) and list with scores when present
    games = []
    games_with_scores = []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # Expect first columns: time, team1, score1, team2, score2, ...
            if len(row) < 2:
                continue
            time_str = row[0].strip()
            t1 = row[1].strip() if len(row) > 1 else ''
            s1 = None
            t2 = ''
            s2 = None
            if len(row) > 2:
                s1 = row[2].strip()
            if len(row) > 3:
                t2 = row[3].strip()
            if len(row) > 4:
                s2 = row[4].strip()
            # parse time
            dt = None
            try:
                dt = datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S')
            except Exception:
                try:
                    dt = datetime.strptime(time_str, '%m/%d/%Y')
                except Exception:
                    dt = None
            if dt is None:
                continue
            games.append((dt, t1, t2))
            try:
                s1i = int(s1) if s1 not in (None, '') else None
                s2i = int(s2) if s2 not in (None, '') else None
                if s1i is not None and s2i is not None:
                    games_with_scores.append((dt, t1, s1i, t2, s2i))
            except Exception:
                pass
    games.sort()
    games_with_scores.sort()
    return games, games_with_scores


# Compute team statistics using elo df and game results

def compute_team_stats(team_to_players, players, dates, elo_df, games, games_with_scores):
    # players: list of player names (columns in elo_df)
    # elo_df: DataFrame indexed by Time with player columns
    # dates: list of datetimes (index order)

    # current elos: last row of elo_df
    if elo_df.shape[0] == 0:
        raise ValueError('No ELO data available')
    last_row = elo_df.iloc[-1]
    current_elos = {p: float(last_row[p]) for p in players if p in elo_df.columns}

    # team derived stats
    team_stats = {}
    # prepare team_first_last maps
    team_first_last = {}
    for dt, t1, t2 in games:
        if not t1 or not t2:
            continue
        team_first_last.setdefault(t1, [dt, dt])
        team_first_last.setdefault(t2, [dt, dt])
        if dt < team_first_last[t1][0]:
            team_first_last[t1][0] = dt
        if dt > team_first_last[t1][1]:
            team_first_last[t1][1] = dt
        if dt < team_first_last[t2][0]:
            team_first_last[t2][0] = dt
        if dt > team_first_last[t2][1]:
            team_first_last[t2][1] = dt
    for t, fl in list(team_first_last.items()):
        team_first_last[t] = (fl[0], fl[1])

    # compute team_elo_history from elo_df: for each team, the average of member columns at each date
    team_elo_history = {}
    for team, members in team_to_players.items():
        valid_members = [m for m in members if m in elo_df.columns]
        if valid_members:
            # compute mean across valid_members for each timestamp
            team_elo_history[team] = elo_df[valid_members].mean(axis=1)
        else:
            # fill with INITIAL_ELO
            team_elo_history[team] = pd.Series([INITIAL_ELO] * elo_df.shape[0], index=elo_df.index)

    # compute last game datetime for each team by scanning all games and taking the latest datetime
    team_last_game = {}
    for dt, t1, t2 in games:
        # normalize team names via strip
        if t1:
            t1n = t1.strip()
            if t1n:
                if t1n not in team_last_game or dt > team_last_game[t1n]:
                    team_last_game[t1n] = dt
        if t2:
            t2n = t2.strip()
            if t2n:
                if t2n not in team_last_game or dt > team_last_game[t2n]:
                    team_last_game[t2n] = dt

    # compute win/loss/draw records from games_with_scores
    team_record = {t: {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0} for t in team_to_players.keys()}
    for dt, t1, s1, t2, s2 in games_with_scores:
        if not t1 or not t2:
            continue
        team_record.setdefault(t1, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
        team_record.setdefault(t2, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
        team_record[t1]['games'] += 1
        team_record[t2]['games'] += 1
        if s1 > s2:
            team_record[t1]['wins'] += 1
            team_record[t2]['losses'] += 1
        elif s2 > s1:
            team_record[t2]['wins'] += 1
            team_record[t1]['losses'] += 1
        else:
            team_record[t1]['draws'] += 1
            team_record[t2]['draws'] += 1

    # assemble table rows
    rows = []
    for team, members in team_to_players.items():
        # current elo
        if members:
            member_elos = [current_elos.get(m) for m in members if m in current_elos]
            current_avg = float(np.mean(member_elos)) if member_elos else None
        else:
            current_avg = None
        # last game elo: take last game date group and sample team_elo_history at that index
        # lookup last game using stripped team name (games file may have slightly different spacing)
        last_game_dt = team_last_game.get(team.strip()) if isinstance(team, str) else team_last_game.get(team)
        last_game_elo = None
        last_played_str = None
        if last_game_dt is not None:
            # Normalize to pandas Timestamp for robust comparison with series index
            try:
                last_ts = pd.to_datetime(last_game_dt)
            except Exception:
                last_ts = None
            if last_ts is not None:
                idx = team_elo_history[team].index
                # Try exact match first
                try:
                    # if the last game is after our ELO data, use last available ELO and mark
                    if last_ts > idx[-1]:
                        last_game_elo = float(team_elo_history[team].iloc[-1])
                    elif last_ts in idx:
                        # .loc may return multiple rows if the index contains duplicates; take the last
                        val = team_elo_history[team].loc[last_ts]
                        try:
                            # if val is a Series (duplicate timestamps), take the last entry
                            if isinstance(val, (pd.Series, pd.DataFrame)):
                                val = val.iloc[-1]
                            last_game_elo = float(val)
                        except Exception:
                            last_game_elo = None
                    else:
                        # find last index <= last_ts
                        pos = idx.searchsorted(last_ts, side='right')
                        if pos > 0:
                            last_game_elo = float(team_elo_history[team].iloc[pos - 1])
                        else:
                            # no earlier index, take first available
                            last_game_elo = float(team_elo_history[team].iloc[0])
                except Exception:
                    last_game_elo = None
                # format last played datetime for display
                try:
                    last_played_str = last_ts.strftime('%m/%d/%Y %H:%M:%S')
                    # annotate if the last game is after the last ELO snapshot
                    try:
                        if elo_df is not None and not elo_df.empty and last_ts > elo_df.index[-1]:
                            last_played_str += ' (after last ELO)'
                    except Exception:
                        pass
                except Exception:
                    last_played_str = str(last_game_dt)
        # beginning elo: elo immediately before first game (we define as the value one step before first_game index if exists)
        beg_elo = None
        if team in team_first_last:
            first_game = team_first_last[team][0]
            try:
                # find first index >= first_game, then step back one if possible
                idxs = list(team_elo_history[team].index)
                pos = next(i for i, d in enumerate(idxs) if d >= first_game)
                sample_idx = max(0, pos - 1)
                beg_elo = float(team_elo_history[team].iloc[sample_idx])
            except StopIteration:
                beg_elo = float(team_elo_history[team].iloc[0])
            except Exception:
                beg_elo = None
        else:
            # team never played: set beginning equal to current if available else INITIAL_ELO
            if current_avg is not None:
                beg_elo = current_avg
            else:
                beg_elo = float(INITIAL_ELO)

        rec = team_record.get(team, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
        games_played = rec.get('games', 0)
        wins = rec.get('wins', 0)
        draws = rec.get('draws', 0)
        win_pct = (wins + 0.5 * draws) / games_played if games_played > 0 else None

        rows.append({
            'team': team,
            'current_elo': None if current_avg is None else round(current_avg, 2),
            'last_game_elo': None if last_game_elo is None else round(last_game_elo, 2),
            'beginning_elo': None if beg_elo is None else round(beg_elo, 2),
            'win_pct': None if win_pct is None else round(win_pct, 3),
            'wins': wins,
            'losses': rec.get('losses', 0),
            'draws': draws,
            'games': games_played,
            'members': members,
            'last_played': last_played_str,
            # keep a datetime version for sorting (hidden from table)
            'last_played_dt': last_game_dt
        })

    # compute ranks for shown metrics (1 = best/highest)
    # For None values, put NaN so ranking excludes them
    df = pd.DataFrame(rows)
    # sort teams by most recent game (descending). Rows with no last_played_dt go to the end.
    if 'last_played_dt' in df.columns:
        try:
            df = df.sort_values(by='last_played_dt', ascending=False, na_position='last').reset_index(drop=True)
        except Exception:
            # fallback: if sorting fails, leave as-is
            pass
    for col in ['current_elo', 'last_game_elo', 'beginning_elo', 'win_pct']:
        df[col + '_rank'] = df[col].rank(ascending=False, method='min', na_option='keep')

    return df, team_elo_history, team_first_last, team_record
    


# Plotting: reuse the plotting style from elo_explore but render to PNG and return base64

def build_team_figure(team, dates, team_elos_series, team_to_players, elo_df, team_first_last=None):
    """Return a Plotly figure for the team with individual players and team average."""
    fig = go.Figure()

    members = team_to_players.get(team, [])
    valid_players = [p for p in members if p in elo_df.columns]

    # index and bounds
    idx = elo_df.index
    n = len(idx)

    # Determine team's first/last game timestamps
    if team_first_last and team in team_first_last:
        try:
            team_first_game, team_last_game = team_first_last[team]
        except Exception:
            team_first_game = None
            team_last_game = None
    else:
        team_first_game = None
        team_last_game = None

    # Compute per-player first/last game timestamps (across all teams the player has been on)
    player_windows = {}
    latest_player_end_pos = 0
    for p in valid_players:
        # find all teams the player has been on
        player_teams = [t for t, mem in team_to_players.items() if p in mem]
        p_games = [g for g in globals().get('games', []) if (g[1] in player_teams) or (g[2] in player_teams)]
        if p_games:
            p_first = min(g[0] for g in p_games)
            p_last = max(g[0] for g in p_games)
            p_start_pos = idx.searchsorted(p_first, side='left') - 1
            if p_start_pos < 0:
                p_start_pos = 0
            p_end_pos = idx.searchsorted(p_last, side='right') - 1
            if p_end_pos < 0:
                p_end_pos = n - 1
        else:
            # default to full range
            p_start_pos = 0
            p_end_pos = n - 1
        player_windows[p] = (p_start_pos, p_end_pos)
        if p_end_pos > latest_player_end_pos:
            latest_player_end_pos = p_end_pos

    # Axis window: start before team's first game, end at last player last-game position
    if team_first_game is not None:
        axis_start_pos = idx.searchsorted(team_first_game, side='left') - 1
        if axis_start_pos < 0:
            axis_start_pos = 0
    else:
        axis_start_pos = 0
    axis_end_pos = max(latest_player_end_pos, axis_start_pos)
    axis_dates = idx[axis_start_pos:axis_end_pos+1]

    # plot each player over their own window
    palette = px.colors.qualitative.Plotly
    for i, p in enumerate(valid_players):
        col = palette[i % len(palette)]
        p_start, p_end = player_windows.get(p, (0, n - 1))
        seg_idx = idx[p_start:p_end+1]
        try:
            series = elo_df[p].iloc[p_start:p_end+1]
            y = series.values
        except Exception:
            y = np.array([INITIAL_ELO] * len(seg_idx))
        # plot player trace with legend (clickable)
        fig.add_trace(go.Scatter(x=seg_idx, y=y, mode='lines', line=dict(color=col, width=1.5), name=p, hovertemplate='%{x|%m/%d/%Y %H:%M}: %{y:.0f}<extra>'+p+'</extra>', showlegend=True, opacity=0.85))
        # markers and annotations: before team's first game, after team's last game, and after player's last game
        try:
            # value at team's before timestamp (if within player's range)
            if axis_start_pos >= p_start and axis_start_pos <= p_end:
                v_before_team = elo_df[p].iloc[axis_start_pos]
                fig.add_trace(go.Scatter(x=[idx[axis_start_pos]], y=[v_before_team], mode='markers', marker=dict(color=col, size=8), showlegend=False))
                fig.add_annotation(x=idx[axis_start_pos], y=float(v_before_team), text=f"{float(v_before_team):.0f}", showarrow=False, yshift=14, font=dict(color=col, size=12), bgcolor='rgba(255,255,255,0)')
            # value at team's after-last-game timestamp (if team_last_game within player's range)
            if team_last_game is not None:
                team_last_pos = idx.searchsorted(team_last_game, side='right') - 1
                if team_last_pos >= p_start and team_last_pos <= p_end:
                    v_after_team = elo_df[p].iloc[team_last_pos]
                    fig.add_trace(go.Scatter(x=[idx[team_last_pos]], y=[v_after_team], mode='markers', marker=dict(color=col, size=8), showlegend=False))
                    fig.add_annotation(x=idx[team_last_pos], y=float(v_after_team), text=f"{float(v_after_team):.0f}", showarrow=False, yshift=14, font=dict(color=col, size=12), bgcolor='rgba(255,255,255,0)')
            # value at player's own last game
            v_player_last = elo_df[p].iloc[p_end]
            fig.add_trace(go.Scatter(x=[idx[p_end]], y=[v_player_last], mode='markers', marker=dict(color=col, size=8), showlegend=False))
            # annotate player's last value (if different from team last)
            fig.add_annotation(x=idx[p_end], y=float(v_player_last), text=f"{float(v_player_last):.0f}", showarrow=False, yshift=14, font=dict(color=col, size=12), bgcolor='rgba(255,255,255,0)')
            # left-side unlabeled marker at player's start (avoid duplicate if same as team's before marker)
            try:
                if not (axis_start_pos >= p_start and axis_start_pos <= p_end):
                    fig.add_trace(go.Scatter(x=[idx[p_start]], y=[y[0]], mode='markers', marker=dict(color=col, size=6), showlegend=False))
            except Exception:
                pass
        except Exception:
            pass

    # Plot team average from before team's first game (axis_start_pos) to axis_end_pos
    try:
        team_series_full = team_elos_series.iloc[axis_start_pos:axis_end_pos+1]
        fig.add_trace(go.Scatter(x=axis_dates, y=team_series_full.values, mode='lines+markers', line=dict(color='black', width=3), marker=dict(size=9), name=f'{team} (Team Average)', hovertemplate='%{x|%m/%d/%Y %H:%M}: %{y:.0f}<extra>'+team+'</extra>', showlegend=True))
        # annotate team before (at axis_start_pos)
        try:
            v_team_before = team_elos_series.iloc[axis_start_pos]
            fig.add_annotation(x=idx[axis_start_pos], y=float(v_team_before), text=f"{float(v_team_before):.0f}", showarrow=False, yshift=20, font=dict(size=13))
        except Exception:
            pass
        # annotate team at team's last game (if available)
        if team_last_game is not None:
            try:
                team_last_pos = idx.searchsorted(team_last_game, side='right') - 1
                v_team_after = team_elos_series.iloc[team_last_pos]
                fig.add_annotation(x=idx[team_last_pos], y=float(v_team_after), text=f"{float(v_team_after):.0f}", showarrow=False, yshift=20, font=dict(size=13))
            except Exception:
                pass
        # annotate team at axis end
        try:
            v_team_axis_end = team_elos_series.iloc[axis_end_pos]
            fig.add_annotation(x=idx[axis_end_pos], y=float(v_team_axis_end), text=f"{float(v_team_axis_end):.0f}", showarrow=False, yshift=20, font=dict(size=13))
        except Exception:
            pass
    except Exception:
        pass

    # shaded region for team's active window
    if team_first_game is not None and team_last_game is not None:
        try:
            fig.add_vrect(x0=team_first_game, x1=team_last_game, fillcolor='lightblue', opacity=0.25, layer='below', line_width=0)
        except Exception:
            pass

    # horizontal starting ELO (no text label)
    fig.add_hline(y=INITIAL_ELO, line=dict(color='red', dash='dash'))

    fig.update_layout(
        title=f"{team} ELO History",
        xaxis_title='Date',
        yaxis_title='Team ELO Rating',
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=60, r=100, t=100, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=820,
        width=1200,
        font=dict(size=14),
        title_font=dict(size=20)
    )
    fig.update_xaxes(tickformat='%m/%d/%Y %H:%M', tickfont=dict(size=12))
    return fig


# Build Dash app layout
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load data once at startup
try:
    players, dates, elo_df = load_elo_results(ELO_RESULTS)
except Exception as e:
    players, dates, elo_df = [], [], pd.DataFrame()

team_to_players = load_teams(TEAMS_FILE)
games, games_with_scores = load_games(GAMES_FILE)

df_teams, team_elo_history, team_first_last, team_record = compute_team_stats(team_to_players, players, dates, elo_df, games, games_with_scores)

# --- Player stats and helpers
def compute_player_stats(team_to_players, players, dates, elo_df, games, games_with_scores):
    """Return (df_players, player_last_game_map, player_team_membership)
    df_players columns: player, current_elo, peak_elo, min_elo, win_pct, games, last_played
    """
    # build player list from teams or elo_df
    all_players = set(players) if players else set()
    for t, members in team_to_players.items():
        for p in members:
            all_players.add(p)
    all_players = sorted(all_players)

    # current, peak, min from elo_df if available
    player_current = {}
    player_peak = {}
    player_min = {}
    if not elo_df.empty:
        for p in all_players:
            if p in elo_df.columns:
                s = elo_df[p]
                try:
                    player_current[p] = float(s.iloc[-1])
                    player_peak[p] = float(s.max())
                    player_min[p] = float(s.min())
                except Exception:
                    player_current[p] = None
                    player_peak[p] = None
                    player_min[p] = None
            else:
                player_current[p] = None
                player_peak[p] = None
                player_min[p] = None

    # init stats
    player_stats = {p: {'wins':0,'losses':0,'draws':0,'games':0} for p in all_players}
    # map player -> last played datetime
    player_last = {p: None for p in all_players}

    # iterate games_with_scores to compute player-level records
    for dt, t1, s1, t2, s2 in games_with_scores:
        # get members
        m1 = team_to_players.get(t1, [])
        m2 = team_to_players.get(t2, [])
        if not m1 and not m2:
            continue
        # update games for members
        for p in m1:
            player_stats.setdefault(p, {'wins':0,'losses':0,'draws':0,'games':0})
            player_stats[p]['games'] += 1
            if s1 > s2:
                player_stats[p]['wins'] += 1
            elif s1 < s2:
                player_stats[p]['losses'] += 1
            else:
                player_stats[p]['draws'] += 1
            # last played
            if player_last.get(p) is None or dt > player_last.get(p):
                player_last[p] = dt
        for p in m2:
            player_stats.setdefault(p, {'wins':0,'losses':0,'draws':0,'games':0})
            player_stats[p]['games'] += 1
            if s2 > s1:
                player_stats[p]['wins'] += 1
            elif s2 < s1:
                player_stats[p]['losses'] += 1
            else:
                player_stats[p]['draws'] += 1
            if player_last.get(p) is None or dt > player_last.get(p):
                player_last[p] = dt

    # assemble rows
    rows = []
    for p in all_players:
        st = player_stats.get(p, {'wins':0,'losses':0,'draws':0,'games':0})
        games_played = st.get('games',0)
        wins = st.get('wins',0)
        draws = st.get('draws',0)
        win_pct = (wins + 0.5*draws) / games_played if games_played>0 else None
        last_dt = player_last.get(p)
        last_played_str = None
        if last_dt is not None:
            try:
                last_played_str = pd.to_datetime(last_dt).strftime('%m/%d/%Y %H:%M:%S')
            except Exception:
                last_played_str = str(last_dt)

        rows.append({
            'player': p,
            'current_elo': None if player_current.get(p) is None else round(player_current.get(p),2),
            'peak_elo': None if player_peak.get(p) is None else round(player_peak.get(p),2),
            'min_elo': None if player_min.get(p) is None else round(player_min.get(p),2),
            'win_pct': None if win_pct is None else round(win_pct,3),
            'wins': wins,
            'losses': st.get('losses', 0),
            'draws': draws,
            'games': games_played,
            'last_played': last_played_str,
            'last_played_dt': last_dt
        })

    df = pd.DataFrame(rows)
    # sort by last_played_dt desc
    if 'last_played_dt' in df.columns:
        try:
            df = df.sort_values(by='last_played_dt', ascending=False, na_position='last').reset_index(drop=True)
        except Exception:
            pass
    return df, player_last


def _apply_sort(rows, sort_by_list):
    """Utility: sort a list of dict rows according to DataTable sort_by list.
    Special-case: when sorting the visible percent display column ('win_pct_display'),
    use the numeric 'win_pct' value when available so sorting is numeric, not lexicographic.
    """
    if not sort_by_list:
        return rows
    for s in reversed(sort_by_list):
        col = s.get('column_id')
        direction = s.get('direction', 'asc')

        # Special-case win_pct_display: ensure numeric comparison and keep missing values at the end
        if col == 'win_pct_display':
            def keyfn_wp(r):
                wp = r.get('win_pct')
                # treat NaN/None/empty as missing
                if wp is None or (isinstance(wp, float) and pd.isna(wp)):
                    return (1, 0.0)
                try:
                    v = float(wp)
                except Exception:
                    return (1, 0.0)
                # when descending, invert numeric so we can sort ascending while respecting direction
                if direction == 'desc':
                    return (0, -v)
                return (0, v)

            try:
                rows.sort(key=keyfn_wp)
            except Exception:
                pass
            continue

        # Generic keyfn: return raw value; we'll sort ascending on a stable comparable tuple
        def keyfn(r):
            return r.get(col)

        def key_wrapper(r):
            val = keyfn(r)
            missing = val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, str) and val == '')
            # For numeric values keep as float, else use lowercase string for deterministic ordering
            if missing:
                return (1, '')
            if isinstance(val, (int, float, np.number)):
                try:
                    num = float(val)
                    return (0, num if direction == 'asc' else -num)
                except Exception:
                    pass
            # fallback string
            return (0, str(val).lower() if direction == 'asc' else ''.join(chr(255 - ord(c)) for c in str(val)))

        try:
            rows.sort(key=key_wrapper)
        except Exception:
            pass
    return rows

def build_player_figure(player, dates, elo_df, team_to_players, games, team_first_last=None):
    """Create Plotly figure that matches elo_explore.plot_player_elo for the player.

    Behavior:
    - Determine the player's first and last game (based on teams the player is/was on).
    - Choose the snapshot immediately before the first game as the start sample (or the first available snapshot if none before).
    - Choose the snapshot at or immediately before the last game as the end sample.
    - Plot every ELO datapoint (including duplicates) between these two positional indices so x and y stay aligned.
    - For each team the player has been on, plot the team-average ELO over the same window (same x-axis indices).
    """
    fig = go.Figure()

    if elo_df is None or elo_df.empty:
        return fig

    idx = elo_df.index
    n = len(idx)
    # default window: full range
    start_pos = 0
    end_pos = n - 1

    # find player's teams
    player_teams = [t for t, members in team_to_players.items() if player in members]
    if player_teams and games:
        # collect games involving any of the player's teams
        player_team_games = [g for g in games if (g[1] in player_teams) or (g[2] in player_teams)]
        if player_team_games:
            first_game = min(g[0] for g in player_team_games)
            last_game = max(g[0] for g in player_team_games)
            # start_pos: snapshot immediately before first_game (i.e., last index < first_game), else 0
            p = idx.searchsorted(first_game, side='left') - 1
            if p >= 0:
                start_pos = p
            else:
                start_pos = 0
            # end_pos: last index <= last_game
            q = idx.searchsorted(last_game, side='right') - 1
            if q >= 0:
                end_pos = q
            else:
                end_pos = n - 1

    # validate window
    if start_pos < 0:
        start_pos = 0
    if end_pos < start_pos:
        end_pos = start_pos

    plot_idx = idx[start_pos:end_pos+1]

    # Player trace
    if player in elo_df.columns:
        player_series = elo_df[player].iloc[start_pos:end_pos+1]
        player_y = player_series.values
    else:
        player_y = np.array([INITIAL_ELO] * len(plot_idx))

    fig.add_trace(go.Scatter(x=plot_idx, y=player_y, mode='lines', line=dict(color='blue', width=2), name=player, hovertemplate='%{x|%m/%d/%Y %H:%M}: %{y:.0f}<extra>'+player+'</extra>', showlegend=True))
    # endpoint markers and label
    if len(plot_idx) > 0:
        try:
            # endpoint markers (bigger for readability)
            fig.add_trace(go.Scatter(x=[plot_idx[0]], y=[player_y[0]], mode='markers', marker=dict(color='blue', size=10, line=dict(color='white', width=0.8)), showlegend=False))
            fig.add_trace(go.Scatter(x=[plot_idx[-1]], y=[player_y[-1]], mode='markers', marker=dict(color='blue', size=10, line=dict(color='white', width=0.8)), showlegend=False))
            # annotations above start and end with ELO (larger font)
            fig.add_annotation(x=plot_idx[0], y=player_y[0], text=f"{player_y[0]:.0f}", showarrow=False, yshift=14, font=dict(color='blue', size=12), bgcolor='rgba(255,255,255,0)')
            fig.add_annotation(x=plot_idx[-1], y=player_y[-1], text=f"{player_y[-1]:.0f}", showarrow=False, yshift=14, font=dict(color='blue', size=12), bgcolor='rgba(255,255,255,0)')
        except Exception:
            pass

    # horizontal starting ELO (no text label)
    fig.add_hline(y=INITIAL_ELO, line=dict(color='red', dash='dash'))

    # For each team the player has been on, plot the team-average on that team's own window
    cmap = px.colors.qualitative.Plotly
    for ti, team in enumerate(player_teams):
        members = [p for p in team_to_players.get(team, []) if p in elo_df.columns]
        if not members:
            continue

        # Determine this team's first/last game timestamps
        first_game = None
        last_game = None
        if team_first_last and team in team_first_last:
            try:
                first_game, last_game = team_first_last.get(team)
            except Exception:
                first_game = None
                last_game = None
        if first_game is None or last_game is None:
            # fallback: derive from `games` list
            team_games = [g for g in games if g[1] == team or g[2] == team]
            if team_games:
                first_game = min(g[0] for g in team_games)
                last_game = max(g[0] for g in team_games)
        if first_game is None or last_game is None:
            # no game history for this team; skip plotting
            continue

        # compute positional indices for this team's window: snapshot immediately before first_game
        try:
            t_start_pos = idx.searchsorted(first_game, side='left') - 1
            if t_start_pos < 0:
                t_start_pos = 0
        except Exception:
            t_start_pos = 0
        try:
            t_end_pos = idx.searchsorted(last_game, side='right') - 1
            if t_end_pos < 0:
                t_end_pos = n - 1
        except Exception:
            t_end_pos = n - 1

        if t_end_pos < t_start_pos:
            continue

        seg_idx = idx[t_start_pos:t_end_pos+1]
        team_series_all = elo_df[members].mean(axis=1)
        team_series = team_series_all.iloc[t_start_pos:t_end_pos+1]

        color = cmap[ti % len(cmap)]
        try:
            # show team in legend so user can click to hide/show (thicker line and larger markers)
            fig.add_trace(go.Scatter(x=seg_idx, y=team_series.values, mode='lines', line=dict(color=color, dash='dash', width=2.5), name=team, showlegend=True, hovertemplate='%{x|%m/%d/%Y %H:%M}: %{y:.0f}<extra>'+team+'</extra>'))
            # markers at the segment ends
            fig.add_trace(go.Scatter(x=[seg_idx[0]], y=[team_series.values[0]], mode='markers', marker=dict(color=color, size=8), showlegend=False))
            fig.add_trace(go.Scatter(x=[seg_idx[-1]], y=[team_series.values[-1]], mode='markers', marker=dict(color=color, size=8), showlegend=False))
            # annotations above start and end with ELO (larger font)
            fig.add_annotation(x=seg_idx[0], y=team_series.values[0], text=f"{team_series.values[0]:.0f}", showarrow=False, yshift=14, font=dict(color=color, size=12), bgcolor='rgba(255,255,255,0)')
            fig.add_annotation(x=seg_idx[-1], y=team_series.values[-1], text=f"{team_series.values[-1]:.0f}", showarrow=False, yshift=14, font=dict(color=color, size=12), bgcolor='rgba(255,255,255,0)')
        except Exception:
            try:
                fig.add_trace(go.Scatter(x=seg_idx, y=team_series.values, mode='lines', line=dict(color=color, dash='dash', width=2), showlegend=True, name=team))
            except Exception:
                pass

    fig.update_layout(
        title=f"{player} ELO History",
        xaxis_title='Date',
        yaxis_title='ELO Rating',
        hovermode='x unified',
        margin=dict(l=60, r=100, t=100, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=820,
        width=1200,
        font=dict(size=14),
        title_font=dict(size=20)
    )
    # allow clicking legend items to hide/show traces (default behavior) and place legend above plot
    fig.update_xaxes(tickformat='%m/%d/%Y %H:%M', tickfont=dict(size=12))
    return fig

def render_players_tab():
    # compute df_players at runtime
    # data will be populated server-side via callback to support numeric sorting of Win %
    df_players, player_last_map = compute_player_stats(team_to_players, players, dates, elo_df, games, games_with_scores)
    # remove hidden datetime column before passing to the DataTable (DataTable only accepts primitives)
    # initial data left empty; callback `players_table_data` will provide records
    table = dash_table.DataTable(
        id='players-table',
        columns=[
            {'name': 'Player', 'id': 'player', 'presentation': 'markdown'},
            {'name': 'ELO', 'id': 'current_elo', 'type': 'numeric'},
            {'name': 'Peak ELO', 'id': 'peak_elo', 'type': 'numeric'},
            {'name': 'Min ELO', 'id': 'min_elo', 'type': 'numeric'},
                {'name': 'Win %', 'id': 'win_pct_display', 'type': 'text'},
                {'name': 'Win % (sort)', 'id': 'win_pct', 'type': 'numeric'},
            {'name': 'Games', 'id': 'games', 'type': 'numeric'},
            {'name': 'Last Played', 'id': 'last_played', 'type': 'text'},
        ],
        data=[],
        sort_action='custom',
        # default sort by current ELO (descending)
        sort_by=[{'column_id': 'current_elo', 'direction': 'desc'}],
        filter_action='native',
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_header={'fontWeight': 'bold'},
    page_size=30,
    hidden_columns=['win_pct'],
    style_data_conditional=[
            {
                'if': {'column_id': 'player'},
                'color': 'blue',
                'textDecoration': 'underline',
                'cursor': 'pointer'
            }
        ],
    )
    # Keep player names as plain text in the DataTable and handle clicks via callback
    # Now render player names as markdown links so users can click to navigate
    # table.data already set above
    return html.Div([html.P('Click a player name to view details.'), table])


@app.callback(Output('players-table', 'data'), [Input('players-table', 'sort_by')])
def players_table_data(sort_by):
    # Recompute players records and apply server-side sorting
    df_players, player_last_map = compute_player_stats(team_to_players, players, dates, elo_df, games, games_with_scores)
    records = df_players.drop(columns=['last_played_dt'], errors='ignore').to_dict('records')
    for rec in records:
        try:
            wp = rec.get('win_pct')
            # include win% plus W:L:T breakdown
            wins = rec.get('wins', 0)
            losses = rec.get('losses', 0)
            draws = rec.get('draws', 0)
            if wp is not None and not pd.isna(wp):
                rec['win_pct_display'] = f"{wp*100:.1f}% (W:{wins} L:{losses} T:{draws})"
            else:
                rec['win_pct_display'] = ''
            pname = rec.get('player')
            if pname:
                rec['player'] = f"[{pname}](/player/{urllib.parse.quote(pname)})"
        except Exception:
            pass
    # apply sorting
    records = _apply_sort(records, sort_by)
    return records

def render_player_page(player):
    # match functionality of elo_explore.py 'player' command
    # validate
    if player not in set(players) and player not in [p for plist in team_to_players.values() for p in plist]:
        return html.Div([html.H3(f'Player not found: {player}'), html.A('Back', href='/', target='_self')])
    # current elo and rank
    try:
        current = float(elo_df[player].iloc[-1]) if player in elo_df.columns else None
    except Exception:
        current = None
    # compute rank and percentile
    rank = None
    percentile = None
    if current is not None:
        all_elos = {p: float(elo_df[p].iloc[-1]) for p in elo_df.columns}
        sorted_players = sorted(all_elos.items(), key=lambda x: x[1], reverse=True)
        rank = next((i+1 for i,(p,_) in enumerate(sorted_players) if p == player), None)
        total_players = len(sorted_players)
        if rank is not None and total_players>0:
            percentile = 100 * (total_players - rank + 1) / total_players

    # compute player record
    pstat = {'wins':0,'losses':0,'draws':0,'games':0}
    player_teams = [t for t,members in team_to_players.items() if player in members]
    for dt, t1, s1, t2, s2 in games_with_scores:
        if t1 in player_teams:
            pstat['games'] += 1
            if s1 > s2:
                pstat['wins'] += 1
            elif s1 < s2:
                pstat['losses'] += 1
            else:
                pstat['draws'] += 1
        if t2 in player_teams:
            pstat['games'] += 0  # already counted when team in player_teams above

    # build info block
    info = []
    info.append(html.P(f"Player: {player}"))
    if current is not None:
        info.append(html.P(f"ELO: {current:.2f}  Rank: #{rank} ({percentile:.1f}th percentile)"))
    else:
        info.append(html.P("ELO: N/A"))
    if pstat['games']>0:
        pct = 100.0 * (pstat['wins'] + 0.5*pstat['draws'])/pstat['games']
        info.append(html.P(f"Record (W-L-D): {pstat['wins']}:{pstat['losses']}:{pstat['draws']}  {pct:.1f}%"))
    else:
        info.append(html.P("Record (W-L-D): 0:0:0 N/A"))

    # list teams player has been on and per-team stats
    # For each team show: link to team page, last played datetime, current team ELO, and team win %
    # Sort teams by last played (most recent first). Teams with no last-played go to the end.
    teams_block = [html.H4('Teams')]
    # gather team info
    team_infos = []
    for t in sorted(set(player_teams)):
        # try to find row in df_teams for precomputed stats
        try:
            row = df_teams.loc[df_teams['team'] == t]
        except Exception:
            row = pd.DataFrame()
        if not row.empty:
            r = row.iloc[0]
            last_dt = r.get('last_played_dt')
            last_str = r.get('last_played')
            elo_val = r.get('current_elo')
            win_pct = r.get('win_pct')
            wins = int(r.get('wins')) if r.get('wins') is not None else 0
            losses = int(r.get('losses')) if r.get('losses') is not None else 0
            draws = int(r.get('draws')) if r.get('draws') is not None else 0
        else:
            # fallback: compute from available structures
            last_dt = None
            last_str = None
            series = team_elo_history.get(t)
            if series is not None and not series.empty:
                try:
                    elo_val = round(float(series.iloc[-1]), 2)
                except Exception:
                    elo_val = None
            else:
                elo_val = None
            rec = team_record.get(t, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
            games_played = rec.get('games', 0)
            wins = rec.get('wins', 0)
            losses = rec.get('losses', 0)
            draws = rec.get('draws', 0)
            win_pct = (wins + 0.5 * draws) / games_played if games_played > 0 else None
        team_infos.append({'team': t, 'last_dt': last_dt, 'last_str': last_str, 'elo': elo_val, 'win_pct': win_pct, 'wins': wins, 'losses': losses, 'draws': draws})

    # Normalize last_dt to datetime objects and sort by last_dt (most recent first); None values go to the end
    for ti in team_infos:
        ld = ti.get('last_dt')
        if ld is None:
            continue
        # if it's already a datetime-like, leave it; otherwise try to parse
        if not isinstance(ld, datetime):
            try:
                ti['last_dt'] = pd.to_datetime(ld)
            except Exception:
                ti['last_dt'] = None

    def _sort_key(x):
        # put teams with no last_dt at the end
        if x.get('last_dt') is None:
            return datetime.min
        return x.get('last_dt')

    team_infos.sort(key=_sort_key, reverse=True)

    # render as a small table with link, last played, ELO, Win %
    table_header = html.Tr([html.Th('Team'), html.Th('Last Played'), html.Th('ELO'), html.Th('Win %')])
    table_rows = []
    for tinfo in team_infos:
        tname = tinfo['team']
        url = '/team/' + urllib.parse.quote(tname)
        last_display = tinfo['last_str'] if tinfo['last_str'] else 'Never'
        elo_display = f"{tinfo['elo']:.2f}" if (tinfo['elo'] is not None and not pd.isna(tinfo['elo'])) else 'N/A'
        try:
            if tinfo.get('win_pct') is not None and not pd.isna(tinfo.get('win_pct')):
                w = int(tinfo.get('wins', 0))
                l = int(tinfo.get('losses', 0))
                d = int(tinfo.get('draws', 0))
                win_display = f"{tinfo['win_pct']:.1%} (W:{w} L:{l} T:{d})"
            else:
                win_display = 'N/A'
        except Exception:
            win_display = 'N/A'
        row = html.Tr([
            html.Td(html.A(tname, href=url, target='_self')),
            html.Td(last_display),
            html.Td(elo_display),
            html.Td(win_display),
        ])
        table_rows.append(row)
    teams_block.append(html.Table([table_header] + table_rows, style={'borderCollapse': 'collapse', 'width': '100%'}))

    # player plot
    fig = None
    try:
        fig = build_player_figure(player, dates, elo_df, team_to_players, games, team_first_last)
    except Exception:
        fig = None

    content = [html.Div([html.A('Back to Players', href='/', target='_self')], style={'marginBottom':'10px'}), html.Div(info, style={'width':'35%','display':'inline-block','verticalAlign':'top'}), html.Div(teams_block, style={'width':'35%','display':'inline-block','verticalAlign':'top','paddingLeft':'10px'})]
    if fig is not None:
        content.append(html.Div([dcc.Graph(figure=fig, id='player-graph')], style={'width':'63%','display':'inline-block','paddingLeft':'20px'}))
    return html.Div(content)

# Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='nav-store'),
    html.H2('BARSElo Explorer'),
    # Search bar: find a player or team by name
    html.Div([
        dcc.Input(id='search-input', type='text', placeholder='Find player or team (press Enter or click Go)', style={'width':'60%'}),
        html.Button('Go', id='search-button', n_clicks=0, style={'marginLeft':'8px'}),
        html.Div(id='search-results', style={'marginTop':'6px'})
    ], style={'marginBottom': '12px'}),
    dcc.Tabs(id='tabs', value='players', children=[
        dcc.Tab(label='Players', value='players'),
        dcc.Tab(label='Teams', value='teams'),
        dcc.Tab(label='Changes', value='changes'),
    ]),
    # Render both containers initially (teams hidden) so callbacks that refer
    # to `players-table` and `teams-table` find their component ids on first
    # load — this avoids client-side errors when callbacks reference ids that
    # only appear later. The visible content is still controlled by the
    # `render_page` callback which replaces these children when navigation
    # occurs.
    html.Div(id='page-content', children=[
        html.Div(render_players_tab(), id='players-page'),
        # lightweight hidden placeholder for teams table so the id exists
        html.Div(
            dash_table.DataTable(
                id='teams-table',
                columns=[
                    {'name': 'Team', 'id': 'team'},
                    {'name': 'Last Played', 'id': 'last_played', 'type': 'text'},
                    {'name': 'ELO', 'id': 'current_elo', 'type': 'numeric'},
                    {'name': 'Beginning ELO', 'id': 'beginning_elo', 'type': 'numeric'},
                    {'name': 'Ending ELO', 'id': 'last_game_elo', 'type': 'numeric'},
                    {'name': 'Win %', 'id': 'win_pct', 'type': 'numeric'},
                    {'name': 'Games Played', 'id': 'games', 'type': 'numeric'},
                ],
                data=[],
                style_cell={'display': 'none'}
            ),
            id='teams-page',
            style={'display': 'none'}
        )
    ])
], style={'margin': '20px'})


# Note: we intentionally do NOT update the URL from the tab click to avoid
# creating a callback cycle between tabs.value and url.pathname. The app still
# respects the URL on load / navigation (URL -> tabs), which keeps tab state in
# sync with direct links and browser Back/Forward.

# Clientside callback: update the visual tab selection from the URL pathname
# This runs in the browser and avoids creating a server-side circular callback.
app.clientside_callback(
    """
    function(pathname) {
        if (!pathname) { return window.dash_clientside.no_update; }
        try {
            if (pathname === '/teams' || pathname.indexOf('/team/') === 0) { return 'teams'; }
                if (pathname === '/changes' || pathname.indexOf('/changes') === 0) { return 'changes'; }
            if (pathname === '/players' || pathname.indexOf('/player/') === 0) { return 'players'; }
        } catch (e) {
            return window.dash_clientside.no_update;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('tabs', 'value'),
    [Input('url', 'pathname')]
)


# Page content callback
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname'), Input('tabs', 'value')])
def render_page(pathname, tab_value):
    # Use callback context to see what triggered this render. If the URL changed
    # (user clicked a link), prefer the pathname so detail pages open. If the
    # tabs control changed (user clicked a tab), prefer the tab selection so the
    # UI responds immediately even if the pathname still contains a detail path.
    try:
        ctx = dash.callback_context
        triggered = (ctx.triggered[0]['prop_id'] if ctx.triggered else None)
    except Exception:
        triggered = None

    # Decide which main content to show (detail page or a tab content)
    main_content = None
    # URL navigation requested: honor pathname first
    if triggered and 'url.pathname' in triggered:
        if pathname == '/teams':
            main_content = render_teams_tab()
        elif pathname == '/changes':
            # render changes tab (will show date picker + tables)
            main_content = render_changes_tab()
        elif pathname == '/players':
            main_content = render_players_tab()
        elif pathname and pathname.startswith('/team/'):
            team = urllib.parse.unquote(pathname[len('/team/'):])
            main_content = render_team_page(team)
        elif pathname and pathname.startswith('/player/'):
            player = urllib.parse.unquote(pathname[len('/player/'):])
            main_content = render_player_page(player)

    # Tab click requested: honor the tab selection
    if main_content is None and triggered and 'tabs.value' in triggered:
        if tab_value == 'teams':
            main_content = render_teams_tab()
        elif tab_value == 'players':
            main_content = render_players_tab()
        elif tab_value == 'changes':
            main_content = render_changes_tab()

    # Fallback: URL detail > selected tab > players
    if main_content is None:
        if pathname and pathname.startswith('/team/'):
            team = urllib.parse.unquote(pathname[len('/team/'):])
            main_content = render_team_page(team)
        elif pathname and pathname.startswith('/player/'):
            player = urllib.parse.unquote(pathname[len('/player/'):])
            main_content = render_player_page(player)
        elif tab_value == 'teams':
            main_content = render_teams_tab()
        else:
            main_content = render_players_tab()

    # Ensure that component ids referenced by callbacks exist in the returned
    # layout. If the main content doesn't include one of the tables, append a
    # hidden placeholder DataTable so `players-table` and `teams-table` ids are
    # always present in the DOM and callbacks won't error.
    placeholders = []
    # If players-table not present in main_content, add hidden players table
    try:
        # rough check: see if 'players-table' appears in the main_content repr
        if 'players-table' not in repr(main_content):
            placeholders.append(html.Div(dash_table.DataTable(id='players-table', columns=[{'name': 'Player', 'id': 'player'}], data=[], style_cell={'display': 'none'}), style={'display': 'none'}))
    except Exception:
        pass
    try:
        if 'teams-table' not in repr(main_content):
            placeholders.append(html.Div(dash_table.DataTable(id='teams-table', columns=[{'name': 'Team', 'id': 'team'}], data=[], style_cell={'display': 'none'}), style={'display': 'none'}))
    except Exception:
        pass
    # ensure changes tables exist for callbacks even when not visible
    try:
        if 'changes-players-table' not in repr(main_content):
            placeholders.append(html.Div(dash_table.DataTable(id='changes-players-table', columns=[{'name': 'Player', 'id': 'player'}], data=[], style_cell={'display': 'none'}), style={'display': 'none'}))
    except Exception:
        pass
    try:
        if 'changes-teams-table' not in repr(main_content):
            placeholders.append(html.Div(dash_table.DataTable(id='changes-teams-table', columns=[{'name': 'Team', 'id': 'team'}], data=[], style_cell={'display': 'none'}), style={'display': 'none'}))
    except Exception:
        pass
    try:
        if 'changes-date-picker' not in repr(main_content):
            # provide a hidden DatePickerSingle so callbacks depending on its `date` prop
            # do not error when the Changes tab is not the currently rendered content
            placeholders.append(html.Div(dcc.DatePickerSingle(id='changes-date-picker', date=None), style={'display': 'none'}))
    except Exception:
        pass

    if placeholders:
        return html.Div([main_content] + placeholders)
    return main_content


@app.callback(
    [Output('url', 'pathname'), Output('search-results', 'children')],
    [Input('search-button', 'n_clicks'), Input('search-input', 'n_submit'), Input('nav-store', 'data')],
    [State('search-input', 'value')]
)
def navigate(n_clicks, n_submit, nav_data, query):
    """Navigation callback: responds to search and nav-store events.
    nav-store is written by table click callbacks when tables are present.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    prop = ctx.triggered[0]['prop_id']

    # nav-store triggered by table click
    if prop == 'nav-store.data' and nav_data:
        path = nav_data.get('path')
        if path:
            # clear search-results when navigating from table
            return path, html.Div()
        return dash.no_update, dash.no_update

    # Search triggered
    if prop in ('search-button.n_clicks', 'search-input.n_submit'):
        if not query:
            return dash.no_update, html.Div()
        q = str(query).strip()
        if not q:
            return dash.no_update, html.Div()
        ql = q.lower()
        # exact matches
        teams_exact = [t for t in team_to_players.keys() if t and t.lower() == ql]
        players_exact = [p for p in players if p and p.lower() == ql]
        if teams_exact:
            return '/team/' + urllib.parse.quote(teams_exact[0]), html.Div()
        if players_exact:
            return '/player/' + urllib.parse.quote(players_exact[0]), html.Div()

        # substring matches
        teams_sub = [t for t in team_to_players.keys() if t and ql in t.lower()]
        players_sub = [p for p in players if p and ql in p.lower()]
        if len(teams_sub) == 1 and not players_sub:
            return '/team/' + urllib.parse.quote(teams_sub[0]), html.Div()
        if len(players_sub) == 1 and not teams_sub:
            return '/player/' + urllib.parse.quote(players_sub[0]), html.Div()

        if teams_sub or players_sub:
            items = []
            if teams_sub:
                items.append(html.H5('Teams'))
                for t in teams_sub:
                    items.append(html.Div(html.A(t, href='/team/' + urllib.parse.quote(t), target='_self')))
            if players_sub:
                items.append(html.H5('Players'))
                for p in players_sub:
                    items.append(html.Div(html.A(p, href='/player/' + urllib.parse.quote(p), target='_self')))
            return dash.no_update, html.Div(items)

        return dash.no_update, html.Div('No matches')


# Table click callbacks: when a user clicks a table row, write the desired
# navigation path into the `nav-store`. The main `navigate` callback above
# listens to this store and updates the URL. This avoids multiple callbacks
# writing to `url.pathname` and avoids referencing table inputs from the
# search-triggered path (which can cause missing-component runtime errors).

@app.callback(
    Output('nav-store', 'data'),
    [
        Input('changes-players-table', 'active_cell'),
        Input('changes-teams-table', 'active_cell')
    ],
    [State('changes-players-table', 'derived_virtual_data'), State('changes-teams-table', 'derived_virtual_data')],
    prevent_initial_call=True
)
def _on_table_nav(cp_active, ct_active, cp_data, ct_data):
    """Unified table navigation: respond to active_cell events from players
    and teams tables. Returns a dict {'path': '/player/...'} or
    {'path': '/team/...'} to write into nav-store, or dash.no_update.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trig = ctx.triggered[0]['prop_id']

    try:
        # Changes tables: respond to active_cell events
        if 'changes-players-table.active_cell' in trig and cp_active and cp_data:
            col = cp_active.get('column_id')
            row = cp_active.get('row')
            if col == 'player' and row is not None and 0 <= row < len(cp_data):
                rec = cp_data[row]
                player = rec.get('player')
                if player:
                    return {'path': '/player/' + urllib.parse.quote(player)}
        if 'changes-teams-table.active_cell' in trig and ct_active and ct_data:
            col = ct_active.get('column_id')
            row = ct_active.get('row')
            if col == 'team' and row is not None and 0 <= row < len(ct_data):
                rec = ct_data[row]
                team = rec.get('team')
                if team:
                    return {'path': '/team/' + urllib.parse.quote(team)}
    except Exception:
        pass
    return dash.no_update


def render_teams_tab():
    # build dash datatable from df_teams
    # remove list-valued or datetime fields before passing to the DataTable
    # initial data left empty; server-side callback will populate and support numeric sorting
    records = []
    table = dash_table.DataTable(
        id='teams-table',
        columns=[
            {'name': 'Team', 'id': 'team', 'presentation': 'markdown'},
            {'name': 'Last Played', 'id': 'last_played', 'type': 'text'},
            {'name': 'ELO', 'id': 'current_elo', 'type': 'numeric'},
            {'name': 'Beginning ELO', 'id': 'beginning_elo', 'type': 'numeric'},
            {'name': 'Ending ELO', 'id': 'last_game_elo', 'type': 'numeric'},
            {'name': 'Win %', 'id': 'win_pct_display', 'type': 'text'},
            {'name': 'Win % (sort)', 'id': 'win_pct', 'type': 'numeric'},
            {'name': 'Games Played', 'id': 'games', 'type': 'numeric'},
        ],
        data=[],
        sort_action='custom',
        filter_action='native',
        hidden_columns=['win_pct'],
        style_data_conditional=[
            {
                'if': {'column_id': 'team'},
                'color': 'blue',
                'textDecoration': 'underline',
                'cursor': 'pointer'
            }
        ],
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_header={'fontWeight': 'bold'},
        page_size=20,
    )
    # Keep team names as plain text in the DataTable and handle clicks via callback
    # Now render team names as markdown links so users can click to navigate

    return html.Div([
        html.P('Click a team name to view details.'),
        table,
    ])


@app.callback(Output('teams-table', 'data'), [Input('teams-table', 'sort_by')])
def teams_table_data(sort_by):
    # Build records from precomputed df_teams and apply server-side sorting
    records = df_teams.drop(columns=['members', 'last_played_dt'], errors='ignore').to_dict('records')
    for rec in records:
        try:
            wp = rec.get('win_pct')
            wins = rec.get('wins', 0)
            losses = rec.get('losses', 0)
            draws = rec.get('draws', 0)
            if wp is not None and not pd.isna(wp):
                rec['win_pct_display'] = f"{wp*100:.1f}% (W:{wins} L:{losses} T:{draws})"
            else:
                rec['win_pct_display'] = ''
            tname = rec.get('team')
            if tname:
                rec['team'] = f"[{tname}](/team/{urllib.parse.quote(tname)})"
        except Exception:
            pass
    records = _apply_sort(records, sort_by)
    return records


def render_team_page(team):
    if team not in team_to_players:
        return html.Div([html.H3(f'Team not found: {team}'), html.A('Back', href='/', target='_self')])
    # Prepare detail info
    members = team_to_players.get(team, [])
    # current elo
    current = df_teams.loc[df_teams['team'] == team, 'current_elo']
    current = float(current.iloc[0]) if not current.empty and pd.notna(current.iloc[0]) else None
    last = df_teams.loc[df_teams['team'] == team, 'last_game_elo']
    last = float(last.iloc[0]) if not last.empty and pd.notna(last.iloc[0]) else None
    beg = df_teams.loc[df_teams['team'] == team, 'beginning_elo']
    beg = float(beg.iloc[0]) if not beg.empty and pd.notna(beg.iloc[0]) else None
    rec = team_record.get(team, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})

    # ranks
    def rank_for(col):
        colname = col
        rankcol = col + '_rank'
        try:
            r = df_teams.loc[df_teams['team'] == team, rankcol].iloc[0]
            return int(r) if not pd.isna(r) else None
        except Exception:
            return None

    ranks = {
        'current_rank': rank_for('current_elo'),
        'last_rank': rank_for('last_game_elo'),
        'beginning_rank': rank_for('beginning_elo'),
        'win_pct_rank': rank_for('win_pct')
    }

    # team plot (interactive Plotly)
    team_series = team_elo_history.get(team)
    fig = None
    if team_series is not None and not elo_df.empty:
        fig = build_team_figure(team, dates, team_series, team_to_players, elo_df, team_first_last)

    # Build left-side info panel with ranks inline
    # Recompute player current elos and ranks from elo_df (latest snapshot)
    player_current_elos = {}
    try:
        last_row = elo_df.iloc[-1]
        for col in elo_df.columns:
            try:
                player_current_elos[col] = float(last_row[col])
            except Exception:
                pass
    except Exception:
        player_current_elos = {}

    # Compute global player ranks (1 = highest)
    player_rank_map = {}
    if player_current_elos:
        sorted_players = sorted(player_current_elos.items(), key=lambda x: x[1], reverse=True)
        for i, (p, _) in enumerate(sorted_players, start=1):
            player_rank_map[p] = i

    # Determine team's last-game snapshot position (use team_first_last if available)
    team_last_game = None
    try:
        if team_first_last and team in team_first_last:
            team_last_game = team_first_last[team][1]
    except Exception:
        team_last_game = None

    # default positions
    idx = elo_df.index if (elo_df is not None and not elo_df.empty) else []
    if team_last_game is not None and len(idx) > 0:
        try:
            team_last_pos = idx.searchsorted(team_last_game, side='right') - 1
            if team_last_pos < 0:
                team_last_pos = 0
        except Exception:
            team_last_pos = len(idx) - 1
    else:
        team_last_pos = len(idx) - 1 if len(idx) > 0 else None

    # Team ending ELO (at last game) and ranking among teams using df_teams.last_game_elo
    try:
        team_row = df_teams.loc[df_teams['team'] == team].iloc[0]
    except Exception:
        team_row = None

    # number of teams for ranking
    total_teams = len(df_teams) if df_teams is not None else 0
    # ending elo from df_teams (precomputed) falls back to current
    ending_elo = None
    ending_rank = None
    if team_row is not None:
        try:
            ending_elo = team_row.get('last_game_elo') if not pd.isna(team_row.get('last_game_elo')) else None
        except Exception:
            ending_elo = None
        try:
            ending_rank = int(team_row.get('last_game_elo_rank')) if team_row.get('last_game_elo_rank') is not None and not pd.isna(team_row.get('last_game_elo_rank')) else None
        except Exception:
            ending_rank = None

    # current elo/rank (as before)
    cur_elo = current
    cur_rank = ranks.get('current_rank')

    # compute percentiles
    def pct_from_rank(rank, total):
        try:
            return 100.0 * (total - rank + 1) / total
        except Exception:
            return None

    ending_pct = pct_from_rank(ending_rank, total_teams) if ending_rank is not None else None
    cur_pct = pct_from_rank(cur_rank, total_teams) if cur_rank is not None else None

    # Team header: show ending ELO/rank/percentile then current in parens
    if ending_elo is not None:
        header_text = f"Team: {team}  —  Ending ELO: {ending_elo:.2f}"
        if ending_rank is not None:
            header_text += f"  Rank: #{ending_rank}/{total_teams} ({ending_pct:.1f}th)"
    else:
        header_text = f"Team: {team}  —  Ending ELO: N/A"
    if cur_elo is not None:
        header_text += f"  (current: ELO {cur_elo:.2f}"
        if cur_rank is not None:
            header_text += f"  Rank: #{cur_rank}/{total_teams} ({cur_pct:.1f}th)"
        header_text += ")"

    header_lines = [html.P(header_text)]

    # Player list: compute each member's ELO at the team's last game snapshot and sort by that value
    player_lines = []
    if members:
        # determine snapshot row at team's last game
        if team_last_pos is not None and elo_df is not None and not elo_df.empty:
            try:
                row_at_last = elo_df.iloc[team_last_pos]
            except Exception:
                row_at_last = None
        else:
            row_at_last = None

        # build global ranking at that snapshot for percentiles/ranks
        player_rank_at_last = {}
        total_players_at_last = 0
        if row_at_last is not None:
            vals = {p: row_at_last.get(p) for p in elo_df.columns}
            # filter numeric
            valid = {p: float(v) for p, v in vals.items() if v is not None and not pd.isna(v)}
            total_players_at_last = len(valid)
            sorted_all = sorted(valid.items(), key=lambda x: x[1], reverse=True)
            for i, (p, _) in enumerate(sorted_all, start=1):
                player_rank_at_last[p] = i

        # prepare member list with ending and current elos
        members_info = []
        for p in members:
            try:
                ending_elo = float(row_at_last.get(p)) if (row_at_last is not None and p in row_at_last.index and not pd.isna(row_at_last.get(p))) else None
            except Exception:
                ending_elo = None
            current_elo = player_current_elos.get(p)
            current_rank = player_rank_map.get(p)
            ending_rank = player_rank_at_last.get(p)
            members_info.append({'player': p, 'ending_elo': ending_elo, 'ending_rank': ending_rank, 'current_elo': current_elo, 'current_rank': current_rank})

        # sort members by ending_elo (None -> lowest)
        def _mi_key(m):
            v = m.get('ending_elo')
            return v if v is not None else -1e9

        members_info.sort(key=_mi_key, reverse=True)

        # format a compact table for members: Player | Ending ELO | Ending Rank | Ending % | Current ELO | Current Rank | Current %
        table_header = html.Tr([
            html.Th('Player'), html.Th('Ending ELO'), html.Th('Ending Rank'), html.Th('Ending %'),
            html.Th('Current ELO'), html.Th('Current Rank'), html.Th('Current %')
        ])
        table_rows = []
        for info in members_info:
            p = info['player']
            e = info['ending_elo']
            er = info.get('ending_rank')
            ce = info.get('current_elo')
            cr = info.get('current_rank')
            # compute pct strings
            if e is not None and er is not None and total_players_at_last > 0:
                ending_pct = 100.0 * (total_players_at_last - er + 1) / total_players_at_last
                ending_pct_s = f"{ending_pct:.1f}%"
                ending_rank_s = f"#{er}/{total_players_at_last}"
                ending_elo_s = f"{e:.2f}"
            else:
                ending_pct_s = 'N/A'
                ending_rank_s = 'N/A'
                ending_elo_s = 'N/A'

            if ce is not None and cr is not None and len(player_rank_map) > 0:
                total_now = len(player_rank_map)
                cur_pct = 100.0 * (total_now - cr + 1) / total_now
                cur_pct_s = f"{cur_pct:.1f}%"
                cur_rank_s = f"#{cr}/{total_now}"
                cur_elo_s = f"{ce:.2f}"
            else:
                cur_pct_s = 'N/A'
                cur_rank_s = 'N/A'
                cur_elo_s = 'N/A'

            # prepare a markdown link for the DataTable (always add row)
            player_md = f"[{p}](/player/{urllib.parse.quote(p)})"
            table_rows.append({
                'player': player_md,
                'player_name': p,
                'ending_elo': ending_elo_s,
                'ending_elo_num': e if e is not None else None,
                'ending_rank': ending_rank_s,
                'ending_rank_num': er if er is not None else None,
                'ending_pct': ending_pct_s,
                'ending_pct_num': ending_pct if (e is not None and er is not None and total_players_at_last > 0) else None,
                'current_elo': cur_elo_s,
                'current_elo_num': ce if ce is not None else None,
                'current_rank': cur_rank_s,
                'current_rank_num': cr if cr is not None else None,
                'current_pct': cur_pct_s,
                'current_pct_num': (cur_pct if (ce is not None and cr is not None and len(player_rank_map) > 0) else None),
            })

        # Use a Dash DataTable for compact display and interactive sorting
        player_table = dash_table.DataTable(
            id='team-players-table',
            columns=[
                {'name': 'Player', 'id': 'player', 'presentation': 'markdown'},
                {'name': 'Ending ELO', 'id': 'ending_elo', 'type': 'text'},
                {'name': 'Ending Rank', 'id': 'ending_rank', 'type': 'text'},
                {'name': 'Ending Percentile', 'id': 'ending_pct', 'type': 'text'},
                {'name': 'Current ELO', 'id': 'current_elo', 'type': 'text'},
                {'name': 'Current Rank', 'id': 'current_rank', 'type': 'text'},
                {'name': 'Current Percentile', 'id': 'current_pct', 'type': 'text'},
            ],
            data=table_rows,
            sort_action='custom',
            # default sort: Ending ELO (use visible column id; server callback maps to numeric helper)
            sort_by=[{'column_id': 'ending_elo', 'direction': 'desc'}],
            style_cell={
                'textAlign': 'left',
                'padding': '4px',
                'fontSize': '12px',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_header={'fontWeight': 'bold', 'fontSize': '12px'},
            style_table={'overflowX': 'auto'},
            style_cell_conditional=[
                {'if': {'column_id': 'player'}, 'textAlign': 'left', 'width': '22%'},
                {'if': {'column_id': 'ending_elo'}, 'textAlign': 'center', 'width': '12%'},
                {'if': {'column_id': 'ending_rank'}, 'textAlign': 'center', 'width': '12%'},
                {'if': {'column_id': 'ending_pct'}, 'textAlign': 'center', 'width': '12%'},
                {'if': {'column_id': 'current_elo'}, 'textAlign': 'center', 'width': '12%'},
                {'if': {'column_id': 'current_rank'}, 'textAlign': 'center', 'width': '12%'},
                {'if': {'column_id': 'current_pct'}, 'textAlign': 'center', 'width': '12%'},
            ],
            page_size=max(1, len(table_rows)),
            hidden_columns=['player_name','ending_elo_num','ending_rank_num','ending_pct_num','current_elo_num','current_rank_num','current_pct_num'],
        )
        player_lines.append(player_table)
    else:
        player_lines.append(html.P('No players'))

    # Beginning and Ending ELO with ranks inline (show rank and percentile)
    try:
        total_teams_beginning = int(df_teams['beginning_elo'].count())
    except Exception:
        total_teams_beginning = len(df_teams) if df_teams is not None else 0

    if beg is not None:
        beg_text = f"Beginning ELO: {beg:.2f}"
        b_rank = ranks.get('beginning_rank')
        if b_rank and total_teams_beginning > 0:
            b_pct = 100.0 * (total_teams_beginning - b_rank + 1) / total_teams_beginning
            beg_text += f"  Rank: #{b_rank}/{total_teams_beginning} ({b_pct:.1f}%)"
    else:
        beg_text = "Beginning ELO: N/A"

    try:
        total_teams_last = int(df_teams['last_game_elo'].count())
    except Exception:
        total_teams_last = len(df_teams) if df_teams is not None else 0

    if last is not None:
        end_text = f"Ending ELO: {last:.2f}"
        l_rank = ranks.get('last_rank')
        if l_rank and total_teams_last > 0:
            end_pct = 100.0 * (total_teams_last - l_rank + 1) / total_teams_last
            end_text += f"  Rank: #{l_rank}/{total_teams_last} ({end_pct:.1f}%)"
        elif l_rank:
            end_text += f"  Rank: #{l_rank}"
    else:
        end_text = "Ending ELO: N/A"

    # Compute top 5 biggest team ELO changes from games_with_scores (derive before/after for each game)
    top_changes = []
    try:
        for g in games_with_scores:
            # g is (dt, t1, s1, t2, s2)
            g_dt, g_t1, g_s1, g_t2, g_s2 = g
            involved = None
            if g_t1 == team:
                involved = g_t1
                opp = g_t2
            elif g_t2 == team:
                involved = g_t2
                opp = g_t1
            if not involved:
                continue
            # sample team series to get before/after
            try:
                ts = pd.to_datetime(g_dt)
                series = team_elo_history.get(team)
                if series is None or series.empty:
                    continue
                idx = series.index
                # after: value at or immediately after the game snapshot (prefer exact match, else last <= ts)
                after = None
                if ts in idx:
                    val = series.loc[ts]
                    if isinstance(val, (pd.Series, pd.DataFrame)):
                        val = val.iloc[-1]
                    after = float(val)
                else:
                    pos = idx.searchsorted(ts, side='right')
                    if pos > 0:
                        after = float(series.iloc[pos - 1])
                    else:
                        after = float(series.iloc[0])
                # before: value immediately before the snapshot (index left of ts)
                pos_left = idx.searchsorted(ts, side='left') - 1
                if pos_left >= 0:
                    before = float(series.iloc[pos_left])
                else:
                    before = float(series.iloc[0])
                delta = after - before
                # include score string similar to calculate_elo.py
                try:
                    score_str = f"{g_t1} {g_s1} - {g_s2} {g_t2}"
                except Exception:
                    score_str = ''
                top_changes.append({'time': g_dt, 'opponent': opp, 'before': before, 'after': after, 'ELO Before': before, 'ELO After': after, 'delta': delta, 'score': score_str})
            except Exception:
                continue
        # sort by absolute delta descending
        top_changes.sort(key=lambda x: abs(x['delta']), reverse=True)
        top_changes = top_changes[:5]
    except Exception:
        top_changes = []

    # Record: show win percentage and W:L:T breakdown
    wins = rec.get('wins', 0)
    losses = rec.get('losses', 0)
    draws = rec.get('draws', 0)
    games = rec.get('games', wins + losses + draws)
    try:
        # Use wins + 0.5 * draws as requested: ties count half a win
        win_pct = (100.0 * (wins + 0.5 * draws) / games) if games > 0 else None
    except Exception:
        win_pct = None
    if win_pct is not None:
        record_text = f"Record {win_pct:.1f}% W:{wins} L:{losses} T:{draws}"
    else:
        record_text = f"Record W:{wins} L:{losses} T:{draws}"

    # Top section: header and player table take full width
    top_section = html.Div([
        html.Div([html.A('Back to Teams', href='/teams', target='_self')], style={'marginBottom': '10px'}),
        html.Div(header_lines + [html.H4('Players')] + player_lines, style={'width': '100%'}),
    ], style={'width': '100%','marginBottom':'12px'})

    # Add metrics and top changes in a left column below the top section
    metrics_block = [
        html.P(beg_text),
        html.P(record_text),
        html.H4('Top 5 Biggest ELO Changes')
    ]
    if top_changes:
        for tc in top_changes:
            tm = tc['time']
            try:
                timestr = pd.to_datetime(tm).strftime('%m/%d/%Y %H:%M')
            except Exception:
                timestr = str(tm)
            metrics_block.append(html.P(f"{timestr}: vs {tc['opponent']}  {tc['before']:.2f} -> {tc['after']:.2f}  Δ{tc['delta']:+.2f}"))
    else:
        metrics_block.append(html.P('No game change data'))

    # Build bottom row: metrics column (left) and figure column (right)
    metrics_div = html.Div(metrics_block, style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'})
    if fig is not None:
        graph_div = html.Div([dcc.Graph(figure=fig, id='team-graph')], style={'width': '63%', 'display': 'inline-block', 'paddingLeft': '20px', 'verticalAlign': 'top'})
    else:
        graph_div = html.Div()

    bottom_row = html.Div([metrics_div, graph_div], style={'width': '100%'})

    return html.Div([top_section, bottom_row])


def render_changes_tab():
    """Render the Changes tab: date picker + players and teams change tables."""
    if elo_df is None or elo_df.empty:
        return html.Div([html.H3('No ELO data available')])

    # default to most recent date in the data
    try:
        default_dt = dates[-1].date().isoformat()
    except Exception:
        default_dt = datetime.now().date().isoformat()

    players_table = dash_table.DataTable(
        id='changes-players-table',
        columns=[
            {'name': 'Player', 'id': 'player'},
            {'name': 'Before', 'id': 'before', 'type': 'numeric'},
            {'name': 'After', 'id': 'after', 'type': 'numeric'},
            {'name': 'Delta', 'id': 'delta', 'type': 'numeric'},
            {'name': 'Rank Before', 'id': 'rank_before', 'type': 'numeric'},
            {'name': 'Rank After', 'id': 'rank_after', 'type': 'numeric'},
            # visible display column
            {'name': 'Rank Change', 'id': 'rank_change_display', 'type': 'text'},
            # hidden numeric column used for sorting
            {'name': 'Rank Change (sort)', 'id': 'rank_change', 'type': 'numeric'},
        ],
        data=[],
    sort_action='custom',
    # default sort: most improved (largest up) first -> rank_change ascending (negative means moved up)
    # initial sort handled server-side
    hidden_columns=['rank_change','win_pct'],
        page_size=40,
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_data_conditional=[
            {
                'if': {'column_id': 'player'},
                'color': 'blue',
                'textDecoration': 'underline',
                'cursor': 'pointer'
            }
        ],
    )

    teams_table = dash_table.DataTable(
        id='changes-teams-table',
        columns=[
            {'name': 'Team', 'id': 'team'},
            {'name': 'Before', 'id': 'before', 'type': 'numeric'},
            {'name': 'After', 'id': 'after', 'type': 'numeric'},
            {'name': 'Delta', 'id': 'delta', 'type': 'numeric'},
            # Win percentage display (string). Keep numeric win_pct for sorting as a hidden column
            {'name': 'Rank Before', 'id': 'rank_before', 'type': 'numeric'},
            {'name': 'Rank After', 'id': 'rank_after', 'type': 'numeric'},
            {'name': 'Rank Change', 'id': 'rank_change_display', 'type': 'text'},
            {'name': 'Rank Change (sort)', 'id': 'rank_change', 'type': 'numeric'},
            {'name': 'Win %', 'id': 'win_pct_display', 'type': 'text'},
            {'name': 'Win % (sort)', 'id': 'win_pct', 'type': 'numeric'},
        ],
        data=[],
    sort_action='custom',
    # default teams sorting: largest delta first (handled server-side)
    hidden_columns=['rank_change','win_pct'],
        page_size=40,
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_data_conditional=[
            {
                'if': {'column_id': 'team'},
                'color': 'blue',
                'textDecoration': 'underline',
                'cursor': 'pointer'
            }
        ],
    )

    layout = html.Div([
        html.H3('ELO Changes by Date'),
        dcc.DatePickerSingle(id='changes-date-picker', date=default_dt, display_format='MM/DD/YYYY'),
        html.Div('Change from the datapoint immediately before the selected date up to the last datapoint on that date.', style={'marginTop': '6px'}),
        html.H4('Teams', style={'marginTop': '12px'}),
        teams_table,
        html.H4('Players', style={'marginTop': '18px'}),
        players_table,
    ])
    return layout


@app.callback(
    Output('team-players-table', 'data'),
    [Input('team-players-table', 'sort_by'), Input('url', 'pathname')]
)
def team_players_table_data(sort_by, pathname):
    """Server-side populate & sort the team players table based on current URL (team page).
    This callback rebuilds the rows for the specified team and applies sorting using
    hidden numeric columns when available.
    """
    # determine team from pathname
    try:
        if not pathname or not pathname.startswith('/team/'):
            return []
        team = urllib.parse.unquote(pathname[len('/team/'):])
    except Exception:
        return []

    # validate team
    if team not in team_to_players:
        return []

    # Recreate the same members_info as in render_team_page
    members = team_to_players.get(team, [])
    if elo_df is None or elo_df.empty:
        return []
    idx = elo_df.index
    # find team's last game position
    team_last_game = None
    try:
        if team_first_last and team in team_first_last:
            team_last_game = team_first_last[team][1]
    except Exception:
        team_last_game = None
    if team_last_game is not None:
        try:
            team_last_pos = idx.searchsorted(team_last_game, side='right') - 1
            if team_last_pos < 0:
                team_last_pos = 0
        except Exception:
            team_last_pos = len(idx) - 1
    else:
        team_last_pos = len(idx) - 1

    # row at last snapshot
    row_at_last = None
    try:
        row_at_last = elo_df.iloc[team_last_pos]
    except Exception:
        row_at_last = None

    # compute global player ranks at last snapshot
    player_rank_at_last = {}
    total_players_at_last = 0
    if row_at_last is not None:
        vals = {p: row_at_last.get(p) for p in elo_df.columns}
        valid = {p: float(v) for p, v in vals.items() if v is not None and not pd.isna(v)}
        total_players_at_last = len(valid)
        sorted_all = sorted(valid.items(), key=lambda x: x[1], reverse=True)
        for i, (p, _) in enumerate(sorted_all, start=1):
            player_rank_at_last[p] = i

    # current ranks
    player_current_elos = {}
    try:
        last_row = elo_df.iloc[-1]
        for col in elo_df.columns:
            try:
                player_current_elos[col] = float(last_row[col])
            except Exception:
                pass
    except Exception:
        player_current_elos = {}
    sorted_players = sorted(player_current_elos.items(), key=lambda x: x[1], reverse=True) if player_current_elos else []
    player_rank_map = {p: i+1 for i, (p, _) in enumerate(sorted_players)}

    rows = []
    for p in members:
        try:
            ending_elo = float(row_at_last.get(p)) if (row_at_last is not None and p in row_at_last.index and not pd.isna(row_at_last.get(p))) else None
        except Exception:
            ending_elo = None
        ending_rank = player_rank_at_last.get(p)
        ending_pct = (100.0 * (total_players_at_last - ending_rank + 1) / total_players_at_last) if (ending_rank is not None and total_players_at_last>0) else None
        current_elo = player_current_elos.get(p)
        current_rank = player_rank_map.get(p)
        current_pct = (100.0 * (len(player_rank_map) - current_rank + 1) / len(player_rank_map)) if (current_rank is not None and len(player_rank_map)>0) else None

        player_md = f"[{p}](/player/{urllib.parse.quote(p)})"
        rows.append({
            'player': player_md,
            'player_name': p,
            'ending_elo': f"{ending_elo:.2f}" if ending_elo is not None else 'N/A',
            'ending_elo_num': ending_elo,
            'ending_rank': f"#{ending_rank}/{total_players_at_last}" if ending_rank is not None else 'N/A',
            'ending_rank_num': ending_rank,
            'ending_pct': f"{ending_pct:.1f}%" if ending_pct is not None else 'N/A',
            'ending_pct_num': ending_pct,
            'current_elo': f"{current_elo:.2f}" if current_elo is not None else 'N/A',
            'current_elo_num': current_elo,
            'current_rank': f"#{current_rank}/{len(player_rank_map)}" if current_rank is not None else 'N/A',
            'current_rank_num': current_rank,
            'current_pct': f"{current_pct:.1f}%" if current_pct is not None else 'N/A',
            'current_pct_num': current_pct,
        })

    # apply server-side sorting if requested
    if sort_by:
        for s in reversed(sort_by):
            col = s.get('column_id')
            direction = s.get('direction', 'asc')
            reverse = True if direction == 'desc' else False
            # map visible columns to numeric helpers when available
            mapping = {
                'ending_elo': 'ending_elo_num',
                'ending_rank': 'ending_rank_num',
                'ending_pct': 'ending_pct_num',
                'current_elo': 'current_elo_num',
                'current_rank': 'current_rank_num',
                'current_pct': 'current_pct_num',
                'player': 'player_name'
            }
            key = mapping.get(col, col)

            def keyfn(r):
                v = r.get(key)
                return (v is None, v)

            try:
                rows.sort(key=lambda r: keyfn(r), reverse=reverse)
            except Exception:
                pass

    return rows


@app.callback(
    [Output('changes-players-table', 'data'), Output('changes-teams-table', 'data')],
    [Input('changes-date-picker', 'date'), Input('changes-players-table', 'sort_by'), Input('changes-teams-table', 'sort_by')]
)
def update_changes_tables(date_iso, players_sort_by, teams_sort_by):
    """Compute before/after/delta for players and teams for the selected date."""
    if not date_iso or elo_df is None or elo_df.empty:
        return [], []

    try:
        sel_date = pd.to_datetime(date_iso).date()
    except Exception:
        return [], []

    # end of selected day
    end_ts = datetime(sel_date.year, sel_date.month, sel_date.day, 23, 59, 59, 999999)

    # helper to get (before, after) for a Series indexed by timestamps
    # if start_ts is provided, 'before' is taken as the last snapshot strictly before start_ts
    # otherwise 'before' is taken as the snapshot immediately before the 'after' snapshot
    def sample_before_after(series, start_ts=None):
        idx = series.index
        if len(idx) == 0:
            return None, None
        # after: last index <= end_ts
        pos_after = idx.searchsorted(end_ts, side='right') - 1
        if pos_after < 0:
            return None, None
        try:
            after_val = float(series.iloc[pos_after])
        except Exception:
            after_val = None

        # before: if start_ts provided, use last index < start_ts
        if start_ts is not None:
            pos_before = idx.searchsorted(start_ts, side='left') - 1
            if pos_before >= 0:
                try:
                    before_val = float(series.iloc[pos_before])
                except Exception:
                    before_val = None
            else:
                before_val = None
        else:
            # default: the sample immediately before the after sample
            prev_pos = pos_after - 1
            if prev_pos >= 0:
                try:
                    before_val = float(series.iloc[prev_pos])
                except Exception:
                    before_val = None
            else:
                before_val = None
        return before_val, after_val

    # Teams: only include teams that actually played on the selected date
    teams_played = set()
    # also capture the first game time on that date for each team
    team_first_game_ts = {}
    for g in games:
        try:
            g_dt, g_t1, g_t2 = g
        except Exception:
            continue
        try:
            if pd.to_datetime(g_dt).date() == sel_date:
                if g_t1:
                    tn = g_t1.strip()
                    teams_played.add(tn)
                    if tn not in team_first_game_ts or g_dt < team_first_game_ts[tn]:
                        team_first_game_ts[tn] = g_dt
                if g_t2:
                    tn = g_t2.strip()
                    teams_played.add(tn)
                    if tn not in team_first_game_ts or g_dt < team_first_game_ts[tn]:
                        team_first_game_ts[tn] = g_dt
        except Exception:
            continue

    # Players
    players_rows = []
    for p in sorted(list(elo_df.columns)):
        series = elo_df[p]
        # determine per-player start_ts: if the player belongs to one or more teams that
        # played on the selected date, use the earliest team's first-game timestamp as
        # the start_ts. This aligns sampling with team-level changes.
        start_ts = None
        try:
            # find teams that include this player
            player_teams = [t for t, members in team_to_players.items() if p in members]
            # collect first-game timestamps for those teams (if they played that date)
            candidate_starts = [team_first_game_ts.get(t.strip()) for t in player_teams if team_first_game_ts.get(t.strip()) is not None]
            if candidate_starts:
                # use earliest first-game timestamp among player's teams
                start_ts = min(candidate_starts)
            else:
                start_ts = None
        except Exception:
            start_ts = None
        before, after = sample_before_after(series, start_ts=start_ts)
        if before is None and after is None:
            continue
        delta = None if (before is None or after is None) else round(after - before, 2)
        players_rows.append({'player': p, 'before': None if before is None else round(before, 2), 'after': None if after is None else round(after, 2), 'delta': delta})

    # Teams
    teams_rows = []
    # Build a record of scored games up to the selected date so win% reflects history through sel_date
    team_record_upto = {}
    for dt, t1, s1, t2, s2 in games_with_scores:
        try:
            if pd.to_datetime(dt).date() <= sel_date:
                t1s = t1.strip() if isinstance(t1, str) else t1
                t2s = t2.strip() if isinstance(t2, str) else t2
                team_record_upto.setdefault(t1s, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
                team_record_upto.setdefault(t2s, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
                team_record_upto[t1s]['games'] += 1
                team_record_upto[t2s]['games'] += 1
                if s1 > s2:
                    team_record_upto[t1s]['wins'] += 1
                    team_record_upto[t2s]['losses'] += 1
                elif s2 > s1:
                    team_record_upto[t2s]['wins'] += 1
                    team_record_upto[t1s]['losses'] += 1
                else:
                    team_record_upto[t1s]['draws'] += 1
                    team_record_upto[t2s]['draws'] += 1
        except Exception:
            continue

    # Build a map from stripped team name -> canonical team key in team_to_players
    team_name_map = {t.strip(): t for t in team_to_players.keys()}

    # Build player-level record up to selected date (from scored games)
    player_record_upto = {}
    for dt, t1, s1, t2, s2 in games_with_scores:
        try:
            if pd.to_datetime(dt).date() <= sel_date:
                t1s = t1.strip() if isinstance(t1, str) else t1
                t2s = t2.strip() if isinstance(t2, str) else t2
                key1 = team_name_map.get(t1s, t1s)
                key2 = team_name_map.get(t2s, t2s)
                m1 = team_to_players.get(key1, [])
                m2 = team_to_players.get(key2, [])
                for p in m1:
                    player_record_upto.setdefault(p, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
                    player_record_upto[p]['games'] += 1
                    if s1 > s2:
                        player_record_upto[p]['wins'] += 1
                    elif s2 > s1:
                        player_record_upto[p]['losses'] += 1
                    else:
                        player_record_upto[p]['draws'] += 1
                for p in m2:
                    player_record_upto.setdefault(p, {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0})
                    player_record_upto[p]['games'] += 1
                    if s2 > s1:
                        player_record_upto[p]['wins'] += 1
                    elif s1 > s2:
                        player_record_upto[p]['losses'] += 1
                    else:
                        player_record_upto[p]['draws'] += 1
        except Exception:
            continue

    # Only include teams that played on the selected date
    teams_played_stripped = {t.strip() for t in teams_played}
    for team, series in team_elo_history.items():
        if series is None or series.empty:
            continue
        if isinstance(team, str):
            if team.strip() not in teams_played_stripped:
                continue
        else:
            continue
        # Use the team's first-game timestamp on the selected date so 'before' is the
        # last snapshot strictly before that first-game time (matches test.py behaviour).
        start_ts = team_first_game_ts.get(team.strip()) if isinstance(team, str) else None
        before, after = sample_before_after(series, start_ts=start_ts)
        if before is None and after is None:
            continue
        delta = None if (before is None or after is None) else round(after - before, 2)
        # include win percentage from computed scored-game record up to selected date; fall back to overall team_record
        # Try multiple key forms: prefer scored-games-up-to-date, then the overall record.
        rec = team_record_upto.get(team.strip()) or team_record.get(team) or team_record.get(team.strip()) or {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
        games_played = rec.get('games', 0)
        wins = rec.get('wins', 0)
        losses = rec.get('losses', 0)
        draws = rec.get('draws', 0)
        win_pct = (wins + 0.5 * draws) / games_played if games_played > 0 else None
        teams_rows.append({'team': team, 'before': None if before is None else round(before, 2), 'after': None if after is None else round(after, 2), 'delta': delta, 'win_pct': None if win_pct is None else round(win_pct, 3), 'wins': wins, 'losses': losses, 'draws': draws})

    # Players: only include players whose elo has changed (delta not None and not zero)
    players_rows = [r for r in players_rows if r.get('delta') is not None and r.get('delta') != 0]

    # sort tables by absolute delta descending for convenience (initial order)
    teams_rows.sort(key=lambda r: (abs(r['delta']) if r['delta'] is not None else 0), reverse=True)
    players_rows.sort(key=lambda r: (abs(r['delta']) if r['delta'] is not None else 0), reverse=True)

    # Compute ranks for before/after and rank change for players and teams
    try:
        # Players ranks
        if players_rows:
            dfp = pd.DataFrame(players_rows)
            for col in ['before', 'after']:
                if col in dfp.columns:
                    dfp[col + '_rank'] = dfp[col].rank(ascending=False, method='min', na_option='keep')
            # rank change: after_rank - before_rank (positive means rank number increased)
            def _rank_change(row):
                if pd.isna(row.get('before_rank')) or pd.isna(row.get('after_rank')):
                    return None
                return int(row['after_rank'] - row['before_rank'])
            if 'before_rank' in dfp.columns and 'after_rank' in dfp.columns:
                dfp['rank_change'] = dfp.apply(_rank_change, axis=1)
            # rebuild players_rows with rank fields and preserving rounding
            new_players = []
            for _, r in dfp.iterrows():
                def _to_int(v):
                    return int(v) if not pd.isna(v) else None
                pname = r['player']
                before_val = None if pd.isna(r.get('before')) else round(float(r.get('before')), 2)
                after_val = None if pd.isna(r.get('after')) else round(float(r.get('after')), 2)
                delta_val = None if pd.isna(r.get('delta')) else (round(float(r.get('delta')), 2) if isinstance(r.get('delta'), (int, float)) else r.get('delta'))
                rc = _to_int(r.get('rank_change'))
                if rc is None:
                    rc_display = ''
                else:
                    if rc < 0:
                        rc_display = '↑' + str(abs(rc))
                    elif rc > 0:
                        rc_display = '↓' + str(rc)
                    else:
                        rc_display = '0'
                # player win percentage up to selected date (from player_record_upto)
                prec = player_record_upto.get(pname) or {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
                pgames = prec.get('games', 0)
                pwins = prec.get('wins', 0)
                pdraws = prec.get('draws', 0)
                p_win_pct = (pwins + 0.5 * pdraws) / pgames if pgames > 0 else None
                plosses = prec.get('losses', 0)
                p_win_display = (f"{p_win_pct*100:.1f}% (W:{pwins} L:{plosses} T:{pdraws})" if p_win_pct is not None else '')
                new_players.append({
                    'player': pname,
                    'before': before_val,
                    'after': after_val,
                    'delta': delta_val,
                    'rank_before': _to_int(r.get('before_rank')),
                    'rank_after': _to_int(r.get('after_rank')),
                    'rank_change': rc,
                    'rank_change_display': rc_display,
                    'win_pct': None if p_win_pct is None else round(p_win_pct, 3),
                    'win_pct_display': p_win_display
                })
            players_rows = new_players
    except Exception:
        # if ranking fails, leave rows as-is
        pass

    try:
        # Teams ranks
        if teams_rows:
            dft = pd.DataFrame(teams_rows)
            for col in ['before', 'after']:
                if col in dft.columns:
                    dft[col + '_rank'] = dft[col].rank(ascending=False, method='min', na_option='keep')
            def _t_rank_change(row):
                if pd.isna(row.get('before_rank')) or pd.isna(row.get('after_rank')):
                    return None
                return int(row['after_rank'] - row['before_rank'])
            if 'before_rank' in dft.columns and 'after_rank' in dft.columns:
                dft['rank_change'] = dft.apply(_t_rank_change, axis=1)
            new_teams = []
            for _, r in dft.iterrows():
                def _to_int(v):
                    return int(v) if not pd.isna(v) else None
                tname = r['team']
                before_val = None if pd.isna(r.get('before')) else round(float(r.get('before')), 2)
                after_val = None if pd.isna(r.get('after')) else round(float(r.get('after')), 2)
                delta_val = None if pd.isna(r.get('delta')) else (round(float(r.get('delta')), 2) if isinstance(r.get('delta'), (int, float)) else r.get('delta'))
                rc = _to_int(r.get('rank_change'))
                if rc is None:
                    rc_display = ''
                else:
                    if rc < 0:
                        rc_display = '↑' + str(abs(rc))
                    elif rc > 0:
                        rc_display = '↓' + str(rc)
                    else:
                        rc_display = '0'
                new_teams.append({
                    'team': tname,
                    'before': before_val,
                    'after': after_val,
                    'delta': delta_val,
                    'rank_before': _to_int(r.get('before_rank')),
                    'rank_after': _to_int(r.get('after_rank')),
                    'rank_change': rc,
                    'rank_change_display': rc_display,
                    'win_pct': None if pd.isna(r.get('win_pct')) else round(float(r.get('win_pct')), 3),
                    'wins': int(r.get('wins')) if r.get('wins') is not None else 0,
                    'losses': int(r.get('losses')) if r.get('losses') is not None else 0,
                    'draws': int(r.get('draws')) if r.get('draws') is not None else 0,
                    'win_pct_display': (f"{float(r.get('win_pct'))*100:.1f}% (W:{int(r.get('wins') if r.get('wins') is not None else 0)} L:{int(r.get('losses') if r.get('losses') is not None else 0)} T:{int(r.get('draws') if r.get('draws') is not None else 0)})" if (not pd.isna(r.get('win_pct')) and r.get('win_pct') is not None) else '')
                })
            teams_rows = new_teams
    except Exception:
        pass

    # Server-side sorting: DataTable sends `sort_by` lists when sort_action='custom'.
    def apply_sort(rows, sort_by_list):
        if not sort_by_list:
            return rows
        # apply sorts in reverse order to mimic multi-column stable sort
        for s in reversed(sort_by_list):
            col = s.get('column_id')
            direction = s.get('direction', 'asc')
            reverse = True if direction == 'desc' else False

            def keyfn(r):
                # If the visible column is the percentage display, prefer numeric 'win_pct' for sorting
                if col == 'win_pct_display':
                    v = r.get('win_pct')
                    if v is None:
                        # fallback to display string for missing numeric
                        return r.get('win_pct_display')
                    return v
                # otherwise use the raw value if present
                return r.get(col)

            try:
                rows.sort(key=lambda r: (keyfn(r) is None, keyfn(r)), reverse=reverse)
            except Exception:
                # best-effort: leave as-is on failure
                pass
        return rows

    # Apply any user-requested sorts; if none provided, fallback to default ordering
    if players_sort_by:
        players_rows = apply_sort(players_rows, players_sort_by)
    else:
        try:
            players_rows.sort(key=lambda r: (r.get('rank_change') is None, r.get('rank_change') if r.get('rank_change') is not None else 999999))
        except Exception:
            pass

    if teams_sort_by:
        teams_rows = apply_sort(teams_rows, teams_sort_by)
    else:
        try:
            teams_rows.sort(key=lambda r: (r.get('delta') if r.get('delta') is not None else float('-inf')), reverse=True)
        except Exception:
            pass

    return players_rows, teams_rows


# (Table click navigation is handled by the unified `navigate` callback above.)


if __name__ == '__main__':
    print('Starting Dash app on http://127.0.0.1:8050')
    # Newer Dash versions use app.run instead of app.run_server
    try:
        app.run(debug=True)
    except TypeError:
        # fallback for older versions
        app.run_server(debug=True)
