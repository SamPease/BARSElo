"""Profile per-checkpoint convergence to understand time distribution."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.data_loader import load_games, load_teams, parse_time_maybe
from models.ttt import TTTModel

teams_map = load_teams(os.path.join('data', 'Sports Elo - Teams.csv'))
games = load_games(os.path.join('data', 'Sports Elo - Games.csv'))

# Build rows_with_time like evaluate_new_team
rows_with_time = []
first_time_by_team = {}
for r in games:
    if len(r) < 5:
        continue
    t = parse_time_maybe(r[0])
    if t is None:
        continue
    rows_with_time.append((t, r))
    for tid in (r[1], r[3]):
        prev = first_time_by_team.get(tid)
        if prev is None or t < prev:
            first_time_by_team[tid] = t
rows_with_time.sort(key=lambda x: x[0])

teams_by_ft = {}
for team, ft in first_time_by_team.items():
    teams_by_ft.setdefault(ft, []).append(team)
ordered_checkpoints = sorted(teams_by_ft.keys())

m = TTTModel(beta=6.4, gamma=0.11)
train_idx = 0
converge_times = []

for i, checkpoint in enumerate(ordered_checkpoints):
    games_before = train_idx
    while train_idx < len(rows_with_time) and rows_with_time[train_idx][0] < checkpoint:
        _, r = rows_with_time[train_idx]
        m.update(r, teams_map.get(r[1], []), teams_map.get(r[3], []))
        train_idx += 1
    new_games = train_idx - games_before
    
    t0 = time.time()
    m.converge()
    elapsed = time.time() - t0
    converge_times.append((i, train_idx, new_games, elapsed))

print(f"{'CP':>3} {'Games':>5} {'New':>4} {'Time':>7}")
print("-" * 25)
total = 0
for cp, total_games, new, elapsed in converge_times:
    total += elapsed
    if elapsed > 0.01 or new > 0:  # Only show interesting rows
        print(f"{cp:3d} {total_games:5d} {new:4d} {elapsed:7.3f}s")
print(f"\nTotal convergence time: {total:.1f}s across {len(ordered_checkpoints)} checkpoints")
print(f"Checkpoints with new games: {sum(1 for _,_,n,_ in converge_times if n > 0)}")
print(f"Checkpoints skipped (no new games): {sum(1 for _,_,n,_ in converge_times if n == 0)}")
