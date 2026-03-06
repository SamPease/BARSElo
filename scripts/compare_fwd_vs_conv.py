#!/usr/bin/env python3
"""Compare forward-pass vs converged TTT ratings, and benchmark convergence scaling."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import trueskillthroughtime as ttt_lib
from data.data_loader import load_teams, load_games, parse_time_maybe
from models.ttt import TTTModel

teams_map = load_teams('data/Sports Elo - Teams.csv')
games = load_games('data/Sports Elo - Games.csv')
all_players = sorted(set(p for plist in teams_map.values() for p in plist))

# Build forward-pass ratings via the model
m = TTTModel()
for r in games:
    if len(r) < 5:
        continue
    m.update(r, teams_map.get(r[1], []), teams_map.get(r[3], []))

fwd = dict(m._forward_ratings)

# Prepare data for History convergence
compositions = []
results_list = []
times_days = []
for r in games:
    if len(r) < 5:
        continue
    t1_players = teams_map.get(r[1], [])
    t2_players = teams_map.get(r[3], [])
    if not t1_players or not t2_players:
        continue
    try:
        s1, s2 = int(r[2]), int(r[4])
    except Exception:
        continue
    compositions.append([list(t1_players), list(t2_players)])
    if s1 > s2:
        results_list.append([0, 1])
    elif s2 > s1:
        results_list.append([1, 0])
    else:
        results_list.append([0, 0])
    t = parse_time_maybe(r[0])
    times_days.append(t.timestamp() / 86400.0 if t else 0.0)

# Benchmark convergence at different game counts
print("=== Convergence scaling ===")
for n in [50, 100, 200, 400, len(compositions)]:
    t0 = time.time()
    h = ttt_lib.History(
        composition=compositions[:n],
        results=results_list[:n],
        times=times_days[:n],
        sigma=m.sigma, beta=m.beta, gamma=m.gamma, p_draw=m.p_draw,
    )
    try:
        h.convergence(verbose=False)
        elapsed = time.time() - t0
        print(f"  {n:>4} games: {elapsed:.3f}s")
    except ValueError as e:
        elapsed = time.time() - t0
        print(f"  {n:>4} games: FAILED ({e}) after {elapsed:.3f}s")

# Full convergence for comparison — use all games
print("\nAttempting full convergence with all games...")
t0 = time.time()
h = ttt_lib.History(
    composition=compositions,
    results=results_list,
    times=times_days,
    sigma=m.sigma, beta=m.beta, gamma=m.gamma, p_draw=m.p_draw,
)
try:
    h.convergence(verbose=False)
    t1 = time.time()
    print(f"Full convergence: {t1-t0:.1f}s on {len(compositions)} games")
    converged = True
except ValueError as e:
    # Try manual convergence with limited iterations
    print(f"Standard convergence failed: {e}")
    print("Trying limited iterations...")
    t0 = time.time()
    h = ttt_lib.History(
        composition=compositions,
        results=results_list,
        times=times_days,
        sigma=m.sigma, beta=m.beta, gamma=m.gamma, p_draw=m.p_draw,
    )
    for _ in range(3):
        try:
            h.iteration()
        except ValueError:
            break
    t1 = time.time()
    print(f"Partial convergence (<=3 iters): {t1-t0:.1f}s")
    converged = True

lc = h.learning_curves()
conv = {}
for pid, curve in lc.items():
    if curve:
        conv[pid] = (curve[-1][1].mu, curve[-1][1].sigma)

# Compare
common = sorted([p for p in all_players if p in conv and p in fwd])
print(f"\n{'Player':<30} {'Fwd mu':>10} {'Conv mu':>10} {'Diff':>8}")
print("-" * 62)
for p in sorted(common, key=lambda p: abs(fwd[p][0] - conv[p][0]), reverse=True)[:15]:
    f_mu = fwd[p][0]
    c_mu = conv[p][0]
    print(f"{p:<30} {f_mu:>10.3f} {c_mu:>10.3f} {c_mu-f_mu:>+8.3f}")

all_diffs = [conv[p][0] - fwd[p][0] for p in common]
fwd_vals = [fwd[p][0] for p in common]
conv_vals = [conv[p][0] for p in common]

print(f"\n{len(common)} players compared:")
print(f"  Mean |diff|:  {np.mean(np.abs(all_diffs)):.3f}")
print(f"  Max |diff|:   {np.max(np.abs(all_diffs)):.3f}")
print(f"  Pearson r:    {np.corrcoef(fwd_vals, conv_vals)[0,1]:.4f}")

from scipy.stats import spearmanr
rho, _ = spearmanr(fwd_vals, conv_vals)
print(f"  Spearman rho: {rho:.4f}")

# Estimate total time for new_team mode with convergence at 44 checkpoints
# Assume growing game counts at checkpoints
print("\n=== Estimated new_team cost with convergence ===")
# Rough: average convergence ~3.5s * 44 checkpoints * 200 trials
avg_time = (t1-t0) / 2  # rough average across checkpoints
est_per_trial = avg_time * 44
est_total = est_per_trial * 200
print(f"  Avg convergence: ~{avg_time:.1f}s")
print(f"  Per trial (44 checkpoints): ~{est_per_trial:.0f}s")
print(f"  200 trials: ~{est_total/3600:.1f} hours")
