#!/usr/bin/env python3
"""Evaluate model generalization to newly appearing teams.

For each team T:
  - Find the earliest timestamp of any game involving T.
  - Train the model on all games BEFORE that timestamp that do NOT involve T.
  - Evaluate predictive performance on ALL games involving T from its first game onward
    WITHOUT updating the model on those games (freeze-eval by default).

This measures how well the model predicts outcomes for a new team formation without
being rewarded for simply incorporating its early wins into later skill updates.

Metrics per team:
  team, first_time, eval_games, wins, losses, draws, nll, brier, win_acc,
  avg_pred_win_if_win, avg_pred_win_if_loss, avg_pred_win_all

Aggregate metrics also written.

Usage:
  python evaluate_new_team_generalization.py --model elo 
  python evaluate_new_team_generalization.py --model bt_mov --config train/config_bt_mov.json

Options:
  --freeze-eval / --no-freeze-eval  (default: freeze) whether to update model on eval games.
  --limit-eval-games N  evaluate only the first N games for each team (default: all).
  --output outputs/new_team_eval.csv  path to write per-team metrics.

"""

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

from data.data_loader import load_teams, load_games, parse_time_maybe
from models.elo import EloModel
from models.trueskill import TrueSkillModel
from models.trueskill_mov import TrueSkillMovModel
from models.bt_mov import BTMOVModel
from models.bt_mov_time_decay import BTMOVTimeDecayModel

MODEL_CLASSES = {
    'elo': EloModel,
    'trueskill': TrueSkillModel,
    'trueskill_mov': TrueSkillMovModel,
    'bt_mov': BTMOVModel,
    'bt_mov_time_decay': BTMOVTimeDecayModel,
}


def load_model_config(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f'Config file not found: {path}')
    if path.endswith('.yaml') or path.endswith('.yml'):
        if not _HAS_YAML:
            raise SystemExit('PyYAML not installed; cannot parse YAML config')
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f) or {}


def build_games_by_team(games):
    team_games = {}
    for row in games:
        if len(row) < 5:
            continue
        t1 = row[1]
        t2 = row[3]
        team_games.setdefault(t1, []).append(row)
        team_games.setdefault(t2, []).append(row)
    return team_games


def row_time(row):
    if not row or len(row) == 0:
        return None
    return parse_time_maybe(row[0])


def chronological_key(row):
    t = row_time(row)
    if t is None:
        # Put unknown timestamps first to avoid leaking future info
        return (True, datetime.min)
    return (False, t)


def evaluate_team(model_name, base_params, team, games_all, team_games, freeze_eval=True, limit_eval=None):
    """Train on games before team's first appearance; evaluate on its games."""
    ModelClass = MODEL_CLASSES[model_name]
    # Identify first timestamp for this team's games (ignore rows with no parseable time)
    times = [row_time(r) for r in team_games if row_time(r) is not None]
    if not times:
        # Cannot evaluate reliably
        return None
    first_time = min(times)

    # Split training/eval sets
    train_rows = []
    eval_rows = []
    for r in games_all:
        t = row_time(r)
        if t is None:
            # Skip rows without time for training to avoid leakage
            continue
        involves_team = (r[1] == team or r[3] == team)
        if t < first_time and not involves_team:
            train_rows.append(r)
        elif involves_team and t >= first_time:
            eval_rows.append(r)

    eval_rows_sorted = sorted(eval_rows, key=chronological_key)
    if limit_eval is not None:
        eval_rows_sorted = eval_rows_sorted[:limit_eval]

    # Build teams map (players for lineups). We load full teams once outside; here we just use passed config.
    teams_map = load_teams(os.path.join('data', 'Sports Elo - Teams.csv'))
    model = ModelClass(**base_params)

    # Train phase (online updates with chronological order)
    for r in sorted(train_rows, key=chronological_key):
        t1 = r[1]; t2 = r[3]
        team1 = teams_map.get(t1, [])
        team2 = teams_map.get(t2, [])
        try:
            model.update(r, team1, team2)
        except Exception:
            # Skip problematic rows
            continue

    # Evaluation phase (frozen or updating)
    nll_sum = 0.0
    brier_sum = 0.0
    win_acc_count = 0
    win_acc_correct = 0
    wins = losses = draws = 0
    preds_if_win = []
    preds_if_loss = []
    preds_all = []

    for r in eval_rows_sorted:
        t1 = r[1]; t2 = r[3]
        try:
            s1 = int(r[2]); s2 = int(r[4])
        except Exception:
            continue
        team1 = teams_map.get(t1, [])
        team2 = teams_map.get(t2, [])
        try:
            p = model.predict_win_prob(team1, team2)
        except Exception:
            continue
        p = max(1e-12, min(1 - 1e-12, float(p)))

        # Determine which side is the target team for perspective
        if t1 == team:
            target_win_prob = p
            target_won = s1 > s2
            opponent_won = s2 > s1
        else:
            # Team appears as team2; probability of team winning is (1-p)
            target_win_prob = 1.0 - p
            target_won = s2 > s1
            opponent_won = s1 > s2

        if target_won:
            wins += 1
            nll_sum += -math.log(max(target_win_prob, 1e-12))
            preds_if_win.append(target_win_prob)
        elif opponent_won:
            losses += 1
            nll_sum += -math.log(max(1.0 - target_win_prob, 1e-12))
            preds_if_loss.append(target_win_prob)
        else:
            draws += 1
            # Draw contribution optional; skip for NLL

        # Brier score for binary outcome (excluding draws)
        if not (s1 == s2):
            outcome = 1.0 if target_won else 0.0
            brier_sum += (target_win_prob - outcome) ** 2

        if not (s1 == s2):
            win_acc_count += 1
            if (target_win_prob >= 0.5 and target_won) or (target_win_prob < 0.5 and opponent_won):
                win_acc_correct += 1

        preds_all.append(target_win_prob)

        if not freeze_eval:
            # Optionally allow model to learn after prediction (online updating) — disabled by default.
            try:
                model.update(r, team1, team2)
            except Exception:
                pass

    total_eval_games = len(eval_rows_sorted)
    nll_count = wins + losses
    brier_count = wins + losses
    nll_avg = nll_sum / nll_count if nll_count else None
    brier_avg = brier_sum / brier_count if brier_count else None
    win_acc = win_acc_correct / win_acc_count if win_acc_count else None

    return {
        'team': team,
        'first_time': first_time.isoformat(),
        'eval_games': total_eval_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'nll': nll_avg,
        'brier': brier_avg,
        'win_acc': win_acc,
        'avg_pred_win_if_win': (sum(preds_if_win) / len(preds_if_win)) if preds_if_win else None,
        'avg_pred_win_if_loss': (sum(preds_if_loss) / len(preds_if_loss)) if preds_if_loss else None,
        'avg_pred_win_all': (sum(preds_all) / len(preds_all)) if preds_all else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_CLASSES.keys()))
    ap.add_argument('--config', help='Optional model config JSON/YAML with params')
    ap.add_argument('--freeze-eval', dest='freeze_eval', action='store_true', default=True, help='Freeze model during eval (default)')
    ap.add_argument('--no-freeze-eval', dest='freeze_eval', action='store_false', help='Allow updates during evaluation games')
    ap.add_argument('--limit-eval-games', type=int, default=None, help='Evaluate only first N games per team')
    ap.add_argument('--output', default='outputs/new_team_eval.csv', help='Per-team metrics CSV output path')
    args = ap.parse_args()

    params_cfg = load_model_config(args.config) if args.config else {}
    base_params = params_cfg.get('defaults', params_cfg)  # accept same structure as train configs

    teams_map = load_teams(os.path.join('data', 'Sports Elo - Teams.csv'))
    games_all = load_games(os.path.join('data', 'Sports Elo - Games.csv'))

    team_games_map = build_games_by_team(games_all)

    results = []
    for team, tg in team_games_map.items():
        r = evaluate_team(
            model_name=args.model,
            base_params=base_params,
            team=team,
            games_all=games_all,
            team_games=tg,
            freeze_eval=args.freeze_eval,
            limit_eval=args.limit_eval_games,
        )
        if r:
            results.append(r)

    # Aggregate metrics (over teams with at least one eval game)
    agg = {}
    nll_vals = [r['nll'] for r in results if r['nll'] is not None]
    brier_vals = [r['brier'] for r in results if r['brier'] is not None]
    win_acc_vals = [r['win_acc'] for r in results if r['win_acc'] is not None]
    agg['teams_evaluated'] = len(results)
    agg['mean_nll'] = sum(nll_vals) / len(nll_vals) if nll_vals else None
    agg['mean_brier'] = sum(brier_vals) / len(brier_vals) if brier_vals else None
    agg['mean_win_acc'] = sum(win_acc_vals) / len(win_acc_vals) if win_acc_vals else None
    agg['datetime'] = datetime.now(timezone.utc).isoformat()
    agg['model'] = args.model
    agg['freeze_eval'] = args.freeze_eval

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # Write per-team CSV
    fieldnames = list(results[0].keys()) if results else ['team','first_time','eval_games','wins','losses','draws','nll','brier','win_acc','avg_pred_win_if_win','avg_pred_win_if_loss','avg_pred_win_all']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda x: x['team']):
            w.writerow(r)

    # Write aggregate JSON next to CSV
    agg_path = args.output.replace('.csv', '_aggregate.json')
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)

    print('Wrote per-team metrics:', args.output)
    print('Wrote aggregate metrics:', agg_path)
    if agg['mean_nll'] is not None:
        print('Mean NLL:', agg['mean_nll'])
        print('Mean Brier:', agg['mean_brier'])
        print('Mean Win Acc:', agg['mean_win_acc'])


if __name__ == '__main__':
    main()
