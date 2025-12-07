#!/usr/bin/env python3
"""Unified training & evaluation script supporting two modes:

Mode chrono:
  - Online chronological evaluation: iterate games in time order.
  - For each non-draw game, record predicted win probability BEFORE update.
  - Metrics: NLL, Brier score, win accuracy.
  - Hyperparameter objective: average NLL over games.

Mode new_team:
  - New-team generalization: For each team T, train on games BEFORE its first
    timestamp that do not involve T; evaluate predictions for all games
    involving T from its first appearance onward (model frozen during eval).
  - Metrics aggregated across teams: mean team NLL, mean Brier, mean win accuracy.
  - Objective: mean team NLL.

Mode both:
  - Runs hyperparameter searches for selected models in both modes.
  - For each model, outputs cross-mode metrics (best chrono params evaluated in new_team mode and vice versa).

Config file (JSON) structure (see unified_config.json):
{
  "players_on_court_default": 8,
  "models": {
     "elo": {"defaults": {...}, "search": {...}, "trials": 20},
     ...
  },
  "metrics": ["nll","brier","accuracy"],
  "modes": ["chrono","new_team"],
  "default_mode": "both"
}

Usage examples:
  python unified_train.py --config train/unified_config.json --mode chrono --trials-override 5
  python unified_train.py --config train/unified_config.json --mode new_team --trials-override 5
  python unified_train.py --config train/unified_config.json --mode both --trials-override 3

Outputs (written to train/):
  chrono_best_models.json
  new_team_best_models.json
Each contains per-model object with: params, best_value, objective_metric, mode_metrics, cross_mode_metrics.
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

import optuna

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

# ------------------ Utility ------------------

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_param(trial, name, spec):
    t = spec.get('type')
    if t == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    if t == 'int':
        low = int(spec['low']); high = int(spec['high'])
        step = int(spec.get('step', 1))
        if spec.get('log', False):
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)
    # float
    low = float(spec['low']); high = float(spec['high'])
    if spec.get('log', False):
        return trial.suggest_float(name, low, high, log=True)
    return trial.suggest_float(name, low, high)


def clamp_prob(p):
    return max(1e-12, min(1 - 1e-12, float(p)))


# ------------------ Chronological evaluation ------------------

def evaluate_chrono(model_name, params, teams_map, games, players_on_court_default=8):
    ModelClass = MODEL_CLASSES[model_name]
    model = ModelClass(**params)
    # sort games by time
    def _time_key(row):
        try:
            t = parse_time_maybe(row[0])
            return (t is None, t or datetime.min)
        except Exception:
            return (True, datetime.min)
    games_sorted = sorted(games, key=_time_key)

    nll_sum = 0.0
    brier_sum = 0.0
    acc_correct = 0
    acc_total = 0
    wl_count = 0

    for row in games_sorted:
        if len(row) < 5:
            continue
        t1, s1, t2, s2 = row[1], row[2], row[3], row[4]
        team1 = teams_map.get(t1, [])
        team2 = teams_map.get(t2, [])
        try:
            s1_i = int(s1); s2_i = int(s2)
        except Exception:
            # skip if scores invalid
            continue
        # predict before update
        try:
            p1 = clamp_prob(model.predict_win_prob(team1, team2))
        except Exception:
            continue
        if s1_i != s2_i:
            # outcome perspective: team1 win probability
            outcome = 1.0 if s1_i > s2_i else 0.0
            nll_sum += -math.log(p1 if outcome == 1.0 else (1 - p1))
            brier_sum += (p1 - outcome) ** 2
            wl_count += 1
            pred_win = p1 >= 0.5
            if (pred_win and outcome == 1.0) or (not pred_win and outcome == 0.0):
                acc_correct += 1
            acc_total += 1
        # update after prediction
        try:
            model.update(row, team1, team2)
        except Exception:
            pass

    metrics = {
        'nll': (nll_sum / wl_count) if wl_count else None,
        'brier': (brier_sum / wl_count) if wl_count else None,
        'accuracy': (acc_correct / acc_total) if acc_total else None,
        'games_scored': wl_count,
    }
    return metrics


# ------------------ New-team generalization evaluation ------------------

def row_time(row):
    if not row: return None
    return parse_time_maybe(row[0])

def chronological_key(row):
    t = row_time(row)
    return (t is None, t or datetime.min)


def evaluate_new_team(model_name, params, games, teams_map, limit_eval_games=None):
    # map teams to games
    team_games = {}
    for r in games:
        if len(r) < 5:
            continue
        team_games.setdefault(r[1], []).append(r)
        team_games.setdefault(r[3], []).append(r)

    ModelClass = MODEL_CLASSES[model_name]
    team_nll = []
    team_brier = []
    team_acc = []

    for team, tg in team_games.items():
        times = [row_time(r) for r in tg if row_time(r) is not None]
        if not times:
            continue
        first_time = min(times)
        train_rows, eval_rows = [], []
        for r in games:
            t = row_time(r)
            if t is None:
                continue
            involves = (r[1] == team or r[3] == team)
            if t < first_time and not involves:
                train_rows.append(r)
            elif involves and t >= first_time:
                eval_rows.append(r)
        eval_rows = sorted(eval_rows, key=chronological_key)
        if limit_eval_games:
            eval_rows = eval_rows[:limit_eval_games]
        if not eval_rows:
            continue
        model = ModelClass(**params)
        # train phase
        for r in sorted(train_rows, key=chronological_key):
            model.update(r, teams_map.get(r[1], []), teams_map.get(r[3], []))
        # eval phase (frozen)
        nll_sum = brier_sum = 0.0
        acc_correct = acc_total = 0
        wl_count = 0
        for r in eval_rows:
            try:
                s1 = int(r[2]); s2 = int(r[4])
            except Exception:
                continue
            team1 = teams_map.get(r[1], [])
            team2 = teams_map.get(r[3], [])
            try:
                p1 = clamp_prob(model.predict_win_prob(team1, team2))
            except Exception:
                continue
            # probability from perspective of target team
            if r[1] == team:
                p_team = p1
                won = s1 > s2; lost = s2 > s1
            else:
                p_team = 1 - p1
                won = s2 > s1; lost = s1 > s2
            if s1 != s2:
                outcome = 1.0 if won else 0.0
                nll_sum += -math.log(p_team if outcome == 1.0 else (1 - p_team))
                brier_sum += (p_team - outcome) ** 2
                wl_count += 1
                pred_win = p_team >= 0.5
                if (pred_win and outcome == 1.0) or (not pred_win and outcome == 0.0):
                    acc_correct += 1
                acc_total += 1
        if wl_count:
            team_nll.append(nll_sum / wl_count)
            team_brier.append(brier_sum / wl_count)
            team_acc.append(acc_correct / acc_total if acc_total else None)

    metrics = {
        'mean_nll': (sum(team_nll) / len(team_nll)) if team_nll else None,
        'mean_brier': (sum(team_brier) / len(team_brier)) if team_brier else None,
        'mean_accuracy': (sum(a for a in team_acc if a is not None) / len([a for a in team_acc if a is not None])) if team_acc else None,
        'teams_scored': len(team_nll),
    }
    return metrics


# ------------------ Hyperparameter search wrappers ------------------

def search_model(model_name, mode, defaults, search_spec, trials, games, teams_map):
    objective_metric = 'nll' if mode == 'chrono' else 'mean_nll'

    def objective(trial):
        params = dict(defaults)
        for name, spec in search_spec.items():
            params[name] = build_param(trial, name, spec)
        if mode == 'chrono':
            m = evaluate_chrono(model_name, params, teams_map, games)
            val = m['nll'] if m['nll'] is not None else float('inf')
        else:
            m = evaluate_new_team(model_name, params, games, teams_map)
            val = m['mean_nll'] if m['mean_nll'] is not None else float('inf')
        trial.set_user_attr('metrics', m)
        trial.set_user_attr('params', params)
        return val

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    return {
        'best_value': best.value,
        'objective_metric': objective_metric,
        'params': best.user_attrs.get('params', {}),
        'mode_metrics': best.user_attrs.get('metrics', {}),
        'trial': best.number,
        'trials': trials,
    }


def cross_mode_metrics(model_name, params, games, teams_map):
    chrono = evaluate_chrono(model_name, params, teams_map, games)
    new_team = evaluate_new_team(model_name, params, games, teams_map)
    return {'chrono': chrono, 'new_team': new_team}


# ------------------ Main orchestration ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Unified config JSON')
    ap.add_argument('--mode', choices=['chrono','new_team','both'], default=None, help='Training mode to run')
    ap.add_argument('--model', choices=['elo','trueskill','trueskill_mov','bt_mov','bt_mov_time_decay'], help='Train only this model (default: all models in config)')
    ap.add_argument('--trials-override', type=int, help='Override trials for all models (quick test)')
    ap.add_argument('--limit-eval-games', type=int, default=None, help='(Future) limit eval games for speed in new_team mode')
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get('default_mode', 'both')

    players_on_court_default = int(cfg.get('players_on_court_default', 8))
    models_cfg = cfg['models']
    
    # Filter to single model if requested
    if args.model:
        if args.model not in models_cfg:
            raise SystemExit(f'Model {args.model} not found in config')
        models_cfg = {args.model: models_cfg[args.model]}

    teams_map = load_teams(os.path.join('data','Sports Elo - Teams.csv'))
    games = load_games(os.path.join('data','Sports Elo - Games.csv'))

    results_chrono = {}
    results_new_team = {}

    # Run chrono searches
    if mode in ('chrono','both'):
        for mname, mc in models_cfg.items():
            trials = args.trials_override or mc.get('trials', 20)
            res = search_model(mname, 'chrono', mc.get('defaults', {}), mc.get('search', {}), trials, games, teams_map)
            # cross-mode
            res['cross_mode_metrics'] = cross_mode_metrics(mname, res['params'], games, teams_map)['new_team']
            results_chrono[mname] = res
        out_path = os.path.join(_HERE, 'chrono_best_models.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'mode': 'chrono', 'datetime': datetime.now(timezone.utc).isoformat(), 'models': results_chrono}, f, indent=2)
        print('Wrote', out_path)

    # Run new_team searches
    if mode in ('new_team','both'):
        for mname, mc in models_cfg.items():
            trials = args.trials_override or mc.get('trials', 20)
            res = search_model(mname, 'new_team', mc.get('defaults', {}), mc.get('search', {}), trials, games, teams_map)
            # cross-mode
            res['cross_mode_metrics'] = cross_mode_metrics(mname, res['params'], games, teams_map)['chrono']
            results_new_team[mname] = res
        out_path = os.path.join(_HERE, 'new_team_best_models.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'mode': 'new_team', 'datetime': datetime.now(timezone.utc).isoformat(), 'models': results_new_team}, f, indent=2)
        print('Wrote', out_path)

    # If both, also produce a summary side-by-side file
    if mode == 'both':
        summary = {
            'datetime': datetime.now(timezone.utc).isoformat(),
            'chrono': results_chrono,
            'new_team': results_new_team,
        }
        out_path = os.path.join(_HERE, 'both_modes_summary.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print('Wrote', out_path)


if __name__ == '__main__':
    main()
