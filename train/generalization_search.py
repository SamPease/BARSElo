#!/usr/bin/env python3
"""Hyperparameter search for new-team generalization.

Objective: Minimize mean NLL predicting outcomes for each team's games from
its first appearance onward, training only on games prior to that appearance
that do not involve the team (see evaluate_new_team_generalization.py).

Config format: same as existing train configs (model, defaults, search, trials).

Outputs written to train/:
  <model>_generalization_hyperparam_trials.csv
  <model>_generalization_best_params.json
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

MODEL_CLASSES = {
    'elo': EloModel,
    'trueskill': TrueSkillModel,
    'trueskill_mov': TrueSkillMovModel,
    'bt_mov': BTMOVModel,
}


def load_config(path):
    if not os.path.exists(path):
        raise SystemExit(f'Config file not found: {path}')
    if path.endswith('.yaml') or path.endswith('.yml'):
        if not _HAS_YAML:
            raise SystemExit('PyYAML not installed; cannot parse YAML')
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
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


def row_time(row):
    try:
        return parse_time_maybe(row[0])
    except Exception:
        return None


def chronological_key(row):
    t = row_time(row)
    if t is None:
        return (True, datetime.min)
    return (False, t)


def evaluate_generalization(model_name, params, games_all, teams_map, limit_eval_games=None, max_teams=None, freeze_eval=True):
    """Return mean NLL across teams under new-team generalization protocol."""
    # Build mapping team -> list of games
    team_games = {}
    for r in games_all:
        if len(r) < 5:
            continue
        team_games.setdefault(r[1], []).append(r)
        team_games.setdefault(r[3], []).append(r)
    teams = sorted(team_games.keys())
    if max_teams:
        teams = teams[:max_teams]

    nll_values = []

    ModelClass = MODEL_CLASSES[model_name]
    for team in teams:
        tg = team_games[team]
        times = [row_time(r) for r in tg if row_time(r) is not None]
        if not times:
            continue
        first_time = min(times)
        train_rows = []
        eval_rows = []
        for r in games_all:
            t = row_time(r)
            if t is None:
                continue
            involves_team = (r[1] == team or r[3] == team)
            if t < first_time and not involves_team:
                train_rows.append(r)
            elif involves_team and t >= first_time:
                eval_rows.append(r)
        eval_rows = sorted(eval_rows, key=chronological_key)
        if limit_eval_games:
            eval_rows = eval_rows[:limit_eval_games]
        if not eval_rows:
            continue
        model = ModelClass(**params)
        # Train
        for r in sorted(train_rows, key=chronological_key):
            t1 = r[1]; t2 = r[3]
            model.update(r, teams_map.get(t1, []), teams_map.get(t2, []))
        # Eval
        nll_sum = 0.0
        count = 0
        for r in eval_rows:
            t1 = r[1]; t2 = r[3]
            try:
                s1 = int(r[2]); s2 = int(r[4])
            except Exception:
                continue
            team1 = teams_map.get(t1, [])
            team2 = teams_map.get(t2, [])
            try:
                p1 = model.predict_win_prob(team1, team2)
            except Exception:
                continue
            p1 = max(1e-12, min(1 - 1e-12, float(p1)))
            # probability that TEAM wins
            if t1 == team:
                p_team = p1
                if s1 > s2:
                    nll_sum += -math.log(p_team); count += 1
                elif s2 > s1:
                    nll_sum += -math.log(1 - p_team); count += 1
            else:
                p_team = 1 - p1
                if s2 > s1:
                    nll_sum += -math.log(p_team); count += 1
                elif s1 > s2:
                    nll_sum += -math.log(1 - p_team); count += 1
            if not freeze_eval:
                model.update(r, team1, team2)
        if count:
            nll_values.append(nll_sum / count)
    if not nll_values:
        return float('inf')
    return sum(nll_values) / len(nll_values)


def run_search(cfg, limit_eval_games=None, max_teams=None, freeze_eval=True):
    model_name = cfg['model']
    defaults = cfg.get('defaults', {})
    search = cfg.get('search', {})
    trials = int(cfg.get('trials', 20))
    teams_map = load_teams(os.path.join('data', 'Sports Elo - Teams.csv'))
    games_all = load_games(os.path.join('data', 'Sports Elo - Games.csv'))

    study = optuna.create_study(direction='minimize')

    def objective(trial):
        params = dict(defaults)
        for name, spec in search.items():
            params[name] = build_param(trial, name, spec)
        val = evaluate_generalization(model_name, params, games_all, teams_map,
                                      limit_eval_games=limit_eval_games,
                                      max_teams=max_teams,
                                      freeze_eval=freeze_eval)
        trial.set_user_attr('params', params)
        return val

    study.optimize(objective, n_trials=trials)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, f'{model_name}_generalization_hyperparam_trials.csv')
    rows = []
    keys = set()
    for t in study.trials:
        r = {'trial': t.number, 'value': t.value}
        p = t.user_attrs.get('params', {})
        for k, v in p.items():
            r[k] = v; keys.add(k)
        rows.append(r)
    keys = sorted(keys)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['trial','value'] + keys)
        for r in sorted(rows, key=lambda x: (float('inf') if x['value'] is None else x['value'])):
            w.writerow([r['trial'], r['value']] + [r.get(k) for k in keys])

    best = study.best_trial
    best_out = {
        'trial': best.number,
        'value': best.value,
        'params': best.user_attrs.get('params', {}),
        'datetime': datetime.now(timezone.utc).isoformat(),
        'model': model_name,
        'objective': 'mean_team_generalization_nll',
        'limit_eval_games': limit_eval_games,
        'max_teams': max_teams,
        'freeze_eval': freeze_eval,
    }
    json_path = os.path.join(script_dir, f'{model_name}_generalization_best_params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_out, f, indent=2)

    print('Wrote', csv_path)
    print('Wrote', json_path)
    print('Best value:', best_out['value'])
    return csv_path, json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Model config JSON/YAML')
    ap.add_argument('--trials', type=int, help='Override trials')
    ap.add_argument('--limit-eval-games', type=int, default=None, help='First N eval games per team')
    ap.add_argument('--max-teams', type=int, default=None, help='Limit number of teams for speed')
    ap.add_argument('--no-freeze-eval', dest='freeze_eval', action='store_false', help='Update model during eval games')
    ap.set_defaults(freeze_eval=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.trials is not None:
        cfg['trials'] = args.trials

    run_search(cfg,
               limit_eval_games=args.limit_eval_games,
               max_teams=args.max_teams,
               freeze_eval=args.freeze_eval)


if __name__ == '__main__':
    main()
