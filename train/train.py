#!/usr/bin/env python3
"""Hyperparameter search driver for rating models.

Usage:
  python train.py --config configs/sample_trueskill.json --trials 50

The config file is JSON (or YAML if you have PyYAML installed) and should
contain keys:
  - model: one of 'elo'|'trueskill'|'trueskill_mov'
  - defaults: {param: value, ...}  # passed to model constructor when param not searched
  - search: {param: {type: 'float'|'int'|'categorical', ...}, ...}
  - players_on_court_default: 8
  - trials: optional override

The script runs Optuna, evaluates average negative log-likelihood (NLL)
online by iterating games chronologically, and writes two outputs under
`viz/`:
  - `<model>_hyperparam_trials.csv` (all trials sorted by NLL)
  - `<model>_best_params.json` (best trial params and NLL)

The objective policy:
  - For each game, compute predicted win probability p = model.predict_win_prob(...)
    (pass players_on_court if present in the game row). If p is not implemented,
    trial is pruned as non-applicable.
  - Accumulate NLL = -log(p) for the eventual winner (skip draws for scoring).
  - After accumulating over games, return average NLL (lower is better).

"""

import argparse
import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone

# Ensure repo root is on sys.path when this script is executed from `train/` directory
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
from models.bt_vet import BTVetModel

MODEL_CLASSES = {
    'elo': EloModel,
    'trueskill': TrueSkillModel,
    'trueskill_mov': TrueSkillMovModel,
    'bt_mov': BTMOVModel,
    'bt_mov_time_decay': BTMOVTimeDecayModel,
    'bt_vet': BTVetModel,
}


def load_config(path):
    # Accept either absolute/relative path. If not found, try resolving
    # relative to this script's directory (so `train/sample_trueskill.json`
    # is found when running `train/train.py`).
    if not os.path.exists(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, path)
        if os.path.exists(candidate):
            path = candidate
        else:
            # also try a `configs/` subfolder inside train/
            candidate2 = os.path.join(script_dir, 'configs', os.path.basename(path))
            if os.path.exists(candidate2):
                path = candidate2
    if not os.path.exists(path):
        raise SystemExit(f'Config file not found: {path}')

    if path.endswith('.yaml') or path.endswith('.yml'):
        if not _HAS_YAML:
            raise SystemExit('PyYAML is not installed; cannot read YAML config')
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def build_param_from_search(trial, name, spec):
    t = spec.get('type')
    if t == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    if t == 'int':
        low = int(spec['low'])
        high = int(spec['high'])
        step = int(spec.get('step', 1))
        if spec.get('log', False):
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)
    # default to float
    low = float(spec['low'])
    high = float(spec['high'])
    if spec.get('log', False):
        return trial.suggest_float(name, low, high, log=True)
    return trial.suggest_float(name, low, high)


def evaluate_params(model_name, params, teams_map, games, players_on_court_default=8):
    """Run online evaluation (average NLL) for given parameters.

    The evaluation iterates games chronologically; for each non-draw game we
    compute the predictive probability and accumulate NLL, then update the
    model with the observed result.
    """
    ModelClass = MODEL_CLASSES[model_name]
    model = ModelClass(**params)

    nll_sum = 0.0
    nll_count = 0

    # Ensure games are processed in chronological order (time series evaluation)
    def _time_key(row):
        try:
            t = parse_time_maybe(row[0] if len(row) > 0 else '')
            return (t is None, t or datetime.min)
        except Exception:
            return (True, datetime.min)

    games_sorted = sorted(games, key=_time_key)

    for row in games_sorted:
        # unpack safely
        time = row[0] if len(row) > 0 else ''
        t1 = row[1] if len(row) > 1 else ''
        s1 = row[2] if len(row) > 2 else ''
        t2 = row[3] if len(row) > 3 else ''
        s2 = row[4] if len(row) > 4 else ''
        players_field = row[7] if len(row) > 7 else None

        team1 = teams_map.get(t1, [])
        team2 = teams_map.get(t2, [])

        # determine players_on_court for this game
        try:
            players_total = int(players_field) if players_field and str(players_field).strip() else players_on_court_default
        except Exception:
            players_total = players_on_court_default

        # determine scores
        try:
            s1_i = int(s1)
            s2_i = int(s2)
        except Exception:
            # if scores missing, skip this game for scoring but still update with default
            s1_i, s2_i = None, None

        # compute predicted win probability if model supports it
        try:
            p = model.predict_win_prob(team1, team2, players_on_court=players_total)
        except NotImplementedError:
            # model does not implement probability; mark unsearchable
            return float('inf')
        except Exception:
            return float('inf')

        # clamp
        if p is None or not isinstance(p, (float, int)):
            return float('inf')
        p = max(1e-12, min(1 - 1e-12, float(p)))

        # if we have a non-draw result, accumulate NLL
        if s1_i is not None and s2_i is not None:
            if s1_i > s2_i:
                nll_sum += -math.log(p)
                nll_count += 1
            elif s2_i > s1_i:
                nll_sum += -math.log(1.0 - p)
                nll_count += 1
            else:
                # draw: use symmetric loss (optional). We'll skip draws in scoring.
                pass

        # update model with observed game
        try:
            model.update(row, team1, team2)
        except Exception:
            # If model update fails for these params, penalize
            return float('inf')

    if nll_count == 0:
        return float('inf')
    return nll_sum / nll_count


def run_search(config, teams_map, games, config_path=None, trials_override=None):
    model_name = config['model']
    if model_name not in MODEL_CLASSES:
        raise SystemExit('Unknown model in config: ' + str(model_name))

    defaults = config.get('defaults', {})
    search = config.get('search', {})
    trials = int(config.get('trials', 50))
    players_on_court_default = int(config.get('players_on_court_default', 8))

    # Eval-only shortcut: if trials == 0, compute NLL for defaults and persist minimal outputs.
    if trials == 0:
        params = dict(defaults)
        nll = evaluate_params(model_name, params, teams_map, games, players_on_court_default)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(script_dir, exist_ok=True)
        # Minimal CSV
        csv_path = os.path.join(script_dir, f'{model_name}_hyperparam_trials.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['trial', 'value'] + list(params.keys()))
            w.writerow([0, nll] + [params[k] for k in params.keys()])
        # Minimal best params JSON
        json_path = os.path.join(script_dir, f'{model_name}_best_params.json')
        best_out = {
            'trial': 0,
            'value': nll,
            'params': params,
            'datetime': datetime.now(timezone.utc).isoformat(),
            'eval_only': True,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_out, f, indent=2)
        return csv_path, json_path

    study = optuna.create_study(direction='minimize')

    def objective(trial):
        # build params: use defaults unless searched
        params = dict(defaults)
        for name, spec in search.items():
            params[name] = build_param_from_search(trial, name, spec)

        val = evaluate_params(model_name, params, teams_map, games, players_on_court_default)
        trial.set_user_attr('params', params)
        return val

    study.optimize(objective, n_trials=trials)

    # write trials to CSV inside the `train/` folder (script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    csv_path = os.path.join(script_dir, f'{model_name}_hyperparam_trials.csv')
    keys = set()
    rows = []
    for t in study.trials:
        row = {'trial': t.number, 'value': t.value}
        p = t.user_attrs.get('params', {})
        for k, v in p.items():
            row[k] = v
            keys.add(k)
        rows.append(row)
    keys = sorted(keys)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'value'] + keys)
        # sort by value ascending
        for r in sorted(rows, key=lambda x: (float('inf') if x['value'] is None else x['value'])):
            writer.writerow([r.get('trial'), r.get('value')] + [r.get(k) for k in keys])

    # best trial
    best = study.best_trial
    best_out = {
        'trial': best.number,
        'value': best.value,
        'params': best.user_attrs.get('params', {}),
        'datetime': datetime.now(timezone.utc).isoformat(),
    }
    json_path = os.path.join(script_dir, f'{model_name}_best_params.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_out, f, indent=2)

    # Append run metadata to a persistent training log in the `train/` folder
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, 'training_runs.jsonl')
        run_entry = {
            'datetime': datetime.now(timezone.utc).isoformat(),
            'config_path': config_path,
            'model': model_name,
            'trials_requested': trials,
            'trials_csv': csv_path,
            'best_json': json_path,
            'best': best_out,
        }
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(json.dumps(run_entry) + '\n')
    except Exception:
        # Don't fail the run if logging fails; best outputs already written.
        pass
    # Also maintain an aggregated CSV summary for easy inspection
    try:
        csv_log = os.path.join(script_dir, 'training_runs.csv')
        header = ['datetime', 'config_path', 'model', 'trials_requested', 'trials_csv', 'best_json', 'best_value', 'best_params']
        write_header = not os.path.exists(csv_log)
        with open(csv_log, 'a', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                run_entry['datetime'],
                (config_path or ''),
                model_name,
                trials,
                csv_path,
                json_path,
                best_out.get('value'),
                json.dumps(best_out.get('params', {})),
            ])
    except Exception:
        pass

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, help='Config file (JSON or YAML)')
    parser.add_argument('--trials', '-t', type=int, help='Override number of trials')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.trials:
        cfg['trials'] = args.trials

    teams = load_teams(os.path.join('data', 'Sports Elo - Teams.csv'))
    games = load_games(os.path.join('data', 'Sports Elo - Games.csv'))

    csv_path, json_path = run_search(cfg, teams, games, config_path=args.config, trials_override=args.trials)
    print('Wrote', csv_path)
    print('Wrote', json_path)


if __name__ == '__main__':
    main()
