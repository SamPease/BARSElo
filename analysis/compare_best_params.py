#!/usr/bin/env python3
"""
Compare best parameters found by the two tuners.

Loads `best_trueskill_optuna_calc.json` and `best_trueskill_mov_optuna.json`,
then evaluates log-loss using both the non-MOV and MOV evaluators for each
parameter set (and cross-evaluates) to see which model predicts games better.
"""
import json
import os
from pprint import pprint

from analysis.train_trueskill_hyperparams import evaluate_params
from analysis.train_trueskill_mov_hyperparams import evaluate_params_mov


def load_best(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_params(d, defaults):
    # defaults is dict with 'mu','sigma','beta','tau','draw'
    out = {}
    params = d.get('params', {}) if d else {}
    out['mu'] = defaults.get('mu', 1000.0)
    out['sigma'] = params.get('sigma', defaults.get('sigma'))
    out['beta'] = params.get('beta', defaults.get('beta'))
    out['tau'] = params.get('tau', defaults.get('tau'))
    out['draw'] = params.get('draw', defaults.get('draw'))
    return out


def main():
    # paths produced by the tuner scripts
    calc_best_path = 'analysis/best_trueskill_optuna_calc.json'
    mov_best_path = 'analysis/best_trueskill_mov_optuna.json'

    calc_best = load_best(calc_best_path)
    mov_best = load_best(mov_best_path)

    # Provide defaults matching calculate scripts
    calc_defaults = {'mu': 1000.0, 'sigma': 333.3333333, 'beta': 166.66666665, 'tau': 3.333333333, 'draw': 0.123}
    mov_defaults = {'mu': 1000.0, 'sigma': 333.3333333, 'beta': 372.6779962, 'tau': 3.333333333, 'draw': 0.123}

    params_calc = extract_params(calc_best, calc_defaults)
    params_mov = extract_params(mov_best, mov_defaults)

    print('Best from calc tuner:')
    pprint(params_calc)
    print('\nBest from MOV tuner:')
    pprint(params_mov)

    print('\nEvaluating log-loss on full games:')

    # Evaluate non-MOV evaluator with calc params
    avg1, count1 = evaluate_params(params_calc['mu'], params_calc['sigma'], params_calc['beta'],
                                   'Sports Elo - Teams.csv', 'Sports Elo - Games.csv',
                                   tau=params_calc['tau'], draw_probability=params_calc['draw'])

    # Evaluate MOV evaluator with mov params (use sim_count=10 for stability)
    avg2, count2 = evaluate_params_mov(params_mov['mu'], params_mov['sigma'], params_mov['beta'],
                                       'Sports Elo - Teams.csv', 'Sports Elo - Games.csv',
                                       sim_count=10, tau=params_mov['tau'], draw_probability=params_mov['draw'])

    print('\nNon-MOV evaluator with calc best -> avg_logloss:', avg1, 'games:', count1)
    print('MOV evaluator with mov best -> avg_logloss:', avg2, 'games:', count2)

    # Cross-evaluate
    avg3, c3 = evaluate_params(params_mov['mu'], params_mov['sigma'], params_mov['beta'],
                               'Sports Elo - Teams.csv', 'Sports Elo - Games.csv',
                               tau=params_mov['tau'], draw_probability=params_mov['draw'])
    avg4, c4 = evaluate_params_mov(params_calc['mu'], params_calc['sigma'], params_calc['beta'],
                                   'Sports Elo - Teams.csv', 'Sports Elo - Games.csv',
                                   sim_count=10, tau=params_calc['tau'], draw_probability=params_calc['draw'])

    print('\nCross-evaluations:')
    print('Non-MOV evaluator with MOV best -> avg_logloss:', avg3, 'games:', c3)
    print('MOV evaluator with calc best -> avg_logloss:', avg4, 'games:', c4)

    out = {
        'calc_best': params_calc,
        'mov_best': params_mov,
        'results': {
            'calc_on_calc': {'avg_logloss': avg1, 'games': count1},
            'mov_on_mov': {'avg_logloss': avg2, 'games': count2},
            'calc_on_mov': {'avg_logloss': avg3, 'games': c3},
            'mov_on_calc': {'avg_logloss': avg4, 'games': c4},
        }
    }

    os.makedirs('analysis', exist_ok=True)
    with open('analysis/compare_best_params.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('\nWrote results to analysis/compare_best_params.json')


if __name__ == '__main__':
    main()
