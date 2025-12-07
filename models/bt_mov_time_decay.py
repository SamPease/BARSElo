"""
Bradley-Terry with Ties (Davidson) Model with Margin of Victory and time-decay weighting.

This variant matches `BTMOVModel` but applies an exponential decay to older games so
recent results count more: weight_g = w_g * (decay_base ** weeks_ago), where
w_g = 1 + ln(1 + margin) and weeks_ago is computed from a reference time (by default
the latest game timestamp seen, not the wall-clock time when you run the script).
"""

import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from datetime import datetime
from data.data_loader import parse_time_maybe
from .base import Model


# Default hyperparameters
DEFAULT_LAMBDA = 0.04070413082944516  # L2 regularization strength
DEFAULT_NU = 0.008753553317575144  # Tie probability parameter
DEFAULT_DISPLAY_CENTER = 1000.0  # Center for exposed ratings
DEFAULT_DISPLAY_SPREAD = 200.0  # Spread for exposed ratings
DEFAULT_DECAY_BASE = 0.9628070005883128  # Weekly decay factor


class BTMOVTimeDecayModel(Model):
    """Bradley-Terry with Ties model using margin + time decay weighting.
    
    Attributes mirror `BTMOVModel` with an added exponential decay to de-emphasize
    older games.
    """

    def __init__(self, lambda_reg=DEFAULT_LAMBDA, nu=DEFAULT_NU,
                 display_center=DEFAULT_DISPLAY_CENTER,
                 display_spread=DEFAULT_DISPLAY_SPREAD,
                 decay_base=DEFAULT_DECAY_BASE,
                 reference_time=None,
                 warm_start=None):
        self.lambda_reg = lambda_reg
        self.nu = nu
        self.display_center = display_center
        self.display_spread = display_spread
        self.decay_base = decay_base
        # If no reference_time is provided, auto-set to the latest game timestamp seen
        self.reference_time = reference_time
        self._auto_reference = reference_time is None

        if warm_start is None:
            self.theta = defaultdict(float)
        else:
            self.theta = defaultdict(float, warm_start)

        self.games = []
        self._needs_optimization = False

    def _team_skill(self, player_list):
        if not player_list:
            return 0.0
        return np.mean([self.theta[p] for p in player_list])

    def _compute_probabilities(self, skill_a, skill_b):
        exp_diff = np.exp(skill_a - skill_b)
        z = 1.0 + exp_diff + self.nu
        p_a_wins = exp_diff / z
        p_b_wins = 1.0 / z
        p_draw = self.nu / z
        return p_a_wins, p_b_wins, p_draw

    def _game_weight(self, margin, weeks_ago):
        base_weight = 1.0 + np.log(1.0 + abs(margin))
        decay = self.decay_base ** weeks_ago
        return base_weight * decay

    def _ensure_reference_time(self):
        """Set reference_time from data if it was not provided by the caller."""
        if self.reference_time is not None:
            return
        times = [g[4] for g in self.games if g[4] is not None]
        if times:
            self.reference_time = max(times)
        else:
            # Fallback: if no parseable times, freeze at now to avoid None math
            self.reference_time = datetime.now()

    def _negative_log_likelihood(self, theta_vec, player_idx_map):
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        nll = 0.0
        self._ensure_reference_time()

        for game in self.games:
            roster_a, roster_b, outcome, margin, game_time = game
            weeks_ago = self._compute_weeks_ago(game_time)
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a]) if roster_a else 0.0
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b]) if roster_b else 0.0
            p_a_wins, p_b_wins, p_draw = self._compute_probabilities(skill_a, skill_b)
            if outcome == 1:
                likelihood = p_a_wins
            elif outcome == -1:
                likelihood = p_b_wins
            else:
                likelihood = p_draw
            likelihood = max(likelihood, 1e-10)
            weight = self._game_weight(margin, weeks_ago)
            nll -= weight * np.log(likelihood)
        l2_penalty = self.lambda_reg * np.sum(theta_vec ** 2)
        return nll + l2_penalty

    def _gradient(self, theta_vec, player_idx_map):
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        grad = np.zeros_like(theta_vec)
        self._ensure_reference_time()

        for game in self.games:
            roster_a, roster_b, outcome, margin, game_time = game
            weeks_ago = self._compute_weeks_ago(game_time)
            if not roster_a or not roster_b:
                continue
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a])
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b])
            exp_diff = np.exp(skill_a - skill_b)
            z = 1.0 + exp_diff + self.nu
            if outcome == 1:
                dL_dSa = 1.0 - exp_diff / z
            elif outcome == -1:
                dL_dSa = -exp_diff / z
            else:
                dL_dSa = -exp_diff / z
            weight = self._game_weight(margin, weeks_ago)
            dL_dSa *= -weight
            for p in roster_a:
                if p in player_idx_map:
                    grad[player_idx_map[p]] += dL_dSa / len(roster_a)
            for p in roster_b:
                if p in player_idx_map:
                    grad[player_idx_map[p]] -= dL_dSa / len(roster_b)
        grad += 2 * self.lambda_reg * theta_vec
        return grad

    def _optimize_skills(self):
        if not self.games:
            return
        self._ensure_reference_time()
        all_players = set()
        for roster_a, roster_b, _, _, _ in self.games:
            all_players.update(roster_a)
            all_players.update(roster_b)
        player_list = sorted(all_players)
        player_idx_map = {pid: idx for idx, pid in enumerate(player_list)}
        theta_vec = np.array([self.theta[p] for p in player_list])
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=theta_vec,
            args=(player_idx_map,),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        for pid, idx in player_idx_map.items():
            self.theta[pid] = result.x[idx]

    def _ensure_optimized(self):
        if self._needs_optimization:
            self._optimize_skills()
            self._needs_optimization = False

    def _compute_weeks_ago(self, game_time):
        if not game_time or not self.reference_time:
            return 0.0
        delta = self.reference_time - game_time
        seconds = max(delta.total_seconds(), 0.0)
        return seconds / (7 * 24 * 60 * 60)

    def update(self, game_row, team1_players, team2_players):
        try:
            time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = (game_row + [None]*8)[:8]
        except Exception:
            return
        try:
            s1 = max(0, int(s1))
        except Exception:
            s1 = 0
        try:
            s2 = max(0, int(s2))
        except Exception:
            s2 = 0
        if s1 > s2:
            outcome = 1
            margin = s1 - s2
        elif s2 > s1:
            outcome = -1
            margin = s2 - s1
        else:
            outcome = 0
            margin = 0
        game_time = parse_time_maybe(time) if time else None
        if self._auto_reference and game_time:
            if self.reference_time is None or game_time > self.reference_time:
                self.reference_time = game_time
        self.games.append((list(team1_players), list(team2_players), outcome, margin, game_time))
        self._needs_optimization = True

    def expose(self, players):
        self._ensure_optimized()
        raw_ratings = [self.theta[p] for p in players]
        scaled_ratings = [self.display_center + self.display_spread * rating for rating in raw_ratings]
        return [round(r, 2) for r in scaled_ratings]

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        self._ensure_optimized()
        skill_a = self._team_skill(team1_players)
        skill_b = self._team_skill(team2_players)
        p_a_wins, p_b_wins, p_draw = self._compute_probabilities(skill_a, skill_b)
        return p_a_wins

    def load_state(self, filepath, all_players):
        import csv
        import os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if len(rows) < 2:
                return False
            header = rows[0]
            last_row = rows[-1]
            if len(last_row) != len(header):
                return False
            for i, player in enumerate(all_players):
                try:
                    player_idx = header.index(player)
                    displayed_rating = float(last_row[player_idx])
                    theta = (displayed_rating - self.display_center) / self.display_spread
                    self.theta[player] = theta
                except (ValueError, IndexError):
                    continue
            self._needs_optimization = False
            return True
        except Exception:
            return False
