"""
Vectorized Probit Model with Margin of Victory Likelihood (No Player Uncertainty).

This model is a simplified baseline derived from bt_uncert:
- No per-player uncertainty terms
- Team skill is the arithmetic mean of player skills
- Shared observation noise sigma for all games

Loss:
  NLL(data) + sum_i(theta_i^2 / (2 * tau^2) + log(tau))

where tau is learned jointly with player skills.
"""

import os
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.special import ndtr
from scipy.stats import norm

from .base import Model


# Default hyperparameters
DEFAULT_SIGMA = 1.1688266921457915
DEFAULT_TAU = 1.0
DEFAULT_TAU_GAMMA_A = 5
DEFAULT_TAU_GAMMA_B = 20

# Visualization parameters
DEFAULT_RATING_MEAN = 1000.0
DEFAULT_RATING_SPREAD = 200.0


class BTNormalModel(Model):
    """Vectorized BT/Probit model with learned global prior scale tau."""

    def __init__(
        self,
        sigma=DEFAULT_SIGMA,
        tau=DEFAULT_TAU,
        tau_gamma_a=DEFAULT_TAU_GAMMA_A,
        tau_gamma_b=DEFAULT_TAU_GAMMA_B,
        default_rating_mean=DEFAULT_RATING_MEAN,
        default_rating_spread=DEFAULT_RATING_SPREAD,
        warm_start=None,
    ):
        self.sigma = float(sigma)
        self.tau = max(float(tau), 1e-8)
        self.tau_gamma_a = float(tau_gamma_a)
        self.tau_gamma_b = float(tau_gamma_b)
        self.default_rating_mean = float(default_rating_mean)
        self.default_rating_spread = float(default_rating_spread)

        self.theta = defaultdict(float, warm_start if warm_start else {})
        self.games = []
        self.player_game_counts = defaultdict(int)

        self._matrices_dirty = True
        self._X = None
        self._y = None
        self._margins = None
        self._is_margin = None
        self._game_weights = None
        self._player_id_to_idx = {}
        self._idx_to_player_id = []

    def _calc_prob(self, d):
        """Standard probit probability Phi(d / sigma)."""
        z = d / max(self.sigma, 1e-8)
        return ndtr(z)

    def update(self, game_row, team1_players, team2_players):
        """Store game data and mark optimization cache dirty."""
        try:
            _, _, s1, _, s2, outcome_flag, tourney_flag, _ = (game_row + [None] * 8)[:8]
        except Exception:
            return

        is_outcome_only = outcome_flag is not None and str(outcome_flag).strip() != ''
        is_tournament = tourney_flag is not None and str(tourney_flag).strip() != ''

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
            margin = s1 - s2 if not is_outcome_only else 0
        elif s2 > s1:
            outcome = -1
            margin = s2 - s1 if not is_outcome_only else 0
        else:
            outcome = 0
            margin = 0

        self.games.append(
            {
                't1': list(team1_players),
                't2': list(team2_players),
                'y': outcome,
                'm': margin,
                'is_margin': (outcome != 0 and margin > 0 and not is_outcome_only),
                'w': 2.0 if is_tournament else 1.0,
            }
        )

        for player in team1_players:
            self.player_game_counts[player] += 1
        for player in team2_players:
            self.player_game_counts[player] += 1

        self._matrices_dirty = True

    def _build_matrices(self):
        """Compile games into sparse matrix/vector structures."""
        if not self.games:
            return

        all_players = set(self.player_game_counts.keys())
        self._idx_to_player_id = sorted(list(all_players))
        self._player_id_to_idx = {p: i for i, p in enumerate(self._idx_to_player_id)}

        n_p = len(all_players)
        n_g = len(self.games)

        X = lil_matrix((n_g, n_p))
        y = np.zeros(n_g)
        margins = np.zeros(n_g)
        is_margin = np.zeros(n_g, dtype=bool)
        weights = np.zeros(n_g)

        for g_idx, game in enumerate(self.games):
            t1 = game['t1']
            t2 = game['t2']

            n1 = len(t1)
            if n1 > 0:
                idx_1 = [self._player_id_to_idx[p] for p in t1]
                for col in idx_1:
                    X[g_idx, col] = 1.0 / n1

            n2 = len(t2)
            if n2 > 0:
                idx_2 = [self._player_id_to_idx[p] for p in t2]
                for col in idx_2:
                    X[g_idx, col] = -1.0 / n2

            y[g_idx] = game['y']
            margins[g_idx] = game['m'] if game['y'] == 1 else -game['m']
            is_margin[g_idx] = game['is_margin']
            weights[g_idx] = game['w']

        self._X = X.tocsr()
        self._y = y
        self._margins = margins
        self._is_margin = is_margin
        self._game_weights = weights
        self._matrices_dirty = False

    def _loss_and_grad(self, x_vec):
        """Compute joint loss and gradient for theta and log(tau)."""
        n_players = len(self._idx_to_player_id)
        theta_vec = x_vec[:n_players]
        log_tau = x_vec[n_players]
        tau = np.exp(log_tau)
        tau_sq = tau * tau

        d = self._X.dot(theta_vec)

        grad_d = np.zeros_like(d)
        loss = 0.0

        sigma_safe = max(self.sigma, 1e-8)
        sigma_sq = sigma_safe * sigma_safe

        mask_m = self._is_margin
        if np.any(mask_m):
            dm = d[mask_m]
            sm = self._margins[mask_m]
            w_m = self._game_weights[mask_m]

            loss += np.sum(w_m * (0.5 * np.log(sigma_sq) + (dm - sm) ** 2 / (2 * sigma_sq)))
            grad_d[mask_m] = w_m * (dm - sm) / sigma_sq

        mask_o = ~self._is_margin
        if np.any(mask_o):
            do = d[mask_o]
            yo = self._y[mask_o]
            w_o = self._game_weights[mask_o]

            z = do / sigma_safe
            pdf_z = norm.pdf(z)
            cdf_z = ndtr(z)

            mask_win_local = yo == 1
            if np.any(mask_win_local):
                cdf_win = np.maximum(cdf_z[mask_win_local], 1e-10)
                loss -= np.sum(w_o[mask_win_local] * np.log(cdf_win))
                ratio = pdf_z[mask_win_local] / cdf_win
                grad_d[np.where(mask_o)[0][mask_win_local]] = -w_o[mask_win_local] * ratio / sigma_safe

            mask_loss_local = yo == -1
            if np.any(mask_loss_local):
                cdf_inv = np.maximum(1.0 - cdf_z[mask_loss_local], 1e-10)
                loss -= np.sum(w_o[mask_loss_local] * np.log(cdf_inv))
                ratio = pdf_z[mask_loss_local] / cdf_inv
                grad_d[np.where(mask_o)[0][mask_loss_local]] = w_o[mask_loss_local] * ratio / sigma_safe

        grad_theta = self._X.T.dot(grad_d)

        # Gaussian prior with learned tau:
        # sum(theta_i^2 / (2*tau^2) + log(tau))
        theta_sq_sum = np.sum(theta_vec ** 2)
        loss += theta_sq_sum / (2.0 * tau_sq) + n_players * log_tau
        grad_theta += theta_vec / tau_sq
        grad_log_tau = -theta_sq_sum / tau_sq + n_players

        # Weak Gamma-style precision guardrail:
        # add b/tau^2 + a*log(tau)
        # d/dlog(tau) = a - 2b/tau^2
        loss += self.tau_gamma_b / tau_sq + self.tau_gamma_a * log_tau
        grad_log_tau += self.tau_gamma_a - (2.0 * self.tau_gamma_b / tau_sq)

        grad = np.concatenate([grad_theta, np.array([grad_log_tau])])
        return loss, grad

    def _optimize_skills(self):
        """Optimize player skills and tau using L-BFGS-B."""
        if not self.games:
            return

        if self._matrices_dirty:
            self._build_matrices()

        start_theta = np.array([self.theta[p] for p in self._idx_to_player_id], dtype=float)
        start_log_tau = np.log(max(self.tau, 1e-8))
        start_x = np.concatenate([start_theta, np.array([start_log_tau])])

        def combined_func(x):
            return self._loss_and_grad(x)

        result = minimize(
            combined_func,
            start_x,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6},
        )

        n_players = len(self._idx_to_player_id)
        theta_opt = result.x[:n_players]
        tau_opt = float(np.exp(result.x[n_players]))

        for idx, player_id in enumerate(self._idx_to_player_id):
            self.theta[player_id] = theta_opt[idx]
        self.tau = max(tau_opt, 1e-8)

        self._matrices_dirty = False

    def _ensure_optimized(self):
        if self._matrices_dirty:
            self._optimize_skills()

    def converge(self):
        """Batch optimization hook used by training loop checkpoints."""
        self._ensure_optimized()

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        self._ensure_optimized()

        skill_a = np.mean([self.theta[p] for p in team1_players]) if team1_players else 0.0
        skill_b = np.mean([self.theta[p] for p in team2_players]) if team2_players else 0.0
        return float(self._calc_prob(skill_a - skill_b))

    def expose(self, players):
        """Expose display ratings using mean + spread * (theta / tau).

        If "Tau" appears in `players`, it is emitted as the learned tau value
        instead of being treated like a normal player skill.
        """
        self._ensure_optimized()

        tau_safe = max(self.tau, 1e-8)
        output = []
        for p in players:
            if p == 'Tau':
                output.append(round(tau_safe, 6))
            else:
                rating = self.theta[p]
                scaled = self.default_rating_mean + self.default_rating_spread * (rating / tau_safe)
                output.append(round(scaled, 2))
        return output

    def load_state(self, filepath, all_players):
        """Load previous ratings and reconstruct theta from displayed values.

        If a "Tau" column exists, its value is loaded as tau and not treated
        as a player rating.
        """
        import csv

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

            tau_idx = None
            try:
                tau_idx = header.index('Tau')
            except ValueError:
                tau_idx = None

            if tau_idx is not None:
                try:
                    self.tau = max(float(last_row[tau_idx]), 1e-8)
                except (ValueError, IndexError):
                    pass

            tau_safe = max(self.tau, 1e-8)
            spread_safe = max(self.default_rating_spread, 1e-8)

            # displayed_rating = mean + spread * (theta / tau)
            # theta = (displayed_rating - mean) * tau / spread
            for player in all_players:
                if player == 'Tau':
                    continue
                try:
                    player_idx = header.index(player)
                    displayed_rating = float(last_row[player_idx])
                    theta = (displayed_rating - self.default_rating_mean) * tau_safe / spread_safe
                    self.theta[player] = theta
                except (ValueError, IndexError):
                    continue

            self._matrices_dirty = False
            return True

        except Exception:
            return False
