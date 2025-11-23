from collections import defaultdict
import math
from copy import deepcopy
from .base import Model

"""Bradley-Terry style player skill model with margin-of-victory (MOV) term.

Rolling MAP estimation:
  - After each new game, re-optimizes player skills (and alpha) using all games seen.
  - Loss per game = logistic win/loss cross-entropy + lambda_mov * Gaussian MOV squared error.
  - Regularization: lambda_skill * sum(s_i^2).

Game row layout (as passed from calculate.py):
  [time, team1_name, score1, team2_name, score2, outcome_flag, tourney_flag, players_field?]

We derive winner from scores. MOV = score1 - score2 (positive => team1 wins).
Team skill = average of player skills for players provided via calculate.py mapping.

Notes / Limitations:
  - Re-optimization is O(G * P) each step; may get slow with many games. Consider reducing iterations or using incremental approximations if needed.
  - Lineup granularity: current framework passes full team roster each game; if per-game lineups are desired, the upstream call must supply only active players.
  - sigma_mov kept fixed; could be optimized similarly to alpha.
"""

# Default hyperparameters (internal BT math baseline)
DEFAULT_INITIAL_SKILL = 0.0          # Internal latent skill center (0)
DEFAULT_ALPHA = 1.0                  # MOV scale on raw skill difference
DEFAULT_SIGMA_MOV = 10.0             # Base MOV noise scale (higher for dodgeball variability)
DEFAULT_LAMBDA_MOV = 0.01            # Weight for MOV loss term (soft guidance)
DEFAULT_LAMBDA_SKILL = 0.001         # Regularization strength (L2 around dynamic center)
DEFAULT_ITERATIONS = 40              # Gradient descent steps per update
DEFAULT_LR_SKILL = 0.05              # Skill learning rate
DEFAULT_LR_ALPHA = 0.01              # Alpha learning rate (higher to adapt)
DEFAULT_MOV_SIGMA_SCALE = 0.05       # Heteroskedastic scaling: sigma *= (1 + scale * |mov|)
DEFAULT_HUBER_DELTA = 3.0            # Huber threshold for robust MOV loss
DEFAULT_CENTER_MODE = 'mean'         # 'zero' or 'mean' for regularization center
DEFAULT_ROBUST_MOV = True            # Use Huber loss for MOV residuals
DEFAULT_MOV_MIN_RELIABLE = 3.0       # Margin below which MOV signal is down-weighted
DEFAULT_MOV_RELIABILITY_EXP = 1.0    # Exponent shaping growth of MOV reliability weight
DEFAULT_ROSTER_BALANCE_WEIGHT = 1.0  # Influence of roster size balance on MOV weight
DEFAULT_ADAPTIVE_REG = True          # Scale regularization by current skill variance

# Display-only transformation (does not affect internal optimization)
DEFAULT_DISPLAY_CENTER = 1000.0      # Visual center (like Elo)
DEFAULT_DISPLAY_SCALE = 100.0        # Multiply internal skill for display spread


class BTMOVModel(Model):
    def __init__(
        self,
        initial_skill=DEFAULT_INITIAL_SKILL,
        alpha=DEFAULT_ALPHA,
        sigma_mov=DEFAULT_SIGMA_MOV,
        lambda_mov=DEFAULT_LAMBDA_MOV,
        lambda_skill=DEFAULT_LAMBDA_SKILL,
        iterations_per_update=DEFAULT_ITERATIONS,
        lr_skill=DEFAULT_LR_SKILL,
        lr_alpha=DEFAULT_LR_ALPHA,
        mov_sigma_scale=DEFAULT_MOV_SIGMA_SCALE,
        huber_delta=DEFAULT_HUBER_DELTA,
        center_mode=DEFAULT_CENTER_MODE,
        robust_mov=DEFAULT_ROBUST_MOV,
        display_center=DEFAULT_DISPLAY_CENTER,
        display_scale=DEFAULT_DISPLAY_SCALE,
        mov_min_reliable=DEFAULT_MOV_MIN_RELIABLE,
        mov_reliability_exp=DEFAULT_MOV_RELIABILITY_EXP,
        roster_balance_weight=DEFAULT_ROSTER_BALANCE_WEIGHT,
        adaptive_reg=DEFAULT_ADAPTIVE_REG,
    ):
        # Internal parameters
        self.initial_skill = float(initial_skill)
        self.alpha = float(alpha)
        self.sigma_mov = float(sigma_mov)
        self.lambda_mov = float(lambda_mov)
        self.lambda_skill = float(lambda_skill)
        self.iterations_per_update = int(iterations_per_update)
        self.lr_skill = float(lr_skill)
        self.lr_alpha = float(lr_alpha)
        self.mov_sigma_scale = float(mov_sigma_scale)
        self.huber_delta = float(huber_delta)
        self.center_mode = str(center_mode)
        self.robust_mov = bool(robust_mov)
        # MOV reliability weighting parameters
        self.mov_min_reliable = float(mov_min_reliable)
        self.mov_reliability_exp = float(mov_reliability_exp)
        self.roster_balance_weight = float(roster_balance_weight)
        # Adaptive regularization
        self.adaptive_reg = bool(adaptive_reg)
        # Display parameters
        self.display_center = float(display_center)
        self.display_scale = float(display_scale)
        # State
        self.skills = defaultdict(lambda: self.initial_skill)
        # Stored games: list of dicts with keys: team1, team2, winner, mov
        self.games = []

    # --- Utility functions ---
    def _sigmoid(self, x):
        """Standard logistic for BT internal probability."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _team_avg(self, players):
        if not players:
            return self.initial_skill
        return sum(self.skills[p] for p in players) / float(len(players))

    # --- Core API ---
    def update(self, game_row, team1_players, team2_players):
        """Add a game and perform rolling MAP optimization.

        game_row indices (best effort): [time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field]
        We rely on scores s1, s2. If they fail to parse, game ignored.
        """
        if not game_row or len(game_row) < 5:
            return
        try:
            s1 = int(game_row[2])
            s2 = int(game_row[4])
        except Exception:
            return  # skip malformed row

        mov = s1 - s2  # positive => team1 victory margin
        if mov > 0:
            winner = 1  # team1
        elif mov < 0:
            winner = 2  # team2
        else:
            winner = 0  # draw (treated as 0.5 outcome)

        # Record game
        self.games.append({
            'team1': list(team1_players),
            'team2': list(team2_players),
            'winner': winner,
            'mov': mov,
        })

        # Rolling MAP: optimize skills & alpha on all games so far
        self._optimize()

    def _optimize(self):
        if not self.games:
            return
        # For stability, fewer iterations early on
        iterations = max(5, self.iterations_per_update)

        for _ in range(iterations):
            # Accumulate gradients
            grad_skill = defaultdict(float)
            grad_alpha = 0.0
            total_loss = 0.0

            for g in self.games:
                team1 = g['team1']
                team2 = g['team2']
                mov = g['mov']
                winner = g['winner']
                if not team1 or not team2:
                    continue

                S1 = self._team_avg(team1)
                S2 = self._team_avg(team2)
                delta = S1 - S2
                p1 = self._sigmoid(delta)

                # Win/loss target
                if winner == 1:
                    y = 1.0
                elif winner == 2:
                    y = 0.0
                else:
                    y = 0.5  # treat draw as half-win

                # Loss components
                # Binary cross-entropy
                eps = 1e-12
                loss_wl = -(y * math.log(max(p1, eps)) + (1 - y) * math.log(max(1 - p1, eps)))

                # MOV term with heteroskedastic & optional robust (Huber) loss
                mov_pred = self.alpha * delta
                residual = mov - mov_pred
                # Invert previous heteroskedastic logic: SMALL margins get LARGER sigma (less weight)
                # effective_sigma grows when |mov| is close to 0, shrinks moderately for larger margins.
                effective_sigma = self.sigma_mov * (1.0 + self.mov_sigma_scale / (abs(mov) + 1.0))
                if self.robust_mov:
                    abs_r = abs(residual)
                    if abs_r <= self.huber_delta:
                        loss_core = 0.5 * (residual ** 2)
                        grad_r = residual  # derivative wrt residual
                    else:
                        loss_core = self.huber_delta * (abs_r - 0.5 * self.huber_delta)
                        grad_r = self.huber_delta * (1 if residual > 0 else -1)
                else:
                    loss_core = 0.5 * (residual ** 2)
                    grad_r = residual
                # Reliability weighting for MOV term
                # 1) Margin reliability: ramp from near 0 toward 1 once |mov| exceeds mov_min_reliable
                margin_rel = min(1.0, (abs(mov) / max(1e-6, self.mov_min_reliable))) ** self.mov_reliability_exp
                # 2) Roster balance reliability: games with similar roster sizes more reliable
                if team1 and team2:
                    roster_balance = min(len(team1), len(team2)) / float(max(len(team1), len(team2)))
                else:
                    roster_balance = 0.0
                roster_rel = roster_balance ** self.roster_balance_weight
                mov_weight = margin_rel * roster_rel
                # Final MOV loss with weight
                loss_mov = self.lambda_mov * mov_weight * (loss_core / (effective_sigma ** 2))
                total_loss += loss_wl + loss_mov

                # Gradients
                # d loss_wl / d delta = (p1 - y)
                dL_dDelta = (p1 - y)
                # MOV gradient component: lambda_mov * (-alpha * grad_r / effective_sigma^2)
                effective_sigma = self.sigma_mov * (1.0 + self.mov_sigma_scale / (abs(mov) + 1.0))
                dL_dDelta += self.lambda_mov * mov_weight * ( -self.alpha * grad_r / (effective_sigma ** 2) )

                # Per-player contributions
                if team1:
                    coeff1 = 1.0 / len(team1)
                    for p in team1:
                        grad_skill[p] += dL_dDelta * coeff1
                if team2:
                    coeff2 = -1.0 / len(team2)
                    for p in team2:
                        grad_skill[p] += dL_dDelta * coeff2

                # Gradient for alpha from MOV term only:
                # d/d alpha [ lambda_mov * (mov - alpha*delta)^2 / (2 sigma^2) ] =
                #   lambda_mov * (-(mov - alpha*delta) * delta / sigma^2)
                # Gradient w.r.t alpha from MOV: lambda_mov * ( -grad_r * delta / effective_sigma^2 )
                grad_alpha += self.lambda_mov * mov_weight * ( -grad_r * delta / (effective_sigma ** 2) )

            # Regularization: center around dynamic mean if center_mode == 'mean', else around initial_skill (0)
            if self.center_mode == 'mean' and self.skills:
                mean_skill = sum(self.skills.values()) / len(self.skills)
            else:
                mean_skill = self.initial_skill
            # Adaptive regularization: scale lambda_skill down when variance is high to avoid over-shrinking
            if self.adaptive_reg and self.skills:
                mean_skill_for_var = sum(self.skills.values()) / len(self.skills)
                var_skill = sum((v - mean_skill_for_var) ** 2 for v in self.skills.values()) / len(self.skills)
                reg_scale = 1.0 / (1.0 + var_skill)
            else:
                reg_scale = 1.0
            eff_lambda_skill = self.lambda_skill * reg_scale
            for p, val in self.skills.items():
                deviation = val - mean_skill
                total_loss += eff_lambda_skill * (deviation ** 2)
                grad_skill[p] += eff_lambda_skill * 2.0 * deviation

            # Gradient descent updates
            for p, gval in grad_skill.items():
                self.skills[p] -= self.lr_skill * gval
            self.alpha -= self.lr_alpha * grad_alpha
            # Soft clamp alpha to wide symmetric bounds (no sign restriction)
            if self.alpha < -100.0:
                self.alpha = -100.0
            if self.alpha > 100.0:
                self.alpha = 100.0

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        S1 = self._team_avg(team1_players)
        S2 = self._team_avg(team2_players)
        return self._sigmoid(S1 - S2)

    def expose(self, players):
        """Return displayed ratings: display_center + display_scale * internal_skill.

        Internal skill is centered at 0; visual ratings centered at display_center (e.g. 1000).
        """
        return [round(self.display_center + self.display_scale * (self.skills[p]), 2) for p in players]
