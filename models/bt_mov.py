"""
Bradley-Terry Model with Margin of Victory Likelihood.

This model estimates latent player skills (theta) to predict dodgeball match outcomes.
Team skill is the arithmetic mean of player skills on the roster.

The model combines:
  1. Bradley-Terry probabilities for W-L outcomes:
     P(A wins) = r / (1 + r)
     P(B wins) = 1 / (1 + r)
     where d = s_A - s_B, r = exp(d)

  2. Margin likelihood (only when margin observed):
     ℓ_margin = -0.5*log(2πσ²) - (d-m)²/(2σ²)  [Gaussian distribution]

Combined likelihood:
  - With margin: ℓ = log(P_BradleyTerry(outcome)) + ℓ_margin
  - Outcome-only: ℓ = log(P_BradleyTerry(outcome))

Loss function uses negative log-likelihood with L2 regularization:
  Loss = -sum[ℓ_g] + lambda * sum(theta_i^2)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
from collections import defaultdict
from .base import Model


# Default hyperparameters
DEFAULT_LAMBDA = 0.06129515109251031  # L2 regularization strength
DEFAULT_SIGMA = 5.903207466977128  # Margin noise scale (Gaussian/Normal distribution parameter)

# Visualization parameters

DEFAULT_DISPLAY_CENTER = 1000.0  # Center for exposed ratings
DEFAULT_DISPLAY_SPREAD = 200.0  # Spread for exposed ratings


class BTMOVModel(Model):
    """Bradley-Terry model with margin of victory likelihood.
    
    Attributes:
        lambda_reg (float): L2 regularization strength for priors
        sigma (float): Noise scale parameter for margin distribution (Gaussian/Normal)
        display_center (float): Center point for exposed ratings
        display_spread (float): Spread factor for exposed ratings
        theta (dict): Dictionary mapping player IDs to skill ratings (centered at 0)
    """
    
    def __init__(self, lambda_reg=DEFAULT_LAMBDA, sigma=DEFAULT_SIGMA,
                 display_center=DEFAULT_DISPLAY_CENTER, 
                 display_spread=DEFAULT_DISPLAY_SPREAD,
                 warm_start=None):
        """Initialize the Bradley-Terry + MOV model.
        
        Args:
            lambda_reg (float): L2 regularization strength
            sigma (float): Margin noise scale parameter (Gaussian standard deviation)
            display_center (float): Center for exposed ratings
            display_spread (float): Spread for exposed ratings  
            warm_start (dict): Optional dictionary of player_id -> skill to initialize
        """
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.display_center = display_center
        self.display_spread = display_spread
        
        # Initialize player skills
        if warm_start is None:
            self.theta = defaultdict(float)
        else:
            self.theta = defaultdict(float, warm_start)
        
        # Track all games for batch optimization
        self.games = []
        self._needs_optimization = False
        
    def _team_skill(self, player_list):
        """Calculate team skill as arithmetic mean of player skills.
        
        Args:
            player_list (list): List of player IDs on the team
            
        Returns:
            float: Mean skill of the team
        """
        if not player_list:
            return 0.0
        return np.mean([self.theta[p] for p in player_list])
    
    def _compute_probabilities(self, skill_a, skill_b):
        """Compute Bradley-Terry probabilities for binary outcome.
        
        Args:
            skill_a (float): Team A skill
            skill_b (float): Team B skill
            
        Returns:
            tuple: (P(A wins), P(B wins))
        """
        d = skill_a - skill_b
        r = np.exp(d)
        
        # Bradley-Terry model: binary probabilities
        z = 1.0 + r
        
        p_a_wins = r / z
        p_b_wins = 1.0 / z
        
        return p_a_wins, p_b_wins
    
    def _margin_likelihood(self, d, signed_margin):
        """Compute Gaussian margin likelihood.
        
        Args:
            d (float): Skill difference (skill_a - skill_b)
            signed_margin (float): Positive if A won, Negative if B won
            
        Returns:
            float: Log-likelihood from margin
        """
        # Gaussian Log Likelihood (ignoring constant terms like log(2*pi))
        # L = -0.5 * log(sigma^2) - (d - margin)^2 / (2 * sigma^2)
        
        # Prevent sigma from being 0 or negative during optimization exploration
        safe_sigma = max(self.sigma, 1e-4)
        sigma_sq = safe_sigma ** 2
        
        term1 = -0.5 * np.log(sigma_sq)
        term2 = -((d - signed_margin) ** 2) / (2 * sigma_sq)
        
        return term1 + term2
    
    def _negative_log_likelihood(self, theta_vec, player_idx_map):
        """Compute negative log-likelihood (Bradley-Terry + MOV).
        
        Args:
            theta_vec (np.array): Vector of player skills
            player_idx_map (dict): Mapping from player_id to index in theta_vec
            
        Returns:
            float: Negative log-likelihood plus L2 regularization
        """
        # Temporary theta dict for this evaluation
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        
        nll = 0.0
        
        for game in self.games:
            roster_a, roster_b, outcome, margin, is_outcome_only, is_tournament = game
            
            # Calculate team skills
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a]) if roster_a else 0.0
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b]) if roster_b else 0.0
            
            d = skill_a - skill_b
            
            # Bradley-Terry probabilities (binary: no draw)
            p_a_wins, p_b_wins = self._compute_probabilities(skill_a, skill_b)
            
            # Outcome likelihood
            if outcome == 1:
                ll_outcome = np.log(max(p_a_wins, 1e-10))
            elif outcome == -1:
                ll_outcome = np.log(max(p_b_wins, 1e-10))
            else:  # outcome == 0 (draw)
                # Draws are treated as ties: split probability 50/50
                ll_outcome = np.log(0.5)
            
            # Margin likelihood (Gaussian)
            ll_margin = 0.0
            if outcome != 0 and margin > 0 and not is_outcome_only:
                # CRITICAL FIX: Sign the margin based on who won
                signed_margin = float(margin) if outcome == 1 else -float(margin)
                ll_margin = self._margin_likelihood(d, signed_margin)
            
            nll -= (ll_outcome + ll_margin)
        
        # Add L2 regularization
        l2_penalty = self.lambda_reg * np.sum(theta_vec ** 2)
        
        return nll + l2_penalty
    
    def _gradient(self, theta_vec, player_idx_map):
        """Compute gradient of the loss function with Bradley-Terry + MOV.
        
        Derivatives of:
        - Bradley-Terry log-probabilities (binary model)
        - Margin likelihood: Gaussian ℓ_margin = -0.5*log(σ²) - (d-m)²/(2σ²)
        
        Args:
            theta_vec (np.array): Vector of player skills
            player_idx_map (dict): Mapping from player_id to index in theta_vec
            
        Returns:
            np.array: Gradient vector
        """
        # Temporary theta dict
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        
        grad = np.zeros_like(theta_vec)
        sigma_sq = self.sigma ** 2
        
        for game in self.games:
            roster_a, roster_b, outcome, margin, is_outcome_only, is_tournament = game
            
            if not roster_a or not roster_b:
                continue
            
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a])
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b])
            d = skill_a - skill_b
            
            # --- Bradley-Terry Gradient (Binary outcomes) ---
            r = np.exp(d)
            z = 1.0 + r
            
            # Derivatives of log probabilities wrt d:
            # d/dd log(P_A) = d/dd [d - log(1+r)] = 1 - r/(1+r) = 1/(1+r)
            # d/dd log(P_B) = d/dd [-log(1+r)] = -r/(1+r)
            dL_A_dd = 1.0 / z
            dL_B_dd = -r / z

            if outcome == 1:
                dL_OO_dd = dL_A_dd
            elif outcome == -1:
                dL_OO_dd = dL_B_dd
            else:  # outcome == 0 (draw)
                # For draws, treat as 50/50: gradient is 0
                dL_OO_dd = 0.0
            
            # --- Gaussian Margin Gradient ---
            dL_margin_dd = 0.0
            if outcome != 0 and margin > 0 and not is_outcome_only:
                signed_margin = float(margin) if outcome == 1 else -float(margin)
                
                # Derivative of Gaussian Log Likelihood:
                # d/dd [ -(d - m)^2 / 2sigma^2 ]
                # = -2(d - m) / 2sigma^2 = -(d - m) / sigma^2 = (m - d) / sigma^2
                dL_margin_dd = (signed_margin - d) / sigma_sq
            
            # Total derivative (negative because we minimize NLL)
            d_loss_dd = -(dL_OO_dd + dL_margin_dd)
            
            # Distribute gradient
            for p in roster_a:
                if p in player_idx_map:
                    grad[player_idx_map[p]] += d_loss_dd / len(roster_a)
            for p in roster_b:
                if p in player_idx_map:
                    grad[player_idx_map[p]] -= d_loss_dd / len(roster_b)
        
        grad += 2 * self.lambda_reg * theta_vec
        return grad
    
    def _optimize_skills(self):
        """Optimize player skills using L-BFGS-B."""
        if not self.games:
            return
        
        # Get all unique players involved in games
        all_players = set()
        for roster_a, roster_b, _, _, _, _ in self.games:
            all_players.update(roster_a)
            all_players.update(roster_b)
        
        # Create ordered list and index mapping
        player_list = sorted(all_players)
        player_idx_map = {pid: idx for idx, pid in enumerate(player_list)}
        
        # Initialize theta vector with warm start values
        theta_vec = np.array([self.theta[p] for p in player_list])
        
        # Optimize using L-BFGS-B
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=theta_vec,
            args=(player_idx_map,),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Update theta with optimized values
        for pid, idx in player_idx_map.items():
            self.theta[pid] = result.x[idx]
    
    def _ensure_optimized(self):
        """Ensure skills are optimized before prediction/exposure."""
        if self._needs_optimization:
            self._optimize_skills()
            self._needs_optimization = False
    
    def update(self, game_row, team1_players, team2_players):
        """Update model with a new game.
        
        Args:
            game_row: Game data row containing outcome and score information
            team1_players (list): List of player IDs on team 1
            team2_players (list): List of player IDs on team 2
        """
        try:
            # Parse game row
            time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = (game_row + [None]*8)[:8]
        except Exception:
            return
        
        # Check if this is an outcome-only game (no actual scores)
        is_outcome_only = outcome_flag is not None and str(outcome_flag).strip() != ''
        
        # Check if this is a tournament game (no draws allowed)
        is_tournament = tourney_flag is not None and str(tourney_flag).strip() != ''
        
        # Parse scores
        try:
            s1 = max(0, int(s1))
        except Exception:
            s1 = 0
        try:
            s2 = max(0, int(s2))
        except Exception:
            s2 = 0
        
        # Determine outcome and margin
        if s1 > s2:
            outcome = 1  # Team 1 wins
            margin = s1 - s2 if not is_outcome_only else 0
        elif s2 > s1:
            outcome = -1  # Team 2 wins
            margin = s2 - s1 if not is_outcome_only else 0
        else:
            outcome = 0  # Draw
            margin = 0
        
        # Store game data with outcome_only and tournament flags
        self.games.append((
            list(team1_players),
            list(team2_players),
            outcome,
            margin,
            is_outcome_only,
            is_tournament
        ))
        
        # Mark that optimization is needed, but don't do it yet
        # This allows batch optimization when needed (expose/predict/etc)
        self._needs_optimization = True
    
    def expose(self, players):
        """Return scaled ratings for display.
        
        Scales internal 0-centered ratings to display_center with display_spread.
        
        Args:
            players (list): List of player IDs
            
        Returns:
            list: Scaled ratings rounded to 2 decimal places
        """
        # Ensure optimization is done before exposing
        self._ensure_optimized()
        
        # Get current theta values
        raw_ratings = [self.theta[p] for p in players]
        
        # Scale: display_center + display_spread * theta
        # Since theta is centered at 0, this gives a nice spread
        scaled_ratings = [
            self.display_center + self.display_spread * rating 
            for rating in raw_ratings
        ]
        
        return [round(r, 2) for r in scaled_ratings]
    
    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        """Predict probability that team1 beats team2 (using Davidson model).
        
        Returns P(team1 wins) from the Davidson trinomial model.
        Note: P(team1 wins) + P(team2 wins) < 1 due to draw probability.
        For conditional probability (given no draw), use P(A) / (P(A) + P(B)).
        
        Args:
            team1_players (list): List of player IDs on team 1
            team2_players (list): List of player IDs on team 2
            players_on_court: Unused for this model (kept for API compatibility)
            
        Returns:
            float: Probability that team1 wins
        """
        # Ensure optimization is done before predicting
        self._ensure_optimized()
        
        skill_a = self._team_skill(team1_players)
        skill_b = self._team_skill(team2_players)
        
        p_a_wins, p_b_wins = self._compute_probabilities(skill_a, skill_b)
        
        # Return probability of team1 winning
        return p_a_wins

    def load_state(self, filepath, all_players):
        """Load player skills from the last row of a results CSV.

        This method unscales the displayed ratings back to internal 0-centered values.

        Args:
            filepath (str): Path to the bt_mov_results.csv file
            all_players (list): List of all player IDs in order

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        import csv
        import os

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 2:  # Need header + at least one data row
                return False

            header = rows[0]
            last_row = rows[-1]

            # Header is ['Time'] + player names
            # Verify we have the right number of columns
            if len(last_row) != len(header):
                return False

            # Load ratings from last row and unscale them
            # Displayed rating = display_center + display_spread * theta
            # So: theta = (displayed_rating - display_center) / display_spread
            for i, player in enumerate(all_players):
                try:
                    player_idx = header.index(player)
                    displayed_rating = float(last_row[player_idx])
                    # Unscale to internal 0-centered value
                    theta = (displayed_rating - self.display_center) / self.display_spread
                    self.theta[player] = theta
                except (ValueError, IndexError):
                    # Player not found or invalid rating, keep default (0.0)
                    continue

            # Clear needs_optimization flag since we just loaded state
            self._needs_optimization = False
            return True

        except Exception:
            return False
