"""
Bradley-Terry with Ties (Davidson) Model with Margin of Victory Weighting.

This model estimates latent player skills (theta) to predict dodgeball match outcomes.
Team skill is the arithmetic mean of player skills on the roster.

The Davidson formulation handles wins, losses, and draws:
  P(A > B) = exp(S_A - S_B) / Z
  P(B > A) = 1 / Z
  P(Draw) = nu / Z
  where Z = 1 + exp(S_A - S_B) + nu

Loss function uses weighted negative log-likelihood with L2 regularization:
  Loss = -sum[w_g * ln(Likelihood_g)] + lambda * sum(theta_i^2)
  where w_g = 1 + ln(1 + Margin)
"""

import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from .base import Model


# Default hyperparameters
DEFAULT_LAMBDA = 0.10034885180465825  # L2 regularization strength
DEFAULT_NU = 0.0390120365292001  # Tie probability parameter
DEFAULT_DISPLAY_CENTER = 1000.0  # Center for exposed ratings
DEFAULT_DISPLAY_SPREAD = 200.0  # Spread for exposed ratings


class BTMOVModel(Model):
    """Bradley-Terry with Ties model using margin of victory weighting.
    
    Attributes:
        lambda_reg (float): L2 regularization strength for priors
        nu (float): Geometric tie parameter controlling draw probability
        display_center (float): Center point for exposed ratings
        display_spread (float): Spread factor for exposed ratings
        theta (dict): Dictionary mapping player IDs to skill ratings (centered at 0)
        player_ids (list): Ordered list of player IDs for optimization
    """
    
    def __init__(self, lambda_reg=DEFAULT_LAMBDA, nu=DEFAULT_NU, 
                 display_center=DEFAULT_DISPLAY_CENTER, 
                 display_spread=DEFAULT_DISPLAY_SPREAD,
                 warm_start=None):
        """Initialize the Bradley-Terry MOV model.
        
        Args:
            lambda_reg (float): L2 regularization strength
            nu (float): Tie probability parameter
            display_center (float): Center for exposed ratings
            display_spread (float): Spread for exposed ratings  
            warm_start (dict): Optional dictionary of player_id -> skill to initialize
        """
        self.lambda_reg = lambda_reg
        self.nu = nu
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
        """Compute Davidson probabilities for outcomes.
        
        Args:
            skill_a (float): Team A skill
            skill_b (float): Team B skill
            
        Returns:
            tuple: (P(A>B), P(B>A), P(Draw))
        """
        exp_diff = np.exp(skill_a - skill_b)
        z = 1.0 + exp_diff + self.nu
        
        p_a_wins = exp_diff / z
        p_b_wins = 1.0 / z
        p_draw = self.nu / z
        
        return p_a_wins, p_b_wins, p_draw
    
    def _game_weight(self, margin):
        """Calculate weight for a game based on margin of victory.
        
        Args:
            margin (int): Absolute margin of victory
            
        Returns:
            float: Weight w_g = 1 + ln(1 + margin)
        """
        return 1.0 + np.log(1.0 + abs(margin))
    
    def _negative_log_likelihood(self, theta_vec, player_idx_map):
        """Compute weighted negative log-likelihood with L2 regularization.
        
        Args:
            theta_vec (np.array): Vector of player skills
            player_idx_map (dict): Mapping from player_id to index in theta_vec
            
        Returns:
            float: Loss value
        """
        # Temporary theta dict for this evaluation
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        
        nll = 0.0
        
        for game in self.games:
            roster_a, roster_b, outcome, margin = game
            
            # Calculate team skills
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a]) if roster_a else 0.0
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b]) if roster_b else 0.0
            
            # Compute probabilities
            p_a_wins, p_b_wins, p_draw = self._compute_probabilities(skill_a, skill_b)
            
            # Determine likelihood based on outcome
            if outcome == 1:  # Team A wins
                likelihood = p_a_wins
            elif outcome == -1:  # Team B wins
                likelihood = p_b_wins
            else:  # Draw
                likelihood = p_draw
            
            # Add small epsilon to avoid log(0)
            likelihood = max(likelihood, 1e-10)
            
            # Weight by margin
            weight = self._game_weight(margin)
            
            nll -= weight * np.log(likelihood)
        
        # Add L2 regularization
        l2_penalty = self.lambda_reg * np.sum(theta_vec ** 2)
        
        return nll + l2_penalty
    
    def _gradient(self, theta_vec, player_idx_map):
        """Compute gradient of the loss function.
        
        Args:
            theta_vec (np.array): Vector of player skills
            player_idx_map (dict): Mapping from player_id to index in theta_vec
            
        Returns:
            np.array: Gradient vector
        """
        # Temporary theta dict for this evaluation
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        
        grad = np.zeros_like(theta_vec)
        
        for game in self.games:
            roster_a, roster_b, outcome, margin = game
            
            if not roster_a or not roster_b:
                continue
            
            # Calculate team skills
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a])
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b])
            
            # Compute probabilities
            exp_diff = np.exp(skill_a - skill_b)
            z = 1.0 + exp_diff + self.nu
            
            # Derivative components
            # d/dS_A of log-likelihood varies by outcome
            if outcome == 1:  # A wins
                dL_dSa = 1.0 - exp_diff / z
            elif outcome == -1:  # B wins
                dL_dSa = -exp_diff / z
            else:  # Draw
                dL_dSa = -exp_diff / z
            
            # Weight and flip sign (we're minimizing negative log-likelihood)
            weight = self._game_weight(margin)
            dL_dSa *= -weight
            
            # Team A: gradient flows to all players equally (mean)
            for p in roster_a:
                if p in player_idx_map:
                    grad[player_idx_map[p]] += dL_dSa / len(roster_a)
            
            # Team B: opposite gradient
            for p in roster_b:
                if p in player_idx_map:
                    grad[player_idx_map[p]] -= dL_dSa / len(roster_b)
        
        # Add L2 regularization gradient
        grad += 2 * self.lambda_reg * theta_vec
        
        return grad
    
    def _optimize_skills(self):
        """Optimize player skills using L-BFGS-B."""
        if not self.games:
            return
        
        # Get all unique players involved in games
        all_players = set()
        for roster_a, roster_b, _, _ in self.games:
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
            margin = s1 - s2
        elif s2 > s1:
            outcome = -1  # Team 2 wins
            margin = s2 - s1
        else:
            outcome = 0  # Draw
            margin = 0
        
        # Store game data
        self.games.append((
            list(team1_players),
            list(team2_players),
            outcome,
            margin
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
        """Predict probability that team1 beats team2.
        
        Args:
            team1_players (list): List of player IDs on team 1
            team2_players (list): List of player IDs on team 2
            players_on_court: Unused for this model (kept for API compatibility)
            
        Returns:
            float: Probability that team1 wins (not including draws)
        """
        # Ensure optimization is done before predicting
        self._ensure_optimized()
        
        skill_a = self._team_skill(team1_players)
        skill_b = self._team_skill(team2_players)
        
        p_a_wins, p_b_wins, p_draw = self._compute_probabilities(skill_a, skill_b)
        
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
