"""
Probit Model with Margin of Victory Likelihood (Latent Performance Model).

This model estimates latent player skills (theta) to predict dodgeball match outcomes.
Team skill is the arithmetic mean of player skills on the roster.

The model treats all games as observing a latent performance difference y* ~ N(d, σ²):
  - Margin games: observe exact value y* = margin (PDF likelihood)
  - Binary games: observe sign of y* (CDF likelihood)

This unified framework ensures both data types share the same sigma scale.

The model combines:
  1. Probit probabilities for W-L outcomes (Gaussian CDF):
     P(A wins) = Φ((s_A - s_B) / σ)
     where Φ is the standard normal CDF

  2. Margin likelihood (only when margin observed):
     ℓ_margin = -0.5*log(2πσ²) - (d-m)²/(2σ²)  [Gaussian distribution]

Combined likelihood:
  - With margin: ℓ = log(Φ(d/σ)) + ℓ_margin
  - Outcome-only: ℓ = log(Φ(d/σ))

Loss function uses negative log-likelihood with weight-based regularization:
  Loss = -sum[ℓ_g] + sum(w_i * theta_i^2)
  where w_i = sigma_0 / (1 + beta * n_i)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
from scipy.stats import norm  # Gaussian CDF for Probit model
from collections import defaultdict
from .base import Model


# Default hyperparameters
DEFAULT_SIGMA_0 = 0.2246182199770129  # Baseline regularization weight
DEFAULT_BETA = 9.108478936592485  # Weight decay rate per game
DEFAULT_SIGMA = 4.59157898478416  # Margin noise scale (Gaussian/Normal distribution parameter)

# Visualization parameters
DEFAULT_DISPLAY_CENTER = 1000.0  # Center for exposed ratings
DEFAULT_DISPLAY_SPREAD = 100.0  # Spread for exposed ratings


class BTVetModel(Model):
    """Probit model with margin of victory likelihood (Latent Performance Model).
    
    Unifies binary and margin observations through a common Gaussian latent variable.
    Both outcome types use the same sigma scale for mathematical consistency.
    
    Attributes:
        sigma_0 (float): Baseline regularization weight
        beta (float): Weight decay rate per game played
        sigma (float): Noise scale parameter for latent performance (Gaussian std dev)
        display_center (float): Center point for exposed ratings
        display_spread (float): Spread factor for exposed ratings
        theta (dict): Dictionary mapping player IDs to skill ratings (centered at 0)
        player_game_counts (dict): Dictionary tracking number of games per player
    """
    
    def __init__(self, sigma_0=DEFAULT_SIGMA_0, beta=DEFAULT_BETA, sigma=DEFAULT_SIGMA,
                 display_center=DEFAULT_DISPLAY_CENTER, 
                 display_spread=DEFAULT_DISPLAY_SPREAD,
                 warm_start=None):
        """Initialize the Probit + MOV model with weight-based regularization.
        
        Args:
            sigma_0 (float): Baseline regularization weight
            beta (float): Weight decay rate per game played
            sigma (float): Latent performance noise scale (Gaussian standard deviation)
            display_center (float): Center for exposed ratings
            display_spread (float): Spread for exposed ratings
            warm_start (dict): Optional dictionary of player_id -> skill to initialize
        """
        self.sigma_0 = sigma_0
        self.beta = beta
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
        self.player_game_counts = defaultdict(int)
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
        """Compute Probit probabilities (Gaussian CDF) for binary outcome.
        
        This ensures the binary model uses the EXACT same sigma as the 
        margin model, unifying the physics.
        
        Args:
            skill_a (float): Team A skill
            skill_b (float): Team B skill
            
        Returns:
            tuple: (P(A wins), P(B wins))
        """
        d = skill_a - skill_b
        
        # Z-score: how many standard deviations is A better than B?
        # Note: We divide by sigma because the latent noise is N(0, sigma^2)
        z_score = d / self.sigma
        
        p_a_wins = norm.cdf(z_score)
        p_b_wins = 1.0 - p_a_wins
        
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
        """Compute negative log-likelihood (Probit + Margin).
        
        Uses if/else to select between:
        - MARGIN MODEL (PDF): When margin is observed
        - PROBIT MODEL (CDF): When only outcome is observed
        Never both for the same game (fixes double counting).
        
        Args:
            theta_vec (np.array): Vector of player skills
            player_idx_map (dict): Mapping from player_id to index in theta_vec
            
        Returns:
            float: Negative log-likelihood plus L2 regularization
        """
        temp_theta = {pid: theta_vec[idx] for pid, idx in player_idx_map.items()}
        nll = 0.0
        
        for game in self.games:
            roster_a, roster_b, outcome, margin, is_outcome_only, is_tournament = game
            
            skill_a = np.mean([temp_theta.get(p, 0.0) for p in roster_a]) if roster_a else 0.0
            skill_b = np.mean([temp_theta.get(p, 0.0) for p in roster_b]) if roster_b else 0.0
            d = skill_a - skill_b
            
            # --- SELECTION: Do we have high-res (Margin) or low-res (Binary) info? ---
            if outcome != 0 and margin > 0 and not is_outcome_only:
                # MARGIN MODEL (PDF)
                # We observed the exact performance difference.
                signed_margin = float(margin) if outcome == 1 else -float(margin)
                ll_component = self._margin_likelihood(d, signed_margin)
                
            else:
                # PROBIT MODEL (CDF)
                # We only observed who was better, not by how much.
                p_a_wins, p_b_wins = self._compute_probabilities(skill_a, skill_b)
                
                if outcome == 1:
                    ll_component = np.log(max(p_a_wins, 1e-10))
                elif outcome == -1:
                    ll_component = np.log(max(p_b_wins, 1e-10))
                else:
                    # Treat draws as 50/50 coin flip in this binary framework
                    ll_component = np.log(0.5)
            
            nll -= ll_component
        
        # Regularization (Unchanged)
        weight_penalty = 0.0
        for pid, idx in player_idx_map.items():
            n_i = self.player_game_counts.get(pid, 0)
            w_i = self.sigma_0 / (1.0 + self.beta * n_i)
            weight_penalty += w_i * (theta_vec[idx] ** 2)
        
        return nll + weight_penalty
    
    def _gradient(self, theta_vec, player_idx_map):
        """Compute gradient of the loss function with Probit + MOV.
        
        Calculates gradient of Log-Likelihood, then negates to get Loss gradient.
        Uses if/else to select between Margin (PDF) and Probit (CDF) models.
        """
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
            
            # We want the gradient of the Log-Likelihood with respect to d.
            # (Higher Likelihood = Better Fit)
            dL_Likelihood_dd = 0.0
            
            # --- MARGIN GRADIENT (PDF derivative) ---
            if outcome != 0 and margin > 0 and not is_outcome_only:
                signed_margin = float(margin) if outcome == 1 else -float(margin)
                # L = - (d - m)^2 / 2s^2
                # dL/dd = (m - d) / s^2
                dL_Likelihood_dd = (signed_margin - d) / sigma_sq
            
            # --- PROBIT GRADIENT (CDF derivative) ---
            else:
                z = d / self.sigma
                if outcome == 1:
                    # A wins: Maximize log(Phi(z))
                    # d/dd = (1/sigma) * PDF(z)/CDF(z)
                    pdf = norm.pdf(z)
                    cdf = max(norm.cdf(z), 1e-10)
                    dL_Likelihood_dd = (1.0 / self.sigma) * (pdf / cdf)
                    
                elif outcome == -1:
                    # B wins: Maximize log(1 - Phi(z)) = log(Phi(-z))
                    # d/dd = (1/sigma) * -PDF(z)/(1-CDF(z))
                    # (Negative because increasing d decreases prob of B winning)
                    pdf = norm.pdf(z)
                    cdf_inv = max(1.0 - norm.cdf(z), 1e-10)
                    dL_Likelihood_dd = -(1.0 / self.sigma) * (pdf / cdf_inv)
                
                else:
                    # Draw: Gradient is 0
                    dL_Likelihood_dd = 0.0

            # Gradient of Loss = - Gradient of Likelihood
            d_Loss_dd = -dL_Likelihood_dd
            
            # Distribute gradient
            for p in roster_a:
                if p in player_idx_map:
                    grad[player_idx_map[p]] += d_Loss_dd / len(roster_a)
            for p in roster_b:
                if p in player_idx_map:
                    grad[player_idx_map[p]] -= d_Loss_dd / len(roster_b)
        
        # Regularization Gradient (Unchanged)
        for pid, idx in player_idx_map.items():
            n_i = self.player_game_counts.get(pid, 0)
            w_i = self.sigma_0 / (1.0 + self.beta * n_i)
            grad[idx] += 2 * w_i * theta_vec[idx]
        
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
        
        # Update game counts for all players in this game
        for player in team1_players:
            self.player_game_counts[player] += 1
        for player in team2_players:
            self.player_game_counts[player] += 1
        
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
        from data.data_loader import load_teams, load_games, parse_time_maybe

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
            # Timestamp of last processed game
            last_time_str = last_row[0] if len(last_row) > 0 else None
            last_dt = parse_time_maybe(last_time_str) if last_time_str else None

            # Header is ['Time'] + player names
            # Verify we have the right number of columns
            if len(last_row) != len(header):
                return False

            # Reconstruct player_game_counts up to last_dt
            self.player_game_counts = defaultdict(int)
            try:
                teams_map = load_teams(os.path.join('data', 'Sports Elo - Teams.csv'))
                games = load_games(os.path.join('data', 'Sports Elo - Games.csv'))
                if last_dt is not None:
                    for row in games:
                        t = parse_time_maybe(row[0] if len(row) > 0 else '')
                        if t is None or t > last_dt:
                            continue
                        t1 = row[1] if len(row) > 1 else ''
                        t2 = row[3] if len(row) > 3 else ''
                        team1 = teams_map.get(t1, [])
                        team2 = teams_map.get(t2, [])
                        for p in team1:
                            self.player_game_counts[p] += 1
                        for p in team2:
                            self.player_game_counts[p] += 1
            except Exception:
                # If reconstruction fails, leave counts at zero; resume will still proceed
                pass

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
