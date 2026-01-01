"""
Vectorized Probit Model with Margin of Victory Likelihood (Latent Performance Model).

This fully vectorized implementation uses sparse matrices to eliminate all Python loops
from gradient computation. The core math kernels (_calc_prob, _calc_sigma_eff) are 
unified between training and exposure, ensuring consistency.

Key benefits:
- Sparse matrices for O(1) gradient computation (matrix-vector product)
- No Python for loops in gradient calculation
- Shared mathematical kernels between training and prediction
- Scales to 50,000+ games without hanging

Latent performance: y* ~ N(d, σ_eff²) where σ_eff = sqrt(σ² + σ_team_a² + σ_team_b²)
Margin games use the PDF; outcome-only games use the CDF of this shared Gaussian scale.

Loss function: -sum[ℓ_g] + lambda * sum(theta_i²)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import norm
from scipy.special import ndtr  # Vectorized CDF of standard normal
from collections import defaultdict
from .base import Model


# Default hyperparameters
DEFAULT_SIGMA = 3.4241681825996473  # Baseline observation std dev
DEFAULT_ALPHA = 6.375363098674137  # Numerator in sigma_i formula: sigma_i = alpha / sqrt(n_i + 1)
DEFAULT_L2_LAMBDA = 0.0018383063237865321  # L2 regularization weight

# Visualization parameters
DEFAULT_MAX_RATING = 2000.0  # Maximum rating for pairwise win probability scaling


class BTUncertModel(Model):
    """
    Fully Vectorized Probit Model using Sparse Matrices.
    
    Unified implementation where training, prediction, and exposure 
    share the same vectorized mathematical kernels.
    
    Key data structures:
    - _X: Sparse matrix (n_games × n_players) of skill coefficients
      Team 1 players get +1/team_size, Team 2 players get -1/team_size
    - _Ma, _Mb: Binary sparse matrices tracking which players are on each team
    - _y: Outcome vector (-1, 0, or 1)
    - _margins: Signed margin vector
    - _is_margin: Boolean mask for games with observed margins
    """

    def __init__(self, sigma=DEFAULT_SIGMA, alpha=DEFAULT_ALPHA,
                 l2_lambda=DEFAULT_L2_LAMBDA,
                 max_rating=DEFAULT_MAX_RATING,
                 warm_start=None):
        self.sigma = sigma
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.max_rating = max_rating

        # Internal state
        self.theta = defaultdict(float, warm_start if warm_start else {})
        self.games = []
        self.player_game_counts = defaultdict(int)
        
        # Cache for sparse matrices and metadata
        self._matrices_dirty = True
        self._X = None              # Skill direction matrix (n_games × n_players)
        self._Ma = None             # Team A membership matrix (binary)
        self._Mb = None             # Team B membership matrix (binary)
        self._y = None              # Outcome vector
        self._margins = None        # Signed margin vector
        self._is_margin = None      # Boolean mask for margin games
        self._game_weights = None   # Weight per game (1.0 or 2.0 for tournaments)
        self._player_id_to_idx = {} # Mapping from player ID to matrix column index
        self._idx_to_player_id = [] # Inverse mapping

    # =========================================================================
    # 1. CORE MATH KERNELS (Single Source of Truth)
    # =========================================================================
    
    def _calc_sigma_eff(self, sigma_sq_a, sigma_sq_b):
        """
        Combine team variances into effective game variance.
        Works for scalars, vectors (training), and matrices (exposure).
        
        Formula: sqrt(sigma_global² + sigma_team_a² + sigma_team_b²)
        
        Args:
            sigma_sq_a: Team A variance (or array/matrix of variances)
            sigma_sq_b: Team B variance (or array/matrix of variances)
            
        Returns:
            Effective sigma (same shape as inputs)
        """
        total_sq = self.sigma ** 2 + sigma_sq_a + sigma_sq_b
        return np.sqrt(np.maximum(total_sq, 1e-8))

    def _calc_prob(self, d, sigma_eff):
        """
        Standard Probit Probability: Phi(d / sigma_eff).
        Works for scalars, vectors, and matrices.
        
        Args:
            d: Skill difference (Team A - Team B)
            sigma_eff: Effective observation noise
            
        Returns:
            Probability that Team A wins (P(Team A > Team B))
        """
        z = d / sigma_eff
        return ndtr(z)  # Vectorized Gaussian CDF

    def _calc_player_sigmas(self, counts_array):
        """
        Calculate per-player uncertainty sigma_i = alpha / sqrt(n_i + 1).
        Works for vectors.
        
        Args:
            counts_array: Game count for each player
            
        Returns:
            Array of sigma_i values, minimum 1e-8
        """
        return np.maximum(self.alpha / np.sqrt(counts_array + 1.0), 1e-8)

    # =========================================================================
    # 2. DATA MANAGEMENT (Sparse Matrix Construction)
    # =========================================================================

    def update(self, game_row, team1_players, team2_players):
        """
        Store game data. Matrices are rebuilt lazily before optimization.
        
        Args:
            game_row: Game data row containing outcome and score information
            team1_players (list): List of player IDs on team 1
            team2_players (list): List of player IDs on team 2
        """
        try:
            # Parse game row (matching original parsing logic)
            time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = (game_row + [None]*8)[:8]
        except Exception:
            return
        
        # Check if this is an outcome-only game (no actual scores)
        is_outcome_only = outcome_flag is not None and str(outcome_flag).strip() != ''
        
        # Check if this is a tournament game
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
            outcome = 1      # Team 1 wins
            margin = s1 - s2 if not is_outcome_only else 0
        elif s2 > s1:
            outcome = -1     # Team 2 wins
            margin = s2 - s1 if not is_outcome_only else 0
        else:
            outcome = 0      # Draw
            margin = 0
        
        # Store game with computed flags
        self.games.append({
            't1': list(team1_players),
            't2': list(team2_players),
            'y': outcome,
            'm': margin,
            'is_margin': (outcome != 0 and margin > 0 and not is_outcome_only),
            'w': 2.0 if is_tournament else 1.0
        })
        
        # Update game counts for all players
        for player in team1_players:
            self.player_game_counts[player] += 1
        for player in team2_players:
            self.player_game_counts[player] += 1
        
        # Mark matrices as needing rebuild
        self._matrices_dirty = True

    def _build_matrices(self):
        """
        Compile raw game list into sparse matrices.
        
        This is the critical step that transforms list-of-dicts into
        sparse matrix format for O(1) vectorized operations.
        """
        if not self.games:
            return
        
        # 1. Create player indexing
        all_players = set(self.player_game_counts.keys())
        self._idx_to_player_id = sorted(list(all_players))
        self._player_id_to_idx = {p: i for i, p in enumerate(self._idx_to_player_id)}
        n_p = len(all_players)
        n_g = len(self.games)

        # 2. Initialize sparse matrix builders (LIL format for fast construction)
        X = lil_matrix((n_g, n_p))   # Skill difference matrix
        Ma = lil_matrix((n_g, n_p))  # Team A membership (binary)
        Mb = lil_matrix((n_g, n_p))  # Team B membership (binary)
        y = np.zeros(n_g)
        margins = np.zeros(n_g)
        is_margin = np.zeros(n_g, dtype=bool)
        weights = np.zeros(n_g)

        # 3. Fill matrices row by row
        for g_idx, game in enumerate(self.games):
            t1 = game['t1']
            t2 = game['t2']
            
            # Team 1 (positive direction in skill difference)
            n1 = len(t1)
            if n1 > 0:
                idx_1 = [self._player_id_to_idx[p] for p in t1]
                for col in idx_1:
                    X[g_idx, col] = 1.0 / n1
                    Ma[g_idx, col] = 1.0
            
            # Team 2 (negative direction in skill difference)
            n2 = len(t2)
            if n2 > 0:
                idx_2 = [self._player_id_to_idx[p] for p in t2]
                for col in idx_2:
                    X[g_idx, col] = -1.0 / n2
                    Mb[g_idx, col] = 1.0
            
            # Game outcomes and metadata
            y[g_idx] = game['y']
            # Signed margin: positive if team 1 wins, negative if team 2 wins
            margins[g_idx] = game['m'] if game['y'] == 1 else -game['m']
            is_margin[g_idx] = game['is_margin']
            weights[g_idx] = game['w']

        # 4. Convert to CSR format for fast matrix operations
        self._X = X.tocsr()
        self._Ma = Ma.tocsr()
        self._Mb = Mb.tocsr()
        self._y = y
        self._margins = margins
        self._is_margin = is_margin
        self._game_weights = weights
        self._matrices_dirty = False

    # =========================================================================
    # 3. OPTIMIZATION LOOP (Vectorized)
    # =========================================================================

    def _loss_and_grad(self, theta_vec):
        """
        Compute loss and gradient in one vectorized pass using matrix math.
        
        This is the core of the vectorization: zero Python loops, all linear algebra.
        
        Args:
            theta_vec: Player skills vector (n_players,)
            
        Returns:
            (loss, gradient): scalar loss and gradient vector
        """
        # === A. Pre-computation of Skill Differences and Sigmas ===
        
        # Skill difference: d = X * theta (sparse matrix-vector product)
        d = self._X.dot(theta_vec)
        
        # Player sigma values from game counts
        counts_vec = np.array([self.player_game_counts[pid] for pid in self._idx_to_player_id])
        p_sigmas = self._calc_player_sigmas(counts_vec)
        p_sigmas_sq = p_sigmas ** 2
        
        # Team sigmas: sum of sigma_sq, then average
        sum_sq_a = self._Ma.dot(p_sigmas_sq)  # Sum of sigma_i^2 for team A
        sum_sq_b = self._Mb.dot(p_sigmas_sq)  # Sum of sigma_i^2 for team B
        
        # Number of players per team
        n_a = np.maximum(self._Ma.sum(axis=1).A1, 1)  # .A1 converts matrix to 1D array
        n_b = np.maximum(self._Mb.sum(axis=1).A1, 1)
        
        # Team variance: (sqrt(sum)/N)^2 = sum/N^2
        team_sigma_sq_a = sum_sq_a / (n_a ** 2)
        team_sigma_sq_b = sum_sq_b / (n_b ** 2)
        
        # Effective observation noise
        sigma_eff = self._calc_sigma_eff(team_sigma_sq_a, team_sigma_sq_b)
        sigma_eff_sq = sigma_eff ** 2

        # === B. Vectorized Loss & Gradient Calculation ===
        grad_d = np.zeros_like(d)
        loss = 0.0

        # --- Margin Games (PDF Model) ---
        mask_m = self._is_margin
        if np.any(mask_m):
            dm = d[mask_m]
            sm = self._margins[mask_m]
            se2_m = sigma_eff_sq[mask_m]
            w_m = self._game_weights[mask_m]
            
            # Loss: log(sigma^2) + (d - margin)^2 / sigma^2
            # (The 0.5 factors cancel between Gaussian PDF definition and our setup)
            loss += np.sum(w_m * (0.5 * np.log(se2_m) + (dm - sm) ** 2 / (2 * se2_m)))
            
            # Gradient w.r.t. d: (d - margin) / sigma^2
            grad_d[mask_m] = w_m * (dm - sm) / se2_m

        # --- Outcome Games (Probit CDF Model) ---
        mask_o = ~self._is_margin
        if np.any(mask_o):
            do = d[mask_o]
            yo = self._y[mask_o]
            se_o = sigma_eff[mask_o]
            w_o = self._game_weights[mask_o]
            
            # Standard normal PDF and CDF
            z = do / se_o
            pdf_z = norm.pdf(z)
            cdf_z = ndtr(z)
            
            # === Outcome 1: Team A Wins ===
            mask_win = (yo == 1)
            if np.any(mask_win):
                # Loss = -log(Phi(z))
                cdf_win = np.maximum(cdf_z[mask_win], 1e-10)
                loss -= np.sum(w_o[mask_win] * np.log(cdf_win))
                
                # Gradient = -phi / Phi * 1/sigma
                # (Negated for NLL: dLoss/dd = -pdf/cdf / sigma)
                ratio = pdf_z[mask_win] / cdf_win
                grad_d[mask_o][mask_win] = -w_o[mask_win] * ratio / se_o[mask_win]
            
            # === Outcome -1: Team B Wins ===
            mask_loss = (yo == -1)
            if np.any(mask_loss):
                # Loss = -log(1 - Phi(z)) = -log(Phi(-z))
                cdf_inv = np.maximum(1.0 - cdf_z[mask_loss], 1e-10)
                loss -= np.sum(w_o[mask_loss] * np.log(cdf_inv))
                
                # Gradient = phi / (1 - Phi) * 1/sigma
                # (Higher d hurts team A's loss, so positive gradient)
                ratio = pdf_z[mask_loss] / cdf_inv
                grad_d[mask_o][mask_loss] = w_o[mask_loss] * ratio / se_o[mask_loss]
            
            # === Outcome 0: Draw ===
            # Treat as 50/50, gradient is 0
            # (No additional computation needed)

        # === C. Backpropagate to Theta (Chain Rule) ===
        # d_loss / d_theta = X^T * (d_loss / d_d)
        grad_theta = self._X.T.dot(grad_d)

        # === D. L2 Regularization ===
        loss += self.l2_lambda * np.sum(theta_vec ** 2)
        grad_theta += 2 * self.l2_lambda * theta_vec
        
        return loss, grad_theta

    def _optimize_skills(self):
        """Optimize player skills using L-BFGS-B with vectorized gradients."""
        if not self.games:
            return
        
        if self._matrices_dirty:
            self._build_matrices()
        
        # Initial skills from warm start
        start_theta = np.array([self.theta[p] for p in self._idx_to_player_id])
        
        # Minimize with scipy.optimize (using combined loss+grad function)
        def combined_func(x):
            loss, grad = self._loss_and_grad(x)
            return loss, grad
        
        result = minimize(
            combined_func,
            start_theta,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Update theta with optimized values
        for idx, player_id in enumerate(self._idx_to_player_id):
            self.theta[player_id] = result.x[idx]
        
        self._matrices_dirty = False

    def _ensure_optimized(self):
        """Ensure skills are optimized before prediction or exposure."""
        if self._matrices_dirty:
            self._optimize_skills()

    # =========================================================================
    # 4. PREDICTION (Using Shared Math Kernels)
    # =========================================================================

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        """
        Predict probability that team1 beats team2 using Probit model.
        
        Uses the shared _calc_prob and _calc_sigma_eff kernels.
        
        Args:
            team1_players (list): Player IDs on team 1
            team2_players (list): Player IDs on team 2
            players_on_court: Ignored (kept for API compatibility)
            
        Returns:
            float: Probability that team 1 wins
        """
        self._ensure_optimized()
        
        # Team skills as arithmetic mean
        skill_a = np.mean([self.theta[p] for p in team1_players]) if team1_players else 0.0
        skill_b = np.mean([self.theta[p] for p in team2_players]) if team2_players else 0.0
        
        # Effective noise for this specific game
        sigma_eff = self._game_sigma(team1_players, team2_players)
        
        # Probit probability
        return float(self._calc_prob(skill_a - skill_b, sigma_eff))

    def _game_sigma(self, roster_a, roster_b):
        """
        Compute effective observation std dev for a specific game.
        Used by predict_win_prob and expose.
        
        Formula: sigma_team = sqrt(sum(sigma_i^2)) / n
                 sigma_eff = sqrt(sigma^2 + sigma_team_a^2 + sigma_team_b^2)
        """
        # Sigma for each player
        if roster_a:
            # FIX: Parentheses ensure we square (alpha / sqrt(n+1)), not just sqrt(n+1)
            sigma_sq_sum_a = sum((self.alpha / np.sqrt(self.player_game_counts.get(p, 0) + 1.0)) ** 2 
                                 for p in roster_a)
            sigma_team_a = np.sqrt(sigma_sq_sum_a) / len(roster_a)
        else:
            sigma_team_a = 0.0
        
        if roster_b:
            # FIX: Parentheses ensure we square (alpha / sqrt(n+1)), not just sqrt(n+1)
            sigma_sq_sum_b = sum((self.alpha / np.sqrt(self.player_game_counts.get(p, 0) + 1.0)) ** 2 
                                 for p in roster_b)
            sigma_team_b = np.sqrt(sigma_sq_sum_b) / len(roster_b)
        else:
            sigma_team_b = 0.0
        
        # Quadrature
        return self._calc_sigma_eff(sigma_team_a ** 2, sigma_team_b ** 2)

    # =========================================================================
    # 5. EXPOSURE (Vectorized All-vs-All)
    # =========================================================================

    def expose(self, players):
        """
        Vectorized all-vs-all prediction using shared math kernels.
        
        For each player, computes average win probability against all other
        active players in hypothetical 1v1 matchups. Uses the exact same
        math kernels as training and predict_win_prob.
        
        Args:
            players (list): Player IDs to rate
            
        Returns:
            list: Ratings in [0, max_rating], 2 decimal places
        """
        self._ensure_optimized()
        
        if not players:
            return []
        
        # Filter to active players (those with at least one game)
        active_ids = [p for p in players if self.player_game_counts.get(p, 0) > 0]
        
        # Handle edge cases
        if not active_ids:
            # No active players: everyone gets 0.5 * max_rating
            return [round(0.5 * self.max_rating, 2) for _ in players]
        
        if len(active_ids) == 1:
            # Only one active player: they get max rating, others get 0.5 * max_rating
            ratings = []
            for p in players:
                if p in active_ids:
                    ratings.append(round(self.max_rating, 2))
                else:
                    ratings.append(round(0.5 * self.max_rating, 2))
            return ratings
        
        # === Multi-player case: vectorized computation ===
        
        # Extract skills and game counts for active players
        mus = np.array([self.theta[p] for p in active_ids])
        counts = np.array([self.player_game_counts[p] for p in active_ids])
        n = len(active_ids)
        
        # Compute player sigmas using shared kernel
        sigmas = self._calc_player_sigmas(counts)
        sigmas_sq = sigmas ** 2
        
        # Broadcast skill differences: (N, 1) - (1, N) -> (N, N)
        d_matrix = mus[:, None] - mus[None, :]
        
        # Broadcast sigma^2 for each player to form team variance matrices
        # For 1v1 matches, team variance = player variance
        sigma_sq_matrix_a = np.tile(sigmas_sq[:, None], (1, n))
        sigma_sq_matrix_b = np.tile(sigmas_sq[None, :], (n, 1))
        
        # Compute effective sigma matrix using shared kernel
        sigma_eff_matrix = self._calc_sigma_eff(sigma_sq_matrix_a, sigma_sq_matrix_b)
        
        # Compute probability matrix using shared kernel
        prob_matrix = self._calc_prob(d_matrix, sigma_eff_matrix)
        
        # Average across row (excluding diagonal self-match which is 0.5)
        if n > 1:
            avg_probs = (np.sum(prob_matrix, axis=1) - 0.5) / (n - 1)
        else:
            avg_probs = np.array([1.0])
        
        # Map back to all players
        result_map = {pid: prob * self.max_rating for pid, prob in zip(active_ids, avg_probs)}
        final = []
        for p in players:
            if p in result_map:
                final.append(round(result_map[p], 2))
            else:
                final.append(round(0.5 * self.max_rating, 2))
        
        return final

