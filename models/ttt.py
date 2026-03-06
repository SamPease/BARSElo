"""
TrueSkill Through Time Model with temporal skill evolution.

This model uses the trueskillthroughtime package to track player skill
evolution over time, naturally capturing skill changes and learning curves.

Key features:
- Temporal modeling of skill evolution (gamma parameter)
- Convergence algorithm for better estimates across the entire history
- Per-player uncertainty estimates that account for time
- Multi-team game support
"""

import numpy as np
import math
from scipy.stats import norm
from data.data_loader import parse_time_maybe

try:
    import trueskillthroughtime as ttt
except ImportError:
    raise ImportError(
        "trueskillthroughtime package required. Install with: "
        "python3 -m pip install trueskillthroughtime"
    )

from .base import Model


# Default hyperparameters for TrueSkill Through Time
DEFAULT_MU = 25.0                   # Initial skill mean
DEFAULT_SIGMA = 8.3333333333333                 # Initial skill uncertainty (increased for stability)
DEFAULT_BETA = 1.0                  # Performance variability
DEFAULT_GAMMA = 0.02                # Skill evolution rate (reduced for stability)
DEFAULT_P_DRAW = 0.06                       # Probability of draw

# Exposure parameters
DEFAULT_DISPLAY_CENTER = 1000.0     # Center for exposed ratings
DEFAULT_DISPLAY_SPREAD = 200.0      # Spread for exposed ratings


class TTTModel(Model):
    """
    TrueSkill Through Time Model.
    
    Uses temporal Bayesian modeling to track skill evolution over time.
    Each game update is stored with its timestamp, and convergence is
    computed lazily when predictions or exposure are needed.
    
    Key attributes:
    - composition: List of game team compositions (e.g., [[[p1, p2], [p3, p4]], ...])
    - results: List of game results (e.g., [[1, 0], [1, 0], ...])
    - times: List of game timestamps (float or None)
    - priors: Dictionary of Player objects for warm-start
    """

    def __init__(self, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, beta=DEFAULT_BETA,
                 gamma=DEFAULT_GAMMA, p_draw=DEFAULT_P_DRAW,
                 display_center=DEFAULT_DISPLAY_CENTER,
                 display_spread=DEFAULT_DISPLAY_SPREAD,
                 warm_start=None,
                 convergence_iterations=10):
        """
        Initialize TTT model.
        
        Args:
            mu: Initial skill mean (default 25.0)
            sigma: Initial skill uncertainty (default 8.3333333333333)
            beta: Performance variability (default 1.0)
            gamma: Skill evolution rate per day (default 0.02)
                   0 = static skills, higher = faster evolution
            p_draw: Probability of draw (default 0.06), uniform for all games
            display_center: Center for exposed ratings (default 1000.0)
            display_spread: Spread for exposed ratings (default 200.0)
            warm_start: Dictionary of player_id -> Player for warm-start (optional)
            convergence_iterations: Number of forward-backward iterations for
                                    ttt.History convergence (default 10).
        """
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.gamma = gamma
        self.p_draw = p_draw
        self.display_center = display_center
        self.display_spread = display_spread
        self.convergence_iterations = convergence_iterations

        # Game data storage
        self.composition = []        # List of team compositions
        self.results = []            # List of game results
        self.times_seconds = []      # List of game timestamps (seconds)
        self.players_on_court = []   # Active players target from game rows (typically 6 or 8)
        self.priors = warm_start.copy() if warm_start else {}  # Dict of player_id -> Player-like
        
        # Cache for history and convergence
        self._history = None
        self._history_dirty = True
        self._player_ratings = {}    # Cache of latest player ratings (from convergence OR forward pass)
        self._learning_curves = {}

        # Forward-pass running ratings (updated incrementally in update())
        self._forward_ratings = {}   # player_id -> (mu, sigma)
        self._forward_last_time = {} # player_id -> last game time in days
        
        # Incremental valid-game cache (appended in update order)
        self._valid_composition = []
        self._valid_results = []
        self._valid_times_days = []
        self._valid_weights = []
        self._source_games_validated = 0  # How many source games have been filtered
        
        # Time tracking
        self._game_counter = 0       # Counter for games without explicit time
        self._day_to_time_label = {}

        # Convergence throttle: skip reconvergence when very few new games
        # have been added since the last successful convergence.
        self._games_at_last_converge = 0
        self._min_new_games_for_reconverge = 10

    def _seconds_to_days(self, seconds):
        """Convert unix seconds to day units expected by gamma-per-day."""
        return float(seconds) / 86400.0

    def _default_gaussian(self):
        return ttt.Gaussian(self.mu, self.sigma)

    def _default_player(self):
        # Explicitly bind mu/sigma/beta/gamma so package defaults are never used implicitly.
        return ttt.Player(self._default_gaussian(), self.beta, self.gamma)

    def _coerce_prior_player(self, prior_obj):
        """Normalize warm-start values into ttt.Player while preserving signal when possible."""
        if isinstance(prior_obj, ttt.Player):
            return prior_obj

        if isinstance(prior_obj, (tuple, list)) and len(prior_obj) >= 2:
            try:
                mu_val = float(prior_obj[0])
                sigma_val = max(1e-8, float(prior_obj[1]))
                return ttt.Player(ttt.Gaussian(mu_val, sigma_val), self.beta, self.gamma)
            except Exception:
                return self._default_player()

        mu_val = getattr(prior_obj, 'mu', None)
        sigma_val = getattr(prior_obj, 'sigma', None)
        if mu_val is not None and sigma_val is not None:
            try:
                return ttt.Player(ttt.Gaussian(float(mu_val), max(1e-8, float(sigma_val))), self.beta, self.gamma)
            except Exception:
                return self._default_player()

        return self._default_player()

    def _ensure_priors_for_all_seen_players(self, compositions):
        """Ensure all players in the current history have a valid prior player object."""
        seen = set()
        for game in compositions:
            for team in game:
                for player_id in team:
                    seen.add(player_id)

        normalized = {}
        for player_id in seen:
            if player_id in self.priors:
                normalized[player_id] = self._coerce_prior_player(self.priors[player_id])
            else:
                normalized[player_id] = self._default_player()

        self.priors = normalized

    def _team_weights_for_game(self, team_players, active_players_target):
        """Assign per-player weights so oversized rosters do not over-contribute in one game."""
        n_players = len(team_players)
        if n_players == 0:
            return []

        try:
            target = max(1, int(active_players_target))
        except Exception:
            target = 8

        if n_players > target:
            weight = float(target) / float(n_players)
        else:
            weight = 1.0

        return [weight for _ in team_players]

    # =========================================================================
    # 1. CORE UPDATES
    # =========================================================================

    def update(self, game_row, team1_players, team2_players):
        """
        Store game data.
        
        Args:
            game_row: Game data row containing outcome and timing info
            team1_players (list): List of player IDs on team 1
            team2_players (list): List of player IDs on team 2
        """
        try:
            # Parse game row
            time, team1_id, s1, team2_id, s2, outcome_flag, tourney_flag, players_field = (game_row + [None]*8)[:8]
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

        # Determine result (TTT uses ranking: 0 = highest, 1 = second, etc.)
        # TTT expects [0, 1] for team1 wins, [1, 0] for team2 wins, [0, 0] for tie
        if s1 > s2:
            result = [0, 1]  # Team 1 beats Team 2
        elif s2 > s1:
            result = [1, 0]  # Team 2 beats Team 1
        else:
            result = [0, 0]  # Draw/Tie
        
        # Parse timestamp in seconds. We convert to days only when building History.
        game_time_seconds = None
        time_label = str(time) if time is not None and str(time).strip() else ''
        if time is not None:
            try:
                parsed = parse_time_maybe(time)
                if parsed is not None:
                    game_time_seconds = parsed.timestamp()
            except Exception:
                pass
        
        # Use deterministic one-day spacing if no parseable timestamp is available.
        if game_time_seconds is None:
            game_time_seconds = float(self._game_counter * 86400)
            if not time_label:
                time_label = f"TTT_DAY_{self._game_counter}"

        self._game_counter += 1

        day_value = self._seconds_to_days(game_time_seconds)
        if time_label:
            self._day_to_time_label.setdefault(day_value, time_label)

        try:
            players_total = int(players_field) if players_field and str(players_field).strip() else 8
        except Exception:
            players_total = 8

        # Store game
        team1_list = list(team1_players)
        team2_list = list(team2_players)
        
        self.composition.append([team1_list, team2_list])
        self.results.append(result)
        self.times_seconds.append(game_time_seconds)
        self.players_on_court.append(players_total)
        
        # Mark history as needing rebuild (for expose / get_all_historical_ratings)
        self._history_dirty = True

        # --- Incremental forward-pass update for fast predict_win_prob ------
        if team1_list and team2_list:
            self._forward_update_game(
                team1_list, team2_list, result, day_value, players_total, float(self.p_draw)
            )

    def _forward_update_game(self, team1, team2, result, time_days, players_total, p_draw_game):
        """Run a single-game TrueSkill update to maintain running ratings."""
        default_mu = self.mu
        default_sigma = self.sigma

        teams_players = [team1, team2]
        ttt_teams = []
        for team in teams_players:
            team_objs = []
            for p in team:
                mu, sigma = self._forward_ratings.get(p, (default_mu, default_sigma))
                # Apply time-based forgetting
                lt = self._forward_last_time.get(p)
                elapsed = 0.0 if lt is None else max(0.0, time_days - lt)
                prior = ttt.Gaussian(mu, sigma).forget(self.gamma, elapsed)
                prior = ttt.Gaussian(prior.mu, max(1e-8, prior.sigma))
                team_objs.append(ttt.Player(prior, self.beta, self.gamma))
            ttt_teams.append(team_objs)

        weights = [
            self._team_weights_for_game(team1, players_total),
            self._team_weights_for_game(team2, players_total),
        ]

        try:
            g = ttt.Game(ttt_teams, result=result, p_draw=p_draw_game, weights=weights)
            posteriors = g.posteriors()
            for t_i, team in enumerate(teams_players):
                for p_i, p in enumerate(team):
                    post = posteriors[t_i][p_i]
                    self._forward_ratings[p] = (post.mu, max(1e-8, post.sigma))
                    self._forward_last_time[p] = time_days
        except Exception:
            # If single-game update fails, just skip (ratings stay at prior)
            for team in teams_players:
                for p in team:
                    if p not in self._forward_ratings:
                        self._forward_ratings[p] = (default_mu, default_sigma)
                    self._forward_last_time.setdefault(p, time_days)

    # =========================================================================
    # 2. LAZY HISTORY COMPUTATION
    # =========================================================================

    def _build_history(self):
        """Build converged History from accumulated games.

        Forces convergence regardless of the min-games threshold so that
        expose() always produces fully converged final ratings.
        """
        # Reset threshold counter to force convergence even if few games
        # were added since the last call.
        self._games_at_last_converge = 0
        self.converge()

    # =========================================================================
    # 3. CONVERGENCE
    # =========================================================================

    def converge(self):
        """Run convergence to update player ratings from full history.

        Uses ttt.History with a single uniform p_draw for speed.  The
        library's C-level convergence is much faster than the Python-level
        per-batch loop.  If convergence fails numerically (math domain
        error in the library), the last successful converged ratings are
        kept and forward-pass ratings are used for any new players.

        Call this once per "checkpoint" in the training loop (after a
        batch of ``update()`` calls and before ``predict_win_prob()``
        calls) so that predictions benefit from backward-smoothed TTT
        ratings rather than forward-only TrueSkill estimates.

        Skips convergence when fewer than ``_min_new_games_for_reconverge``
        games have been added since the last successful convergence, to
        avoid expensive re-solves for tiny increments.
        """
        if not self._history_dirty and self._player_ratings:
            return  # Already converged on all current games.

        n_games = len(self.composition)
        if (n_games - self._games_at_last_converge) < self._min_new_games_for_reconverge:
            return  # Not enough new games to justify re-convergence.

        # Ensure valid game lists are up-to-date.
        for i in range(self._source_games_validated, len(self.composition)):
            comp = self.composition[i]
            result = self.results[i]
            seconds = self.times_seconds[i]
            players_total = self.players_on_court[i]
            if len(comp) == 2 and len(comp[0]) > 0 and len(comp[1]) > 0:
                self._valid_composition.append(comp)
                self._valid_results.append(result)
                self._valid_times_days.append(self._seconds_to_days(seconds))
                weights_game = [
                    self._team_weights_for_game(comp[0], players_total),
                    self._team_weights_for_game(comp[1], players_total),
                ]
                self._valid_weights.append(weights_game)
        self._source_games_validated = len(self.composition)

        if not self._valid_composition:
            return

        try:
            h = ttt.History(
                composition=self._valid_composition,
                results=self._valid_results,
                times=self._valid_times_days,
                priors=self.priors,
                sigma=self.sigma,
                beta=self.beta,
                gamma=self.gamma,
                p_draw=self.p_draw,
                weights=self._valid_weights,
            )

            # Timeout-protected convergence: the library sometimes hangs
            # during forward/backward passes due to numerical instability.
            import signal
            _convergence_timed_out = False

            def _alarm_handler(signum, frame):
                nonlocal _convergence_timed_out
                _convergence_timed_out = True
                raise TimeoutError("ttt.History.convergence timed out")

            timeout_secs = max(60, self.convergence_iterations * 5)
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_secs)
            try:
                h.convergence(iterations=self.convergence_iterations, verbose=False)
            except TimeoutError:
                pass
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            if _convergence_timed_out:
                # Convergence hung — keep forward-pass ratings.
                return

            lc = h.learning_curves()
            self._learning_curves = lc
            self._player_ratings = {}
            for pid, curve in lc.items():
                if curve:
                    self._player_ratings[pid] = (curve[-1][1].mu, curve[-1][1].sigma)
            self._history = h
            self._history_dirty = False
            self._games_at_last_converge = n_games
        except (ValueError, RuntimeError, ZeroDivisionError, ArithmeticError):
            # Numerical instability — keep last converged ratings,
            # supplement with forward-pass for any missing players.
            pass

    def converge_and_freeze(self):
        """Converge once on ALL games and build a time-indexed snapshot.

        After calling this, ``predict_win_prob()`` will automatically use
        the converged learning curves to look up each player's rating at
        the snapshot time set via ``set_predict_time()``.  This gives
        true backward-smoothed TTT estimates without re-converging at
        every checkpoint.

        Build once, then repeatedly call ``set_predict_time(t)`` and
        ``predict_win_prob(...)`` for O(1) predictions.
        """
        import time as _time
        _t0 = _time.time()
        self.converge()
        _elapsed = _time.time() - _t0
        if _elapsed > 2.0:
            print(f'[TTT] converge_and_freeze: converge() took {_elapsed:.2f}s for {len(self.composition)} games')
        if not self._learning_curves:
            return

        # Pre-sort each player's learning curve for efficient bisect lookup.
        import bisect
        self._sorted_curves = {}
        for pid, curve in self._learning_curves.items():
            if curve:
                times_sorted = [t for t, _g in curve]
                gaussians_sorted = [(_g.mu, _g.sigma) for _t, _g in curve]
                self._sorted_curves[pid] = (times_sorted, gaussians_sorted)

        self._predict_time = None  # Use latest ratings until set_predict_time is called.

    def set_predict_time(self, time_days):
        """Set the time cutoff for predict_win_prob when using converged snapshots.

        After ``converge_and_freeze()``, calling this method configures
        ``predict_win_prob()`` to return the player's converged rating
        *at or before* the given time (in days since epoch).

        Args:
            time_days: Time in days (same units as ``_seconds_to_days()``).
                       Use ``None`` to reset to latest ratings.
        """
        self._predict_time = time_days

    def _rating_at_time(self, player_id, time_days):
        """Look up a player's converged rating at or before a given time.

        Uses bisect on the pre-sorted learning curve for O(log N) lookup.
        Falls back to forward-pass or default if the player has no curve
        or no observations before the requested time.
        """
        import bisect
        curves = getattr(self, '_sorted_curves', None)
        if curves and player_id in curves:
            ts, gs = curves[player_id]
            idx = bisect.bisect_right(ts, time_days) - 1
            if idx >= 0:
                return gs[idx]
        # No converged data at this time — fall back
        if player_id in self._forward_ratings:
            return self._forward_ratings[player_id]
        return (self.mu, self.sigma)

    # =========================================================================
    # 4. EXPOSURE AND PREDICTION
    # =========================================================================

    def expose(self, players):
        """
        Return exposed ratings for a list of players.
        
        Exposed rating uses the conservative TrueSkill estimate: mu - 3*sigma,
        then maps it into display space with center/spread.
        
        Args:
            players (list): List of player IDs
            
        Returns:
            list: Exposed ratings (rounded to 2 decimals)
        """
        self._build_history()
        
        exposed = []
        baseline_conservative = self.mu - (3.0 * self.sigma)
        for p in players:
            if p in self._player_ratings:
                mu, sigma = self._player_ratings[p]
                conservative = mu - (3.0 * sigma)
            else:
                conservative = baseline_conservative

            # Conservative rating scaled for visualization.
            exposed_rating = self.display_center + self.display_spread * conservative
            
            exposed.append(round(exposed_rating, 2))
        
        return exposed

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        """
        Predict probability that team1 beats team2.
        
        Uses team-size normalized composite-team probability:
        Phi((mu_avg1 - mu_avg2) / sqrt((2*beta^2)/N + sigma2_avg1 + sigma2_avg2)).

        When ``converge_and_freeze()`` + ``set_predict_time()`` have been
        called, uses converged learning-curve ratings at the specified time.
        Otherwise uses the most recent converged (or forward-pass) ratings.
        
        Args:
            team1_players (list): Player IDs on team 1
            team2_players (list): Player IDs on team 2
            players_on_court (int): Optional; number of players total (used for weighting)
            
        Returns:
            float: Probability in [0, 1] that team1 wins
        """
        if not team1_players or not team2_players:
            return 0.5

        default_mu = self.mu
        default_sigma = self.sigma

        # Time-aware lookup from converged learning curves
        predict_time = getattr(self, '_predict_time', None)
        sorted_curves = getattr(self, '_sorted_curves', None)

        if predict_time is not None and sorted_curves is not None:
            def _get_rating(p):
                return self._rating_at_time(p, predict_time)
        else:
            # Fall back: converged final ratings > forward-pass > default
            def _get_rating(p):
                if p in self._player_ratings:
                    return self._player_ratings[p]
                if p in self._forward_ratings:
                    return self._forward_ratings[p]
                return (default_mu, default_sigma)

        team1_stats = [_get_rating(p) for p in team1_players]
        team2_stats = [_get_rating(p) for p in team2_players]

        mu_avg_1 = sum(mu for mu, _ in team1_stats) / float(len(team1_stats))
        mu_avg_2 = sum(mu for mu, _ in team2_stats) / float(len(team2_stats))

        sigma2_avg_1 = sum((sigma ** 2) for _, sigma in team1_stats) / float(len(team1_stats))
        sigma2_avg_2 = sum((sigma ** 2) for _, sigma in team2_stats) / float(len(team2_stats))

        if players_on_court is None:
            n_active = 8
        else:
            n_active = max(1, int(players_on_court))

        denom_var = ((2.0 * (self.beta ** 2)) / float(n_active)) + sigma2_avg_1 + sigma2_avg_2
        if denom_var <= 0.0:
            return 0.5

        z = (mu_avg_1 - mu_avg_2) / np.sqrt(denom_var)
        prob = norm.cdf(z)

        return float(prob)

    # =========================================================================
    # 4. HISTORICAL RATINGS (All Time Steps)
    # =========================================================================

    def get_all_historical_ratings(self, all_players):
        """
        Return historical ratings for all time steps (learning curves).
        
        TTT updates all past ratings when a new game is added, so it's more
        efficient to compute all historical ratings at once rather than
        incrementally per game.
        
        Args:
            all_players (list): List of all player IDs in order
            
        Returns:
            dict: {time_str: [ratings...]} where time_str is the timestamp
                  and ratings are exposed values in the order of all_players.
                  Returns {} if no games have been added yet.
        """
        self._build_history()

        if self._history is None and not self._learning_curves:
            return {}
        
        learning_curves = self._learning_curves if self._learning_curves else self._history.learning_curves()

        # Collect all unique times across all players
        all_times = set()
        for player_id, curve in learning_curves.items():
            for time_point, gaussian in curve:
                all_times.add(time_point)

        if not all_times:
            return {}

        # For each time, gather ratings for all players
        result = {}
        for t in sorted(all_times):
            ratings = []
            for player_id in all_players:
                if player_id in learning_curves:
                    # Find the rating at or before time t
                    curve = learning_curves[player_id]
                    matching_gaussian = None
                    for time_point, gaussian in curve:
                        if time_point <= t:
                            matching_gaussian = gaussian
                        else:
                            break

                    if matching_gaussian is not None:
                        mu, sigma = matching_gaussian.mu, matching_gaussian.sigma
                        conservative = mu - (3.0 * sigma)
                        exposed_rating = self.display_center + self.display_spread * conservative
                        ratings.append(round(exposed_rating, 2))
                    else:
                        # No rating before time t
                        baseline = self.mu - (3.0 * self.sigma)
                        ratings.append(round(self.display_center + self.display_spread * baseline, 2))
                else:
                    # Player has no learning curve (never played)
                    baseline = self.mu - (3.0 * self.sigma)
                    ratings.append(round(self.display_center + self.display_spread * baseline, 2))

            # Store ratings for this time.
            time_str = self._format_output_time(t)
            result[time_str] = ratings

        return result

    def _format_output_time(self, time_in_days):
        """Prefer original labels when available; otherwise emit stable day numeric key."""
        if isinstance(time_in_days, (int, float)):
            key = float(time_in_days)
            if key in self._day_to_time_label:
                return self._day_to_time_label[key]
            return f"{key:.6f}"
        return str(time_in_days)

    # =========================================================================
    # 5. OPTIONAL STATE MANAGEMENT
    # =========================================================================

    def load_state(self, filepath, all_players):
        """
        Load model state from a previous results CSV.
        
        Currently not implemented for TTT models since full history
        is needed for convergence.
        
        Args:
            filepath (str): Path to results file
            all_players (list): List of all player IDs
            
        Returns:
            bool: False (no resume support)
        """
        return False
