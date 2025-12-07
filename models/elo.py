from collections import defaultdict
from .base import Model


DEFAULT_INITIAL_ELO = 1000.0
DEFAULT_K_FACTOR = 346.44640893667486
DEFAULT_TOURNAMENT_MULTIPLIER = 1.5140799771798972

class EloModel(Model):
    """Elo model that owns its internal state and exposes a uniform API.

    - Constructor accepts hyperparameters but defaults are provided at module level.
    - `update(game_row, team1_players, team2_players)` updates internal elos.
    - `expose(players)` returns current exposed elos for a list of players.
    """

    def __init__(self, initial_elo=DEFAULT_INITIAL_ELO, K=DEFAULT_K_FACTOR, tournament_multiplier=DEFAULT_TOURNAMENT_MULTIPLIER):
        self.initial_elo = float(initial_elo)
        self.K = float(K)
        self.tournament_multiplier = float(tournament_multiplier)
        self.elos = defaultdict(lambda: self.initial_elo)

    def _average_elo(self, players):
        return sum(self.elos[p] for p in players) / len(players) if players else self.initial_elo

    def _mov_factor(self, goal_diff):
        if goal_diff <= 0:
            return 1
        elif goal_diff == 1:
            return 1
        elif goal_diff == 2:
            return 1.5
        else:
            return (11 + goal_diff) / 8

    def update(self, game_row, team1_players, team2_players):
        """Update internal elos from a game row.

        `game_row` is expected to be an iterable where indices map to:
        [time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field]
        Some inputs (like Elo) may include only 7 columns; we handle both.
        """
        # unpack safely
        try:
            time, t1, s1, t2, s2, outcome_flag, tourney_flag = game_row[:7]
            players_field = game_row[7] if len(game_row) > 7 else None
        except Exception:
            return

        # Outcome handling consistent with previous Elo behavior
        # outcome_flag triggers using running average margin elsewhere; here we
        # use the margins provided directly.
        if outcome_flag.strip():
            try:
                s1, s2 = int(s1), int(s2)
            except Exception:
                s1, s2 = 1, 0
            # convert to (margin, 0) representation
            if s1 > s2:
                s1, s2 = abs(s1 - s2), 0
            elif s2 > s1:
                s1, s2 = 0, abs(s2 - s1)
            else:
                s1 = s2 = 0
        else:
            try:
                s1, s2 = int(s1), int(s2)
            except Exception:
                s1, s2 = 0, 0

        margin = abs(s1 - s2)

        # compute expected scores
        avg1 = self._average_elo(team1_players)
        avg2 = self._average_elo(team2_players)
        expected1 = 1 / (1 + 10 ** ((avg2 - avg1) / 400)) if team1_players and team2_players else 0.5
        expected2 = 1 - expected1

        if s1 > s2:
            actual1, actual2 = 1, 0
        elif s1 < s2:
            actual1, actual2 = 0, 1
        else:
            actual1, actual2 = 0.5, 0.5

        G = self._mov_factor(margin)
        multiplier = self.tournament_multiplier if tourney_flag.strip() else 1.0
        change1 = self.K * G * (actual1 - expected1) * multiplier
        change2 = self.K * G * (actual2 - expected2) * multiplier

        if team1_players:
            for p in team1_players:
                self.elos[p] += change1 / len(team1_players)
        if team2_players:
            for p in team2_players:
                self.elos[p] += change2 / len(team2_players)

        return self.elos

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        avg1 = self._average_elo(team1_players)
        avg2 = self._average_elo(team2_players)
        try:
            prob = 1 / (1 + 10 ** ((avg2 - avg1) / 400))
        except Exception:
            prob = 0.5
        return prob

    def expose(self, players):
        return [round(self.elos[p], 2) for p in players]

    def load_state(self, filepath, all_players):
        """Load Elo ratings from the last row of a results CSV.

        Args:
            filepath (str): Path to the elo_results.csv file
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

            # Load ratings from last row (skip Time column)
            for i, player in enumerate(all_players):
                # Find player in header (skip index 0 which is 'Time')
                try:
                    player_idx = header.index(player)
                    rating = float(last_row[player_idx])
                    self.elos[player] = rating
                except (ValueError, IndexError):
                    # Player not found or invalid rating, keep default
                    continue

            return True

        except Exception:
            return False
