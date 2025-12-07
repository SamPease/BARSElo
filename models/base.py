from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract base class for rating models.

    Subclass this to provide a uniform API used by `calculate.py`:
      - update(game_row, team1_players, team2_players)
      - expose(players)
      - predict_win_prob(team1_players, team2_players, players_on_court=None)
      - load_state(filepath, all_players)
    """

    @abstractmethod
    def update(self, game_row, team1_players, team2_players):
        raise NotImplementedError()

    @abstractmethod
    def expose(self, players):
        raise NotImplementedError()

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        """Optional: return probability team1 beats team2.

        Subclasses may override. Default raises NotImplementedError to signal
        that the model does not provide a probability estimate.
        """
        raise NotImplementedError()

    def load_state(self, filepath, all_players):
        """Load model state from a previous results CSV file.

        Args:
            filepath (str): Path to the results CSV file
            all_players (list): List of all player IDs in order

        Returns:
            bool: True if state was successfully loaded, False otherwise

        Default implementation returns False (no resume support).
        Models that support resuming should override this method.
        """
        return False
