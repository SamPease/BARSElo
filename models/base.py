from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract base class for rating models.

    Subclass this to provide a uniform API used by `calculate.py`:
      - update(game_row, team1_players, team2_players)
      - expose(players)
      - predict_win_prob(team1_players, team2_players, players_on_court=None)
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
