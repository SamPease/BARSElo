from collections import defaultdict
import trueskill as ts
from .base import Model

# Default TrueSkill MOV hyperparameters
DEFAULT_TS_MU = 1000.0
DEFAULT_TS_SIGMA = None
DEFAULT_TS_BETA = None  # when None model will set MOV-specific beta
DEFAULT_TS_TAU = None
DEFAULT_TS_DRAW_PROBABILITY = 0.123


class TrueSkillMovModel(Model):
    """TrueSkill MOV model with uniform `update` API.

    - Defaults at module level. Constructor accepts overrides.
    - `update(game_row, team1_players, team2_players)` applies repeated rating updates.
    """

    def __init__(self, mu=DEFAULT_TS_MU, sigma=DEFAULT_TS_SIGMA, beta=DEFAULT_TS_BETA, tau=DEFAULT_TS_TAU, draw_probability=DEFAULT_TS_DRAW_PROBABILITY):
        self.mu = mu
        if sigma is None:
            sigma = mu / 3.0
        if beta is None:
            beta = (sigma / 2.0) * (5 ** 0.5)
        if tau is None:
            tau = sigma / 100.0
        ts.setup(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability)
        self.ratings = defaultdict(lambda: ts.Rating(mu=mu, sigma=sigma))

    def _team_weights(self, team_players, players_total):
        size = len(team_players)
        if size == 0:
            return []
        mult = max(1.0, float(players_total) / float(size))
        return [mult for _ in team_players]

    def update(self, game_row, team1_players, team2_players):
        try:
            time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = (game_row + [None]*8)[:8]
        except Exception:
            return

        try:
            s1 = max(0, int(s1))
        except Exception:
            s1 = 0
        try:
            s2 = max(0, int(s2))
        except Exception:
            s2 = 0

        team1_ratings = [self.ratings[p] for p in team1_players]
        team2_ratings = [self.ratings[p] for p in team2_players]
        if not team1_ratings or not team2_ratings:
            return

        try:
            players_total = int(players_field) if players_field and str(players_field).strip() else 8
        except Exception:
            players_total = 8

        weights = [self._team_weights(team1_players, players_total), self._team_weights(team2_players, players_total)]

        # Apply s1 times where team1 beats team2
        for _ in range(s1):
            team1_ratings = [self.ratings[p] for p in team1_players]
            team2_ratings = [self.ratings[p] for p in team2_players]
            rated = ts.rate([team1_ratings, team2_ratings], ranks=[0, 1], weights=weights)
            for team_players, rated_team in ((team1_players, rated[0]), (team2_players, rated[1])):
                for p, new_rating in zip(team_players, rated_team):
                    self.ratings[p] = new_rating

        # Apply s2 times where team2 beats team1
        for _ in range(s2):
            team1_ratings = [self.ratings[p] for p in team1_players]
            team2_ratings = [self.ratings[p] for p in team2_players]
            rated = ts.rate([team1_ratings, team2_ratings], ranks=[1, 0], weights=weights)
            for team_players, rated_team in ((team1_players, rated[0]), (team2_players, rated[1])):
                for p, new_rating in zip(team_players, rated_team):
                    self.ratings[p] = new_rating

    def expose(self, players):
        return [round(ts.expose(self.ratings[p]), 2) for p in players]

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):
        """Estimate win probability for team1 vs team2 using weighted TrueSkill formula.

        `players_on_court` defaults to 8 when not provided. The parameter is
        used to weight sigma contributions and the denom term as in the
        TrueSkill approximation.
        """
        import itertools, math
        env = ts.global_env()
        beta = env.beta

        if players_on_court is None:
            players_on_court = 8

        def team_weight(team_players):
            size = len(team_players)
            if size == 0:
                return []
            mult = max(1.0, float(players_on_court) / float(size))
            return [mult for _ in team_players]

        team1_ratings = [self.ratings[p] for p in team1_players]
        team2_ratings = [self.ratings[p] for p in team2_players]

        if not team1_ratings or not team2_ratings:
            return 0.5

        delta_mu = sum(r.mu for r in team1_ratings) - sum(r.mu for r in team2_ratings)

        w1 = team_weight(team1_players)
        w2 = team_weight(team2_players)

        sum_sigma = 0.0
        for r, w in zip(team1_ratings, w1):
            sum_sigma += (w * r.sigma) ** 2
        for r, w in zip(team2_ratings, w2):
            sum_sigma += (w * r.sigma) ** 2

        size = float(players_on_court * 2)
        denom = math.sqrt(size * (beta ** 2) + sum_sigma)
        if denom <= 0:
            return 0.5

        try:
            return env.cdf(delta_mu / denom)
        except Exception:
            return 0.5
