from collections import defaultdict
import trueskill as ts
from .base import Model
import itertools, math

# Default trueskill hyperparameters
DEFAULT_TS_MU = 1000.0
DEFAULT_TS_SIGMA = None  # model will default to mu/3 when None
DEFAULT_TS_BETA = None
DEFAULT_TS_TAU = None
DEFAULT_TS_DRAW_PROBABILITY = 0.123

# TrueSkill models cannot resume because Rating objects are not easily serializable
CAN_RESUME = False


class TrueSkillModel(Model):
    """Uniform TrueSkill model API.

    - Defaults are defined at module level; constructor accepts overrides.
    - `update(game_row, team1_players, team2_players)` updates internal ratings.
    - `expose(players)` returns exposed rating means for players.
    """

    def __init__(self, mu=DEFAULT_TS_MU, sigma=DEFAULT_TS_SIGMA, beta=DEFAULT_TS_BETA, tau=DEFAULT_TS_TAU, draw_probability=DEFAULT_TS_DRAW_PROBABILITY):
        self.mu = mu
        if sigma is None:
            sigma = mu / 3.0
        if beta is None:
            beta = sigma / 2.0
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
        # game_row layout: time,t1,s1,t2,s2,outcome_flag,tourney_flag,players_field
        try:
            time, t1, s1, t2, s2, outcome_flag, tourney_flag, players_field = (game_row + [None]*8)[:8]
        except Exception:
            return

        try:
            s1 = int(s1)
            s2 = int(s2)
        except Exception:
            s1, s2 = 0, 0

        if s1 > s2:
            ranks = [0, 1]
        elif s1 < s2:
            ranks = [1, 0]
        else:
            ranks = [0, 0]

        team1_ratings = [self.ratings[p] for p in team1_players]
        team2_ratings = [self.ratings[p] for p in team2_players]
        if not team1_ratings or not team2_ratings:
            return

        try:
            players_total = int(players_field) if players_field and str(players_field).strip() else 8
        except Exception:
            players_total = 8

        weights = [self._team_weights(team1_players, players_total), self._team_weights(team2_players, players_total)]
        rated_teams = ts.rate([team1_ratings, team2_ratings], ranks=ranks, weights=weights)

        for team_players, rated_team in ((team1_players, rated_teams[0]), (team2_players, rated_teams[1])):
            for p, new_rating in zip(team_players, rated_team):
                self.ratings[p] = new_rating

    def expose(self, players):
        return [round(ts.expose(self.ratings[p]), 2) for p in players]

    def predict_win_prob(self, team1_players, team2_players, players_on_court=None):

        env = ts.global_env()
        beta = env.beta

        if players_on_court is None:
            players_on_court = 8

        # compute weights (as used in rating updates)
        def team_weight(team_players):
            size = len(team_players)
            if size == 0:
                return []
            mult = max(1.0, float(players_on_court) / float(size))
            return [mult for _ in team_players]

        # collect ratings
        team1_ratings = [self.ratings[p] for p in team1_players]
        team2_ratings = [self.ratings[p] for p in team2_players]

        if not team1_ratings or not team2_ratings:
            return 0.5

        # apply the same weighting used during updates to mus and sigmas
        w1 = team_weight(team1_players)
        w2 = team_weight(team2_players)

        # weighted difference in exposed means
        weighted_mu_1 = sum((w * r.mu) for r, w in zip(team1_ratings, w1))
        weighted_mu_2 = sum((w * r.mu) for r, w in zip(team2_ratings, w2))
        delta_mu = weighted_mu_1 - weighted_mu_2

        # contribution to variance from per-player performance randomness (beta)
        sum_beta_var = 0.0
        for w in w1:
            sum_beta_var += (w * beta) ** 2
        for w in w2:
            sum_beta_var += (w * beta) ** 2

        # Sum sigma^2 scaled by weight^2 (uncertainty in skill estimates)
        sum_sigma = 0.0
        for r, w in zip(team1_ratings, w1):
            sum_sigma += (w * r.sigma) ** 2
        for r, w in zip(team2_ratings, w2):
            sum_sigma += (w * r.sigma) ** 2

        denom_var = sum_beta_var + sum_sigma
        if denom_var <= 0:
            return 0.5

        try:
            return env.cdf(delta_mu / math.sqrt(denom_var))
        except Exception:
            return 0.5