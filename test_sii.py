import numpy as np

from approximators.base import powerset
from approximators import SHAPIQEstimator
from games import LookUpGame
from scipy.special import binom


def _compute_sum_sii(N, game_fun, s):
    n = len(N)
    rslt = 0
    for T in powerset(N, 0, s - 1):
        T_complement = set(N) - set(T)
        t = len(T)
        rslt += (
            1
            / s
            * binom(n - t, s - t - 1)
            * (-1) ** t
            * ((-1) ** s * game_fun(T) + game_fun(T_complement))
        )
    return rslt


if __name__ == "__main__":
    game = LookUpGame(n=14, data_folder="nlp_values", set_zero=True)
    N = set(range(game.n))
    order = 2
    budget = 2**game.n

    estimator = SHAPIQEstimator(N=N, order=order, interaction_type="SII", top_order=False)
    sii_values = estimator.compute_interactions_from_budget(game=game.set_call, budget=budget)

    sum_function = _compute_sum_sii(N, game.set_call, 2)
    sum_manual = np.sum([np.sum(sii_values[2])])

    print(f"Sum function: {sum_function}")
    print(f"Sum manual: {sum_manual}")
