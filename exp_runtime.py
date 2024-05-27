"""This script measures the runtime of the different approximators for the NLP Game."""
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from approximators import (
    SHAPIQEstimator,
    PermutationSampling,
    KernelSHAPIQEstimator,
    SvarmIQ,
)
from games import NLPGame

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    budgets = [1000, 2000, 5000, 10000]
    n_runs = 10

    order = 2

    # Setup games ----------------------------------------------------------------------------------
    n_players = 14
    example_sentence = " ".join(["example" for i in range(n_players)])
    game = NLPGame(input_text=example_sentence, set_zero=True)
    assert game.n == n_players
    N = set(range(n_players))

    explainers = {
        "SHAP-IQ": SHAPIQEstimator(N=N, order=order, interaction_type="SII", top_order=False),
        "Permutation": PermutationSampling(
            N=N, order=order, interaction_type="SII", top_order=False
        ),
        "KernelSHAP-IQ": KernelSHAPIQEstimator(
            N=N, order=order, interaction_type="SII", approximator_mode="default"
        ),
        "SVARM-IQ": SvarmIQ(N=N, order=order, interaction_type="SII", sample_strat="ksh"),
    }

    results = []

    pbar = tqdm(total=len(budgets) * n_runs * len(explainers))

    for budget in budgets:
        for iteration in range(n_runs):
            for explainer_name, explainer in explainers.items():
                start = time.time()
                try:
                    explanation = explainer.approximate_with_budget(game.set_call, budget=budget)
                except AttributeError:
                    explanation = explainer.compute_interactions_from_budget(
                        game.set_call, budget=budget
                    )
                end_time = time.time()
                elapsed_time = end_time - start
                results.append(
                    {
                        "budget": budget,
                        "iteration": iteration,
                        "explainer": explainer_name,
                        "runtime": elapsed_time,
                    }
                )
                pbar.update(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join("results_plot", "runtime", "nlp_runtime.csv"), index=False)
