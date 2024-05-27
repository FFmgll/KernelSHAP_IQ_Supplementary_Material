"""This script runs the experiment for the SOUMs dataset."""
import copy
import os
import time

import numpy as np
import pandas as pd

from approximators import (
    SHAPIQEstimator,
    PermutationSampling,
    KernelSHAPIQEstimator,
    SvarmIQ,
)
from experiment import run_approx_experiment
from games import ParameterizedSparseLinearModel
from utils_experiment import get_gt_values_for_game

import warnings

warnings.filterwarnings("ignore")


def run_experiment(save_path_addendum="", inner_save_folder: str = None):
    """Runs the experiment for the given parameters."""

    # Setup games ----------------------------------------------------------------------------------
    all_gt_values = {}

    # initialize games
    game_list = []

    game = ParameterizedSparseLinearModel(
        n=DATA_N,
        weighting_scheme=SOUM_WEIGHTING_SCHEME,
        n_interactions=SOUM_N_INTERACTIONS,
        max_interaction_size=SOUM_MAX_INTERACTION_SIZE,
        min_interaction_size=SOUM_MIN_INTERACTION_SIZE,
        n_non_important_features=SOUM_N_NON_IMPORTANT_FEATURES,
    )
    game_list.append(game)

    # get number of players
    n = game_list[0].n
    N = set(range(n))

    # define folder name and save path
    if inner_save_folder is not None:
        save_folder = os.path.join(
            "results",
            "_".join((DATA_FOLDER, str(n))) + save_path_addendum,
            inner_save_folder,
            INTERACTION_INDEX,
            str(time.time()),
        )
    else:
        save_folder = os.path.join(
            "results",
            "_".join((DATA_FOLDER, str(n))) + save_path_addendum,
            INTERACTION_INDEX,
            str(time.time()),
        )

    file_name = "_".join(
        (
            f"n-{n}",
            f"runs-{1}",
            f"n-inner-{N_INNER_ITERATIONS}",
            f"s0-{MAX_ORDER}",
            f"top-order-{RUN_TOP_ORDER}",
        )
    )
    file_name += ".json"
    SAVE_PATH = os.path.join(save_folder, file_name)

    print("Save path: ", SAVE_PATH)

    print("Loaded games.")
    print("Number of games: ", len(game_list))

    # Initialize estimators ------------------------------------------------------------------------

    approximators_to_run = {}

    # InterSVARM estimator for Shapley-Interaction and Shapley-Taylor indices
    if "SVARM-IQ" in APPROXIMATORS_TO_RUN:
        svarmiq_estimator = SvarmIQ(
            N=N,
            order=MAX_ORDER,
            interaction_type="SII",
            top_order=False,
            replacement=False,
            sample_strat="ksh",
        )
        approximators_to_run["SVARM-IQ"] = svarmiq_estimator

    # SHAP-IQ estimator for all three indices
    if "SHAP-IQ" in APPROXIMATORS_TO_RUN:
        shapiq_estimators = {}
        for order_ in range(1, MAX_ORDER + 1):
            if RUN_TOP_ORDER and order_ != MAX_ORDER:
                continue
            shapiq_estimator = SHAPIQEstimator(
                N=N,
                order=order_,
                interaction_type="SII",
                top_order=True,
            )
            shapiq_estimators[order_] = shapiq_estimator
        approximators_to_run["SHAP-IQ"] = copy.deepcopy(shapiq_estimators)

    # Kernel SHAP-IQ estimator for SII
    if "KernelSHAP-IQ" in APPROXIMATORS_TO_RUN:
        kernelshapiq_estimator = KernelSHAPIQEstimator(
            N=N,
            order=MAX_ORDER,
            interaction_type="SII",
            boosting=BOOSTING,
            approximator_mode=MODE,
            big_m=BIG_M,
        )
        approximators_to_run["KernelSHAP-IQ"] = kernelshapiq_estimator

    # Inconsistent Kernel SHAP-IQ estimator for SII
    if "KernelSHAP-IQ-Inconsistent" in APPROXIMATORS_TO_RUN:
        kernelshapiq_estimator = KernelSHAPIQEstimator(
            N=N,
            order=MAX_ORDER,
            interaction_type="SII",
            boosting=BOOSTING,
            approximator_mode="inconsistent",
            big_m=BIG_M,
        )
        approximators_to_run["KernelSHAP-IQ-Inconsistent"] = kernelshapiq_estimator

    # get baseline estimator
    if "Permutation" in APPROXIMATORS_TO_RUN:
        permutation_estimator = PermutationSampling(
            N=N, order=MAX_ORDER, interaction_type="SII", top_order=False
        )
        approximators_to_run["Permutation"] = permutation_estimator

    print("Initialized estimators.")

    # init nSII conversion estimator if nSII is selected  ------------------------------------------
    n_sii_converter = None
    if INTERACTION_INDEX == "nSII":
        n_sii_converter = SHAPIQEstimator(
            N=N, order=MAX_ORDER, interaction_type="SII", top_order=False
        )

    # Pre-compute the gt values --------------------------------------------------------------------
    print("Precomputing gt values.")

    gt_computer = SHAPIQEstimator(
        N=N, order=MAX_ORDER, interaction_type="SII", top_order=RUN_TOP_ORDER
    )

    for n, game in enumerate(game_list, start=1):
        gt_values = get_gt_values_for_game(game=game, shapiq=gt_computer, order=MAX_ORDER)
        if INTERACTION_INDEX == "nSII":
            gt_values = n_sii_converter.transform_interactions_in_n_shapley(
                interaction_values=gt_values, n=MAX_ORDER
            )
        all_gt_values[n] = gt_values

    # Run experiments ------------------------------------------------------------------------------
    print("Starting experiments.")
    RESULTS = run_approx_experiment(
        top_order=RUN_TOP_ORDER,
        game_list=game_list,
        approximators_to_run=approximators_to_run,
        n_sii_converter=n_sii_converter,
        all_gt_values=all_gt_values,
        order=MAX_ORDER,
        max_budget=MAX_BUDGET,
        pairing=False,
        stratification=False,
        sampling_kernel="ksh",
        budget_steps=BUDGET_STEPS,
        save_path=SAVE_PATH,
        save_folder=save_folder,
        n_inner_iterations=N_INNER_ITERATIONS,
    )

    # Save results ---------------------------------------------------------------------------------
    print("Saving results.")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    results_df = pd.DataFrame(RESULTS)
    results_df.to_json(SAVE_PATH)
    print("Done.")


if __name__ == "__main__":
    # GAME PARAMETERS ------------------------------------------------------------------------------
    DATA_FOLDER = "soum"
    DATA_N = 40

    # SOUM PARAMETERS ------------------------------------------------------------------------------
    SOUM_WEIGHTING_SCHEME = "uniform"
    SOUM_N_INTERACTIONS = 50
    SOUM_MAX_INTERACTION_SIZE = 4
    SOUM_MIN_INTERACTION_SIZE = 1
    SOUM_N_NON_IMPORTANT_FEATURES = 2

    # EXPERIMENT PARAMETERS ------------------------------------------------------------------------
    RUN_TOP_ORDER = False  # True, False

    # RUN PARAMETERS -------------------------------------------------------------------------------

    INTERACTION_INDEX = "SII"  # SII, nSII is supported
    MAX_ORDER = 2  # 1-3
    NUMBER_OF_RUNS = 1  # 1-50
    N_INNER_ITERATIONS = 1  # 1-10

    # BUDGET ---------------------------------------------------------------------------------------
    MAX_BUDGET = min(2**DATA_N, 10_000)
    BUDGET_STEPS = list(np.arange(0.15, 1.05, 0.05))
    # round budget steps to 2 decimals to avoid floating point errors
    BUDGET_STEPS = [round(budget_step, 2) for budget_step in BUDGET_STEPS]
    print("Budget steps: ", BUDGET_STEPS)

    # APPROXIMATORS --------------------------------------------------------------------------------

    APPROXIMATORS_TO_RUN = [
        "Permutation",
        "KernelSHAP-IQ",
        "KernelSHAP-IQ-Inconsistent",
        "SHAP-IQ",
        "SVARM-IQ",
    ]

    # KERNELSHAP-IQ PARAMETERS ---------------------------------------------------------------------

    BOOSTING = True
    MODE = "default"  # "full" "default" "inconsistent"
    BIG_M = 1_000_000

    print(
        "selected parameters:\n",
        f"  data_folder: {DATA_FOLDER}\n",
        f"  data_n: {DATA_N}\n",
        f"  interaction_index: {INTERACTION_INDEX}\n",
        f"  max_order: {MAX_ORDER}\n",
        f"  number_of_runs: {NUMBER_OF_RUNS}\n",
        f"  n_inner_iterations: {N_INNER_ITERATIONS}\n",
        f"  max_budget: {MAX_BUDGET}\n",
        f"  approximators_to_run: {APPROXIMATORS_TO_RUN}\n",
        f"  boosting: {BOOSTING}\n",
    )

    # add SOUM paramters to save path addendum
    SOUM_SAVE_FOLDER = "_".join(
        [
            f"weight-{SOUM_WEIGHTING_SCHEME}",
            f"n-interactions-{SOUM_N_INTERACTIONS}",
            f"max-interaction-size-{SOUM_MAX_INTERACTION_SIZE}",
            f"min-interaction-size-{SOUM_MIN_INTERACTION_SIZE}",
            f"n-non-important-features-{SOUM_N_NON_IMPORTANT_FEATURES}",
        ]
    )
    print("soum_save_folder: ", SOUM_SAVE_FOLDER)

    for i in range(NUMBER_OF_RUNS):
        print(f"Run {i+1}/{NUMBER_OF_RUNS}")
        run_experiment(save_path_addendum="", inner_save_folder=SOUM_SAVE_FOLDER)
