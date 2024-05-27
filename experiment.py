"""This module contains the experiment function for the top order experiment."""
import copy
import os
from typing import Union

import numpy as np
import pandas as pd
import tqdm

from approximators import (
    SHAPIQEstimator,
    PermutationSampling,
    SvarmIQ,
    KernelSHAPIQEstimator,
)
from utils_experiment import get_all_errors, get_mean_and_var_of_approx


def run_approx_experiment(
    top_order: bool,
    game_list: list,
    approximators_to_run: dict[
        str,
        Union[
            SHAPIQEstimator,
            PermutationSampling,
            SvarmIQ,
            KernelSHAPIQEstimator,
            dict[int, SHAPIQEstimator],
            dict[int, PermutationSampling],
        ],
    ],
    all_gt_values: dict,
    max_budget: int,
    order: int,
    n_inner_iterations: int = 1,
    n_sii_converter: SHAPIQEstimator = None,
    sampling_kernel="ksh",
    stratification=False,
    pairing=True,
    budget_steps: list = None,
    save_folder: str = None,
    save_path: str = None,
) -> dict:
    """Computes the experiment for a given list of games and shapiq estiamtors."""

    # get the budget list
    if budget_steps is None:
        budget_steps = np.arange(0, 1.05, 0.05)  # step size of computation budgets

    # initialize results dict from approximators_to_run
    RESULTS = {approximator_name: {} for approximator_name in approximators_to_run.keys()}
    n_approximators_to_run = len(approximators_to_run.keys())

    if "SHAP-IQ" in approximators_to_run.keys():
        n_shapiq_estimators = len(approximators_to_run["SHAP-IQ"].keys())
        if n_shapiq_estimators > 1:
            n_approximators_to_run += n_shapiq_estimators - 1

    pbar = tqdm.tqdm(
        total=np.sum(budget_steps * max_budget)  # budgets
        * len(game_list)  # number of games
        * n_approximators_to_run  # number of approximators
        * n_inner_iterations  # number of inner iterations
    )

    for budget_step in budget_steps:
        budget = int(budget_step * max_budget)

        budget_errors_all = {
            approximator_name: {} for approximator_name in approximators_to_run.keys()
        }

        for i, game in enumerate(game_list, start=1):
            n = game.n
            # get the correct gt_values
            gt_values = all_gt_values[i]

            approx_results_lists = {
                approximator_name: [] for approximator_name in approximators_to_run.keys()
            }

            for inner_iteration in range(n_inner_iterations):
                # approximate with svarmiq ---------------------------------------------------------
                if "SVARM-IQ" in approximators_to_run.keys():
                    svarmiq_estimator = approximators_to_run["SVARM-IQ"]
                    svarmiq_approx = svarmiq_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    if n_sii_converter is not None:
                        svarmiq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=svarmiq_approx, n=order
                        )
                    approx_results_lists["SVARM-IQ"].append(copy.deepcopy(svarmiq_approx))
                    pbar.update(budget)

                # approximate with shapiq ----------------------------------------------------------
                if "SHAP-IQ" in approximators_to_run.keys():
                    shapiq_estimators = approximators_to_run["SHAP-IQ"]
                    shap_iq_approx = {}
                    for order_, shapiq_estimator in shapiq_estimators.items():
                        shap_iq_approx_order = shapiq_estimator.compute_interactions_from_budget(
                            game.set_call,
                            budget,
                            sampling_kernel=sampling_kernel,
                            pairing=pairing,
                            stratification=stratification,
                        )
                        shap_iq_approx[order_] = shap_iq_approx_order[order_]
                        pbar.update(budget)
                    if n_sii_converter is not None:
                        shap_iq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=shap_iq_approx, n=order
                        )
                    approx_results_lists["SHAP-IQ"].append(copy.deepcopy(shap_iq_approx))

                # approximate with kernelshapiq ----------------------------------------------------
                if "KernelSHAP-IQ" in approximators_to_run.keys():
                    kernelshapiq_estimator = approximators_to_run["KernelSHAP-IQ"]
                    kernelshap_iq_approx = kernelshapiq_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    if n_sii_converter is not None:
                        kernelshap_iq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=kernelshap_iq_approx, n=order
                        )
                    approx_results_lists["KernelSHAP-IQ"].append(
                        copy.deepcopy(kernelshap_iq_approx)
                    )
                    pbar.update(budget)

                # approximate with regression ------------------------------------------------------
                if "KernelSHAP-IQ-Inconsistent" in approximators_to_run.keys():
                    kernelshapiq_inc_estimator = approximators_to_run["KernelSHAP-IQ-Inconsistent"]
                    kernelshapiq_inc_approx = kernelshapiq_inc_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    if n_sii_converter is not None:
                        kernelshapiq_inc_approx = (
                            n_sii_converter.transform_interactions_in_n_shapley(
                                interaction_values=kernelshapiq_inc_approx, n=order
                            )
                        )
                    approx_results_lists["KernelSHAP-IQ-Inconsistent"].append(
                        copy.deepcopy(kernelshapiq_inc_approx)
                    )
                    pbar.update(budget)

                # approximate with baseline --------------------------------------------------------
                if "Permutation" in approximators_to_run.keys():
                    permutation_estimator = approximators_to_run["Permutation"]
                    permutation_approx = permutation_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    if n_sii_converter is not None:
                        permutation_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=permutation_approx, n=order
                        )
                    approx_results_lists["Permutation"].append(copy.deepcopy(permutation_approx))
                    pbar.update(budget)

                # approximators for testing different KernelSHAP-IQ Settings ---------------------------
                if "KernelSHAP-IQ-joint-one_order_less" in approximators_to_run.keys():
                    kernelshapiq_estimator = approximators_to_run[
                        "KernelSHAP-IQ-joint-one_order_less"
                    ]
                    kernelshap_iq_approx = kernelshapiq_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    estimator_other_order = approximators_to_run["KernelSHAP-IQ-joint"]
                    kernelshap_iq_approx[order] = estimator_other_order.init_results()[order]
                    if n_sii_converter is not None:
                        kernelshap_iq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=kernelshap_iq_approx, n=order
                        )
                    approx_results_lists["KernelSHAP-IQ-joint-one_order_less"].append(
                        copy.deepcopy(kernelshap_iq_approx)
                    )
                    pbar.update(budget)

                if "KernelSHAP-IQ-separate-one_order_less" in approximators_to_run.keys():
                    kernelshapiq_estimator = approximators_to_run[
                        "KernelSHAP-IQ-separate-one_order_less"
                    ]
                    kernelshap_iq_approx = kernelshapiq_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    estimator_other_order = approximators_to_run["KernelSHAP-IQ-separate"]
                    kernelshap_iq_approx[order] = estimator_other_order.init_results()[order]
                    if n_sii_converter is not None:
                        kernelshap_iq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=kernelshap_iq_approx, n=order
                        )
                    approx_results_lists["KernelSHAP-IQ-separate-one_order_less"].append(
                        copy.deepcopy(kernelshap_iq_approx)
                    )
                    pbar.update(budget)

                if "KernelSHAP-IQ-joint" in approximators_to_run.keys():
                    kernelshapiq_estimator = approximators_to_run["KernelSHAP-IQ-joint"]
                    kernelshap_iq_approx = kernelshapiq_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    if n_sii_converter is not None:
                        kernelshap_iq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=kernelshap_iq_approx, n=order
                        )
                    approx_results_lists["KernelSHAP-IQ-joint"].append(
                        copy.deepcopy(kernelshap_iq_approx)
                    )
                    pbar.update(budget)

                if "KernelSHAP-IQ-separate" in approximators_to_run.keys():
                    kernelshapiq_estimator = approximators_to_run["KernelSHAP-IQ-separate"]
                    kernelshap_iq_approx = kernelshapiq_estimator.approximate_with_budget(
                        game.set_call, budget
                    )
                    if n_sii_converter is not None:
                        kernelshap_iq_approx = n_sii_converter.transform_interactions_in_n_shapley(
                            interaction_values=kernelshap_iq_approx, n=order
                        )
                    approx_results_lists["KernelSHAP-IQ-separate"].append(
                        copy.deepcopy(kernelshap_iq_approx)
                    )
                    pbar.update(budget)

            # if n_inner_iterations > 1, compute mean of approximations  ---------------------------
            approx_mean = {
                approximator_name: None for approximator_name in approximators_to_run.keys()
            }
            approx_var = {
                approximator_name: None for approximator_name in approximators_to_run.keys()
            }
            if n_inner_iterations > 1:
                for approximator_name in approximators_to_run.keys():
                    mean, var = get_mean_and_var_of_approx(approx_results_lists[approximator_name])
                    approx_mean[approximator_name] = mean
                    approx_var[approximator_name] = var
            else:
                for approximator_name in approximators_to_run.keys():
                    approx_mean[approximator_name] = approx_results_lists[approximator_name][0]

            # get errors and append to list --------------------------------------------------------
            for approximator_name in approximators_to_run.keys():
                errors = get_all_errors(
                    approx_mean[approximator_name],
                    gt_values,
                    n=n,
                    order=order,
                    top_order=top_order,
                    variance=approx_var[approximator_name],
                )
                for order_ in errors.keys():
                    try:
                        budget_errors_all[approximator_name][order_].append(errors[order_])
                    except KeyError:
                        try:
                            budget_errors_all[approximator_name][order_] = [errors[order_]]
                        except KeyError:
                            budget_errors_all[approximator_name] = {order_: [errors[order_]]}

        # compute mean and std of errors for each approximator -------------------------------------
        for approximator_name in approximators_to_run.keys():
            for order_ in budget_errors_all[approximator_name].keys():
                approx_order_errors = budget_errors_all[approximator_name][order_]
                errors_df = pd.DataFrame(approx_order_errors)

                mean = dict(errors_df.mean())
                median = dict(errors_df.median())
                q_1 = dict(errors_df.quantile(0.25))
                q_3 = dict(errors_df.quantile(0.75))
                std = dict(errors_df.std())
                var = dict(errors_df.var())

                dict_to_append = {
                    "budget": budget,
                    "mean": mean,
                    "std": std,
                    "var": var,
                    "median": median,
                    "q_1": q_1,
                    "q_3": q_3,
                }

                try:
                    RESULTS[approximator_name][order_].append(dict_to_append)
                except KeyError:
                    RESULTS[approximator_name][order_] = [dict_to_append]

        # save results to json
        if save_folder is not None and save_path is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            pd.DataFrame(RESULTS).to_json(save_path)

    pbar.close()
    return copy.deepcopy(RESULTS)
