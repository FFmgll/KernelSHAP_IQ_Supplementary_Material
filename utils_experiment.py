"""This file contains utility functions for the project."""
import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import kendalltau

from approximators import SHAPIQEstimator


def get_all_errors(
    approx,
    exact,
    n: int,
    order: int,
    top_order: bool = False,
    variance: dict[int, np.ndarray] = None,
):
    """Computes all errors for the given approximation and exact values."""
    errors = {}
    if type(exact) == dict and type(approx) == dict:
        for order in exact.keys():
            try:
                errors[order] = _get_all_errors_arr(
                    approx[order], exact[order], n, order, variance=variance[order]
                )
            except TypeError:
                errors[order] = _get_all_errors_arr(approx[order], exact[order], n, order)
    else:
        errors[order] = _get_all_errors_arr(approx, exact, n, order, variance=variance)
    if not top_order:
        # get all errors of 'approximation_error' for each order, multiply them by the binom(n, order) and sum them up
        errors_all = np.sum(
            [errors[order]["approximation_error"] * binom(n, order) for order in errors.keys()]
        )
        errors_all /= _n_interactions_of_order(n, order)
        errors[0] = {"approximation_error": errors_all}

    return errors


def _get_all_errors_arr(
    approx: np.ndarray, exact: np.ndarray, n: int, order: int, variance: np.ndarray = None
):
    """Helper for computing all errors for the given approximation and exact values."""
    errors = {}
    errors["approximation_error"] = get_approximation_error(approx, exact, n, order)
    errors["precision_at_10"] = get_precision_at_k(approx, exact, k=10)
    errors["approximation_error_at_10"] = get_approximation_error_at_k(approx, exact, k=10)
    errors["kendals_tau"] = get_kendals_tau(approx, exact)
    if variance is not None:
        errors["variance"] = get_variance(variance, n, order)
    else:
        errors["variance"] = 0.0
    return errors


def get_variance(variance: np.ndarray, n: int, order: int) -> float:
    """Computes the variance of the given variance values."""
    variance = np.sum(variance) / binom(n, order)
    return variance


def get_approximation_error(approx: np.ndarray, exact: np.ndarray, n: int, order: int) -> float:
    """Computes the approximation error of the given approximation and exact values."""
    error = np.sum((approx - exact) ** 2) / binom(n, order)
    return error


def get_precision_at_k(approx: np.ndarray, exact: np.ndarray, k: int = 10) -> float:
    """Computes the precision at the top k absolute values."""
    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    top_k_id_exact = set(exact_abs.argsort()[-k:])

    approx_abs = np.abs(approx)
    approx_abs = approx_abs.flatten()
    top_k_id_approx = set(approx_abs.argsort()[-k:])

    wrong_ids = len(top_k_id_approx - top_k_id_exact)
    correct_ids = k - wrong_ids
    return float(correct_ids / k)


def get_approximation_error_at_k(approx: np.ndarray, exact: np.ndarray, k: int = 10) -> float:
    """Computes the approximation error at the top k absolute values."""
    exact_flat = exact.flatten()
    approx_flat = approx.flatten()

    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    top_k_id_exact = set(exact_abs.argsort()[-k:])

    error = np.sum((approx_flat[[*top_k_id_exact]] - exact_flat[[*top_k_id_exact]]) ** 2)
    return float(error)


def get_kendals_tau(approx: np.ndarray, exact: np.ndarray) -> float:
    """Computes the kendal's tau between the approximated and exact values."""
    exact = exact.flatten()
    ranking_exact = exact.argsort()
    approx = approx.flatten()
    ranking_approx = approx.argsort()
    tau, _ = kendalltau(ranking_exact, ranking_approx)
    return tau


def get_gt_values_for_game(game, shapiq: SHAPIQEstimator, order: int) -> Dict[int, np.ndarray]:
    """Computes the ground truth values for a given game and order."""
    try:
        gt_values = game.exact_values(
            gamma_matrix=shapiq.weights[order], min_order=order, max_order=order
        )
        return gt_values
    except AttributeError:
        gt_values = shapiq.compute_interactions_complete(game=game.set_call)
    except TypeError:
        gt_values = game.exact_values()
    return gt_values


def save_values(save_path: str, values: list):
    save_dir = os.path.join(*os.path.split(save_path)[0:-1])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df = pd.DataFrame(values)
    if not os.path.isfile(save_path):
        df.to_csv(save_path, header=True, index=False)
    else:
        df.to_csv(save_path, mode="a", header=False, index=False)


def _n_interactions_of_order(n: int, order: int) -> int:
    """Computes the number of interactions up to a given order starting from order 1."""
    n_interactions = 0
    for i in range(1, order + 1):
        n_interactions += binom(n, i)
    return int(n_interactions)


def get_mean_and_var_of_approx(
    approx: list[dict[int, np.ndarray]]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Computes the mean and variance of the given approximation per order."""
    mean = {}
    var = {}
    for order in approx[0].keys():
        mean[order] = np.mean([approx[i][order] for i in range(len(approx))], axis=0)
        var[order] = np.var([approx[i][order] for i in range(len(approx))], axis=0)
    return mean, var
