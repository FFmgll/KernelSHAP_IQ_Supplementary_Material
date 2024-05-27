"""This script explains a sentence instance and plots the interaction values as force plots from shap."""
import copy
import json
import random

import numpy as np
from matplotlib import pyplot as plt
from shap.plots import force, waterfall
from shap import Explanation
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from approximators.base import powerset
from approximators.plot import network_plot

from approximators import KernelSHAPIQEstimator, SHAPIQEstimator

if __name__ == "__main__":
    # Settings used for the force plot in the paper
    # random_state = 42
    # MAX_INTERACTION_ORDER = 2,3
    # EXPLANATION_INDEX = 2
    # force_limits = (0.4, 6.8)
    # model = GradientBoostingRegressor(max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=100, max_features=1.0)

    imputation_strategy = "mean"

    random_state = 42

    random.seed(random_state)
    np.random.seed(random_state)

    MAX_INTERACTION_ORDER = 2
    EXPLANATION_INDEX = 2
    SAVE_FIGURES = False
    dataset_name: str = "California"

    save_name = f"plots/intro_california_{MAX_INTERACTION_ORDER}_{EXPLANATION_INDEX}"

    n = 8
    N = set(range(n))

    force_limits = (-0.8, 6.8)  # (0.4, 6.8)

    model_flag: str = "GBT"  # "XGB" or "RF", "DT", "GBT", None
    if model_flag is not None:
        print("Model:", model_flag)

    # load the california housing dataset and pre-process ------------------------------------------
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    n_features = X.shape[-1]
    n_samples = len(X)
    feature_names = data["feature_names"]

    # train test split and get explanation datapoint -----------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=random_state
    )
    explanation_id = X_test.index[EXPLANATION_INDEX]

    # get explanation datapoint and index
    x_explain = np.asarray(X_test.iloc[EXPLANATION_INDEX].values)
    y_true_label = y_test.iloc[EXPLANATION_INDEX]

    # transform data to numpy arrays
    X, X_train, X_test, y_train, y_test = (
        X.values,
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
    )

    print("n_features", n_features, "n_samples", n_samples)

    # fit a tree model -----------------------------------------------------------------------------

    if model_flag == "GBT":
        model: GradientBoostingRegressor = GradientBoostingRegressor(
            max_depth=10, learning_rate=0.1, min_samples_leaf=5, n_estimators=10, max_features=1.0
        )
    else:
        model: XGBRegressor = XGBRegressor(random_state=random_state)

    model.fit(X_train, y_train)
    y_prediciton = model.predict(X_test)
    empty_prediction = np.mean(y_prediciton)
    empty_value = empty_prediction
    print("R^2 on test data", model.score(X_test, y_test))

    # run the experiment --------------------------------------------------------------------------

    imputation_values = np.mean(X_test, axis=0)
    imputation_values = np.expand_dims(imputation_values, axis=0)
    print("imputation_values", imputation_values.shape)

    imputation_values_all = copy.deepcopy(X_test)

    def game_fun(S: set):
        if not imputation_strategy == "mean":
            x = random.choice(imputation_values_all)
        else:
            x = copy.deepcopy(imputation_values)
        for i in S:
            x[:, i] = x_explain[i]
        return model.predict(x)[0]

    # explain --------------------------------------------------------------------------------------
    explainer = KernelSHAPIQEstimator(
        N=N,
        order=MAX_INTERACTION_ORDER,
        boosting=True,
        approximator_mode="separate",
        big_m=1_000_000,
    )

    n_sii_converter = SHAPIQEstimator(
        N=N, order=MAX_INTERACTION_ORDER, interaction_type="SII", top_order=False
    )

    # Compute the interaction values ---------------------------------------------------------------

    sii_values = explainer.approximate_with_budget(game_fun=game_fun, budget=2**n)
    n_sii_values = n_sii_converter.transform_interactions_in_n_shapley(
        interaction_values=sii_values, n=MAX_INTERACTION_ORDER
    )

    sum_of_n_sii_values = np.sum([np.sum(n_sii_values[order]) for order in n_sii_values.keys()])
    print(f"Sum of n-SII values: {sum_of_n_sii_values}")

    # transform feature names ----------------------------------------------------------------------
    # create feature names
    feature_names_abbrev = []
    for feature_name in feature_names:
        small_feature_name = feature_name[:5]
        # if the 5th character is a ' ' or '_' or '-' we want to extend the name by two characters
        if small_feature_name[-1] in [" ", "_", "-"]:
            small_feature_name += feature_name[5]
        feature_names_abbrev.append(small_feature_name + ".")

    # feature_names_abbrev = [feature[:5] + "." for feature in feature_names]
    feature_names_values = [
        feature + f"\n({round(x_explain[i], 2)})" for i, feature in enumerate(feature_names_abbrev)
    ]

    # get the feature names of the interactions ----------------------------------------------------
    n_sii_values_sv: np.ndarray = n_sii_values[1].copy()
    single_feature_names = copy.deepcopy(feature_names_abbrev)
    interaction_feature_names = []
    interaction_values = []
    interaction_feature_values = []
    interaction_feature_names_with_values = []
    for order in range(2, MAX_INTERACTION_ORDER + 1):
        for interaction in powerset(set(range(n_features)), min_size=order, max_size=order):
            comb_name: str = ""
            interaction_feature_value = ""
            for feature in interaction:
                if comb_name != "":
                    comb_name += " x "
                    interaction_feature_value += " x "
                comb_name += single_feature_names[feature]
                interaction_feature_value += f"{round(x_explain[feature], 2)}"
            interaction_values.append(n_sii_values[order][interaction])
            interaction_feature_names.append(comb_name)
            interaction_feature_values.append(interaction_feature_value)
            interaction_feature_names_with_values.append(
                comb_name + f"\n({interaction_feature_value})"
            )
    interaction_values = np.asarray(interaction_values)
    interaction_feature_values = np.asarray(interaction_feature_values)
    all_feature_names = single_feature_names + interaction_feature_names
    all_n_sii_values = np.concatenate([n_sii_values_sv, interaction_values])
    all_interaction_feature_values = np.concatenate(
        [np.round(x_explain, 2), interaction_feature_values]
    )
    all_feature_names_with_values = feature_names_values + interaction_feature_names_with_values

    # plot the force plot for the SV ---------------------------------------------------------------

    force(
        empty_value,
        sii_values[1],
        feature_names=feature_names_values,
        matplotlib=True,
        show=False,
        figsize=(15, 3),
    )
    if force_limits is not None:
        plt.xlim(force_limits)
    if SAVE_FIGURES:
        plt.savefig(save_name + "_force_SV.pdf", bbox_inches="tight")
    plt.show()

    # combine
    # all_feature_names = single_feature_names + interaction_feature_names
    # all_n_sii_values = np.concatenate([n_sii_values_sv, interaction_values])

    # print(all_feature_names)
    # all_n_sii_values_plus_empty = all_n_sii_values

    force(
        empty_value,
        all_n_sii_values,
        feature_names=all_feature_names_with_values,
        matplotlib=True,
        show=False,
        figsize=(15, 3),
    )
    if force_limits is not None:
        plt.xlim(force_limits)
    if SAVE_FIGURES:
        plt.savefig(save_name + "_force_2-SII.pdf", bbox_inches="tight")
    plt.show()

