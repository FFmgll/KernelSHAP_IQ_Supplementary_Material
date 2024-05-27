"""This script explains a sentence instance and plots the interaction values as force plots from shap."""
from copy import copy
import json

import numpy as np
from matplotlib import pyplot as plt
from shap.plots import force, waterfall
from shap import Explanation

from approximators.base import powerset
from approximators.plot import network_plot
from games import NLPGame, NLPLookupGame
from approximators import KernelSHAPIQEstimator, SHAPIQEstimator

if __name__ == "__main__":
    # Set up the game and get sentence and parts ---------------------------------------------------
    SAVE_FIGURES = True

    plt.rcParams["font.size"] = 10
    plt.rcParams["legend.fontsize"] = 7

    ORDER = 2
    ORDER_FORCE = 2
    SENTENCE_ID = 1400  # 845, 600
    MAX_BUDGET = 10_000
    run_lookup = False

    if run_lookup:
        game = NLPLookupGame(n=14, sentence_id=SENTENCE_ID, set_zero=True)
        input_sentence = game.input_sentence
    else:
        input_sentence = (
            # Paper examples
            # "TWINS EFFECT is a poor film in so many respects The only good element is that it doesn't take itself seriously"  # 14548
            # "I rented this movie to get an easy, entertained view of the history of Texas. I got a headache instead."  # 2965
            "I have Never forgot this movie All these years and it has remained in my life"  # 25882
            # "I still remember watching Satya for the first time I was completely blown away"  # 26445
            # Interesting examples
            # "I love this movie Purple Rain came out the year I was born and it has had my heart since I can remember"
            # "I am still trying to determine whether the previous installment was worse than this one or vice versa"  # 2827
            # "I was impressed by the beautiful photography in this film which was shot on location in Alaska"  # 2828
        )
        print_sentence = (
            # "TWINS EFFECT is a poor film\nin so many respects. The\nonly good element is\nthat it doesn't take\nitself seriously"
            "I have never forgot this\nmovie. All these years\nand it has remained in\nmy life."
            # "I still remember watching\nSatya for the first\ntime I was com-\npletely blown away."
        )

    TEST_REDUCE = False

    set_zero = False
    force_limits = None  # (-3.2, 2.6)
    show_plots = True

    sentence_code = input_sentence.replace(" ", "_").replace("\n", "_")
    sentence_code = sentence_code.replace(".", "").replace(",", "")
    sentence_code = sentence_code.replace("'", "").replace('"', "")
    # get first letters of each word
    sentence_code = "".join([word[0] for word in sentence_code.split(" ")])
    save_name: str = f"plots/sentence_{sentence_code}_order_{ORDER}"

    nlp_game = NLPGame(input_text=input_sentence, set_zero=set_zero)
    if not run_lookup:
        game = nlp_game

    empty_value = nlp_game.empty_value
    original_empty_value = nlp_game.original_empty_value
    tokens = nlp_game.tokenized_input
    feature_names = [nlp_game.tokenizer.decode([token]) for token in tokens]
    original_output_nlp = nlp_game.original_output[0]

    n = nlp_game.n
    N = set(range(n))
    if not run_lookup:
        BUDGET = min(2**n, MAX_BUDGET)
    else:
        BUDGET = 2**n

    print(
        f"Input sentence: {input_sentence}\n"
        f"Tokenized input: {feature_names}\n"
        f"Number of tokens: {n}\n"
        f"Original output NLP: {original_output_nlp}\n"
        f"Empty value: {empty_value}\n"
        f"Original empty value: {original_empty_value}\n"
        f"Original output NLP - Empty value: {original_output_nlp - empty_value}\n"
        f"Budget: {BUDGET}\n"
    )

    # Set up the explainer -------------------------------------------------------------------------

    explainer = KernelSHAPIQEstimator(
        N=N,
        order=ORDER,
        boosting=True,
        approximator_mode="default",
        big_m=1_000_000,
    )

    n_sii_converter = SHAPIQEstimator(N=N, order=ORDER, interaction_type="SII", top_order=False)

    # Compute the interaction values ---------------------------------------------------------------

    sii_values = explainer.approximate_with_budget(game_fun=game.set_call, budget=BUDGET)
    n_sii_values = n_sii_converter.transform_interactions_in_n_shapley(
        interaction_values=sii_values, n=ORDER
    )

    sum_of_n_sii_values = np.sum([np.sum(n_sii_values[order]) for order in n_sii_values.keys()])
    print(f"Sum of n-SII values: {sum_of_n_sii_values}")

    # plot the force plot for the SV ---------------------------------------------------------------
    force(
        original_empty_value,
        sii_values[1],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        figsize=(15, 5),
    )
    if force_limits is not None:
        plt.xlim(force_limits)
    if SAVE_FIGURES:
        plt.savefig(save_name + "_force_SV.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")

    # get the feature names of the order -----------------------------------------------------------
    n_sii_values_sv: np.ndarray = n_sii_values[1].copy()
    single_feature_names = copy(feature_names)

    interaction_feature_names = []
    interaction_values = []
    interaction_feature_values = []
    for order in range(2, ORDER_FORCE + 1):
        for interaction in powerset(set(range(n)), min_size=order, max_size=order):
            comb_name: str = ""
            for feature in interaction:
                if comb_name != "":
                    comb_name += " x "
                comb_name += single_feature_names[feature]
            interaction_values.append(n_sii_values[order][interaction])
            interaction_feature_names.append(comb_name)
    interaction_values = np.asarray(interaction_values)
    interaction_feature_values = np.asarray(interaction_feature_values)

    # combine
    all_feature_names = single_feature_names + interaction_feature_names
    all_n_sii_values = np.concatenate([n_sii_values_sv, interaction_values])

    print(all_feature_names)
    all_n_sii_values_plus_empty = all_n_sii_values

    """
    force(
        original_empty_value,
        all_n_sii_values_plus_empty,
        feature_names=all_feature_names,
        matplotlib=True,
        show=False,
        figsize=(20, 3),
    )
    if force_limits is not None:
        plt.xlim(force_limits)
    if save_figures:
        plt.savefig(save_name + "_force_n_SII.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")

    # water fall plot ------------------------------------------------------------------------------

    n_sii_explanation = Explanation(
        values=all_n_sii_values,
        base_values=original_empty_value,
        data=None,
        feature_names=all_feature_names,
    )
    waterfall(n_sii_explanation, show=False)
    if save_figures:
        plt.savefig(save_name + "_waterfall_n_sii.pdf", bbox_inches="tight")
    plt.show() if show_plots else plt.close("all")
    """

    # plot the network plot ------------------------------------------------------------------------

    # plot the n-Shapley values
    fig_network, axis_network = network_plot(
        first_order_values=n_sii_values[1],
        second_order_values=n_sii_values[2],
        feature_names=all_feature_names,
        draw_legend=False,
        center_text=print_sentence,
    )
    title_network: str = f"Network Plot\npredicted output: {original_output_nlp:.2f}"
    axis_network.set_title(title_network)
    if SAVE_FIGURES:
        fig_network.subplots_adjust(bottom=0.01, top=0.9, left=0.05, right=0.9)
        fig_network.savefig(save_name + "_network.pdf", bbox_inches=None)
    fig_network.show() if show_plots else plt.close("all")

    # print the top 20 highest n_sii values (all_n_sii_values) -------------------------------------

    print("\nTop 20 highest n-SII values:")
    all_n_sii_values_abs = np.abs(all_n_sii_values)
    top_indices = np.argsort(all_n_sii_values_abs)[::-1][:20]
    for idx in top_indices:
        print(f"{all_feature_names[idx]}: {all_n_sii_values[idx]}")

    # save result dict as json if order is max_order (n) -------------------------------------------

    if ORDER == n:
        result_dict = {
            "n_SII_values": all_n_sii_values.tolist(),
            "feature_names": all_feature_names,
            "input_sentence": input_sentence,
        }
        with open(save_name + "_result_dict.json", "w") as f:
            json.dump(result_dict, f)
