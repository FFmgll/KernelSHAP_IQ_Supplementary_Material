import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

COLORS = {
    "SHAP-IQ": "#ef27a6",
    "Permutation": "#7d53de",
    "SVARM-IQ": "#00b4d8",
    "KernelSHAP-IQ": "#ff6f00",
    "KernelSHAP-IQ-Inconsistent": "#ffba08",
    "Baseline": "#7d53de",
    "KernelSHAP-IQ-joint": "red",
    "KernelSHAP-IQ-separate": "blue",
    "KernelSHAP-IQ-joint-one_order_less": "lightcoral",
    "KernelSHAP-IQ-separate-one_order_less": "skyblue",
}
APPROX_NAMES_DICT = {
    "Permutation": "Permutation",
    "SHAP-IQ": "SHAP-IQ",
    "SVARM-IQ": "SVARM-IQ",
    "KernelSHAP-IQ": "KernelSHAP-IQ",
    "KernelSHAP-IQ-Inconsistent": "KernelSHAP-IQ\n(inconsistent)",
    "KernelSHAP-IQ-joint": "joint",
    "KernelSHAP-IQ-separate": "separate",
    "KernelSHAP-IQ-joint-one_order_less": "joint (one order less)",
    "KernelSHAP-IQ-separate-one_order_less": "separate (one order less)",
}
LINESTYLE_DICT_INDEX = {"SII": "solid", "STI": "dashed", "FSI": "dashdot"}
LINESTYLE_DICT_ORDER = {0: "solid", 1: "dotted", 2: "solid", 3: "dashed", 4: "dashdot"}
ERROR_NAME_DICT = {
    "approximation_error": "MSE",
    "kendals_tau": "Kendall's $\\tau$",
    "precision_at_10": "Prec@10",
    "approximation_error_at_10": "MSE@10",
    "variance": "Variance",
}
LINE_MARKERS_DICT_ORDER = {0: "o", 1: "o", 2: "s", 3: "X", 4: "d"}
LINE_MARKERS_DICT_INDEX = {"SII": "o", "STI": "s", "FSI": "X"}
GAME_NAME_DICT = {
    "vision_transformer": "ViT",
    "nlp_values": "LM",
    "image_classifier": "CNN",
    "california": "CH",
    "bike": "BR",
    "adult": "AC",
    "soum": "SOUM",
}
LINE_THICKNESS_DICT = {
    "KernelSHAP-IQ": 2,
    "KernelSHAP-IQ-Inconsistent": 2,
}


def add_approx_to_figure(approx_id, order_df, ax, x_min_index, x_max_index) -> int:
    if plot_mean:
        error_values = (
            order_df["mean"]
            .apply(lambda x: x[error_to_plot_id])
            .values[x_min_index:x_max_index]
        )
    else:
        error_values = (
            order_df["median"]
            .apply(lambda x: x[error_to_plot_id])
            .values[x_min_index:x_max_index]
        )

    max_value = max(error_values)

    ax.plot(
        x_data,
        error_values,
        color=COLORS[approx_id],
        linestyle=LINESTYLE_DICT_ORDER[order],
        marker=LINE_MARKERS_DICT_ORDER[order],
        mec="white",
        linewidth=LINE_THICKNESS_DICT.get(approx_id, 1),
    )

    if plot_iqr:
        q1_values = (
            order_df["q_1"]
            .apply(lambda x: x[error_to_plot_id])
            .values[x_min_index:x_max_index]
        )
        q3_values = (
            order_df["q_3"]
            .apply(lambda x: x[error_to_plot_id])
            .values[x_min_index:x_max_index]
        )
        ax.fill_between(
            x_data,
            q1_values,
            q3_values,
            alpha=0.2,
            color=COLORS[approx_id],
        )

    if plot_std or plot_std_error:
        if error_to_plot_id == "kendals_tau" or error_to_plot_id == "precision_at_10":
            max_value = 1
        std = (
            order_df["std"]
            .apply(lambda x: x[error_to_plot_id])
            .values[x_min_index:x_max_index]
        )
        if plot_std_error:
            std = std / np.sqrt(NUMBER_OF_RUNS)

        ax.fill_between(
            x_data,
            error_values - std,
            error_values + std,
            alpha=0.2,
            color=COLORS[approx_id],
        )
        return max_value


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SAVE_FIG = False

    # experiment parameters for loading the file ---------------------------------------------------

    GAME_NAME = "soum"  # "california" "splice" "bike", "adult", "vision_transformer", "nlp_values", "image_classifier"
    N_PLAYER = 20  # 9, 12, 14, 16

    INTERACTION_INDEX = "SII"  # index
    TOP_ORDER = False  # weather top order or not (default: False)
    ORDER = 2  # interaction order
    NUMBER_OF_RUNS = 30  # how often the experiment was run
    N_INNER_ITERATIONS = 1  # how often the experiment was run per run

    SAVE_PATH_ADDENDUM = ""  # "_full"
    SAVE_PATH_ADDENDUM += ""  # "_no-boosting" "_no-boosting_second"
    SAVE_PATH_ADDENDUM += ""  # "_big-m-10000"  # "_big-m-10000"
    SAVE_PATH_ADDENDUM += ""  # "_naive-sampling"
    SAVE_PATH_ADDENDUM += ""  # "KernelSHAP-IQ-test"
    SAVE_PATH_ADDENDUM += ""  # "_lookup_min_max"

    # SOUM PARAMETERS (only used for soum) ---------------------------------------------------------

    SOUM_WEIGHTING_SCHEME = "uniform"
    SOUM_N_INTERACTIONS = 50
    SOUM_MAX_INTERACTION_SIZE = 4
    SOUM_MIN_INTERACTION_SIZE = 1
    SOUM_N_NON_IMPORTANT_FEATURES = 2

    SOUM_FOLDER_NAME = "_".join(
        [
            f"weight-{SOUM_WEIGHTING_SCHEME}",
            f"n-interactions-{SOUM_N_INTERACTIONS}",
            f"max-interaction-size-{SOUM_MAX_INTERACTION_SIZE}",
            f"min-interaction-size-{SOUM_MIN_INTERACTION_SIZE}",
            f"n-non-important-features-{SOUM_N_NON_IMPORTANT_FEATURES}",
        ]
    )

    file_name = (
        f"n-{N_PLAYER}_runs-{NUMBER_OF_RUNS}_n-inner-{N_INNER_ITERATIONS}_s0-{ORDER}_top-order-{TOP_ORDER}"
        + ".json"
    )

    if GAME_NAME == "soum" or GAME_NAME == "no_kernel_shapiq_soum":
        file_path = os.path.join(
            "results_plot",
            "_".join((GAME_NAME, str(N_PLAYER))) + SAVE_PATH_ADDENDUM,
            SOUM_FOLDER_NAME,
            INTERACTION_INDEX,
            file_name,
        )
    else:
        file_path = os.path.join(
            "results_plot",
            "_".join((GAME_NAME, str(N_PLAYER))) + SAVE_PATH_ADDENDUM,
            INTERACTION_INDEX,
            file_name,
        )

    approx_kind = ""  # "" "_1" "_2" "_avg"

    # plot parameters ------------------------------------------------------------------------------
    approx_to_plot = (
        "Permutation",
        "SHAP-IQ",
        "SVARM-IQ",
        "KernelSHAP-IQ",
        "KernelSHAP-IQ-Inconsistent",
    )

    error_to_plot_id = "approximation_error"  # "variance" "approximation_error" 'precision_at_10' 'approximation_error_at_10' 'kendals_tau'
    orders_to_plot = [2]
    plot_mean = True
    plot_iqr = False
    plot_std = False
    plot_std_error = True
    y_max_manual = (
        None  # 0.0037 LM  # 0.00006 bike # 0.005, calif # 0.0001 bike appendix
    )
    y_min_manual = 0  # 3e-6  # 0
    x_min_to_plot = 0  # minimum is 1
    x_max_to_plot = 0  # 2**12 + 100  # 16_600  # 513 # 16_600 # 66_400

    log_scale = False
    science_scale = True

    # legend params
    loc_legend = "best"
    n_col_legend = 1
    n_empty_legend_items: int = 0
    plot_order_legend = True
    plot_legend = False

    # load data ------------------------------------------------------------------------------------

    # read json file with pandas
    df = pd.read_json(file_path)
    if "KernelSHAP-IQ" in approx_to_plot:
        kernelshapiq_dict = dict(df["KernelSHAP-IQ" + approx_kind])
    else:
        kernelshapiq_dict = dict(df[approx_to_plot[0]])

    orders_in_file = list(kernelshapiq_dict.keys())
    orders_to_plot = orders_in_file if orders_to_plot is None else orders_to_plot

    params = {
        "legend.fontsize": "x-large",
        "figure.figsize": (6, 7),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    plt.rcParams.update(params)

    # get plot canvas
    fig, ax = plt.subplots()

    y_max_value = 0
    x_data = None

    plotted_legend = False
    ax.plot([], [], label="$\\bf{Method}$", color="none")
    for order in orders_to_plot:
        # get kernelshap-iq df
        kernelshapiq_dict_order_df = pd.DataFrame(kernelshapiq_dict[order])

        # get baseline dataframes

        # get x data
        x_data = kernelshapiq_dict_order_df["budget"].values
        # get first index of x_data that is greater than x_min_to_plot
        x_min_index = next(i for i, x in enumerate(x_data) if x > x_min_to_plot)
        # get first index of x_data that is greater than x_max_to_plot
        x_max_index = len(x_data)
        if x_max_to_plot is not None:
            if not max(x_data) > x_max_to_plot:
                for i, x_index in enumerate(reversed(x_data)):
                    if x_index > x_max_to_plot:
                        x_max_index = x_max_index - i
                    else:
                        break
        x_data = x_data[x_min_index:x_max_index]
        for approx_id in approx_to_plot:
            if approx_id == "KernelSHAP-IQ":
                order_df = kernelshapiq_dict_order_df
            elif approx_id == "Permutation":
                baseline_dict = dict(df["Permutation" + approx_kind])
                baseline_dict_order_df = pd.DataFrame(baseline_dict[order])
                order_df = baseline_dict_order_df
            elif approx_id == "SHAP-IQ":
                shapiq_dict = dict(df["SHAP-IQ" + approx_kind])
                shapiq_dict_order_df = pd.DataFrame(shapiq_dict[order])
                order_df = shapiq_dict_order_df
            elif approx_id == "SVARM-IQ":
                intersvarm_dict = dict(df["SVARM-IQ" + approx_kind])
                intersvarm_dict_order_df = pd.DataFrame(intersvarm_dict[order])
                order_df = intersvarm_dict_order_df
            elif approx_id == "KernelSHAP-IQ-Inconsistent":
                kernelshapiq_inc_dict = dict(
                    df["KernelSHAP-IQ-Inconsistent" + approx_kind]
                )
                kernelshapiq_inc_dict_order_df = pd.DataFrame(
                    kernelshapiq_inc_dict[order]
                )
                order_df = kernelshapiq_inc_dict_order_df
            else:
                try:
                    baseline_dict = dict(df[approx_id])
                    baseline_dict_order_df = pd.DataFrame(baseline_dict[order])
                    order_df = baseline_dict_order_df
                except KeyError:
                    raise ValueError(
                        f"{approx_id + approx_kind} approx_id not recognized."
                    )

            max_value = add_approx_to_figure(
                approx_id, order_df, ax, x_min_index, x_max_index
            )
            try:
                y_max_value = max(y_max_value, max_value)
            except TypeError:
                pass

            if not plotted_legend:
                ax.plot(
                    [],
                    [],
                    color=COLORS[approx_id],
                    linestyle="solid",
                    label=APPROX_NAMES_DICT[approx_id],
                )

        plotted_legend = True

    ax.set_title(f"Interaction index: {INTERACTION_INDEX}")

    if plot_order_legend:
        ax.plot([], [], label="$\\bf{Order}$", color="none")
        for order in orders_to_plot:
            label_text = (
                r"$l$" + f" = {order}"
                if order > 0
                else r"all to $l$" + f" = {max(orders_to_plot)}"
            )
            ax.plot(
                [],
                [],
                color="black",
                linestyle=LINESTYLE_DICT_ORDER[order],
                label=label_text,
                marker=LINE_MARKERS_DICT_ORDER[order],
                mec="white",
            )

        for i in range(n_empty_legend_items):
            ax.plot([], [], color="none", label=" ")

    if plot_legend:
        ax.legend(ncols=n_col_legend, loc=loc_legend)

    order_title = r"$l =$" + f"{max(orders_to_plot)}"

    # set y axis limits
    if not log_scale:
        y_min = 0 if y_min_manual is None else y_min_manual
        ax.set_ylim((y_min, y_max_value * 1.1))
        if error_to_plot_id == "kendals_tau" or error_to_plot_id == "precision_at_10":
            ax.set_ylim((0, 1))
        if y_max_manual is not None:
            ax.set_ylim((y_min, y_max_manual))
        ax.set_ylabel(f"{ERROR_NAME_DICT[error_to_plot_id]}")
    else:  # make y axis log scale
        ax.set_yscale("log")
        # add (log scale) to y label
        ax.set_ylabel(f"{ERROR_NAME_DICT[error_to_plot_id]} (log scale)")

    # set x axis limits to 10% of max value

    try:
        GAME_NAME = GAME_NAME_DICT[GAME_NAME]
    except KeyError:
        warnings.warn(
            f"Game name {GAME_NAME} not found in GAME_NAME_DICT. Using {GAME_NAME} instead."
        )
        GAME_NAME = GAME_NAME

    INTERACTION_INDEX_PRINT = INTERACTION_INDEX
    if INTERACTION_INDEX == "nSII":
        INTERACTION_INDEX_PRINT = r"$k$-SII"

    title = (
        f"{INTERACTION_INDEX_PRINT}: {GAME_NAME} ("
        + rf"$n = {N_PLAYER}$"
        + rf", {NUMBER_OF_RUNS} runs"
        + rf", $k = {ORDER}$"
        + ")"
    )
    ax.set_title(title, fontsize="xx-large")

    ax.set_ylim((y_min_manual, None))

    if not log_scale and science_scale:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()

    x_ticklabels_abs = ax.get_xticks()
    if N_PLAYER <= 20:
        x_tick_relative = [
            x_ticklabel / 2**N_PLAYER for x_ticklabel in x_ticklabels_abs
        ]
        x_ticklabels = [
            f"{abs_:.0f}\n{rel_:.2f}"
            for abs_, rel_ in zip(x_ticklabels_abs, x_tick_relative)
        ]
        x_label = "model evaluations (absolute, relative)"
    else:
        x_ticklabels = [f"{abs_:.0f}" for abs_ in x_ticklabels_abs]
        x_label = "model evaluations"
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label)

    plt.tight_layout()  # for good measure

    # save figure as pdf
    if SAVE_FIG:
        orders_to_plot_str = "-".join([str(order) for order in orders_to_plot])
        if GAME_NAME == "SOUM":
            GAME_NAME += f"_{N_PLAYER}"
        save_name = (
            f"{INTERACTION_INDEX}_{GAME_NAME}_k-{ORDER}_orders-{orders_to_plot_str}_{error_to_plot_id}"
            + SAVE_PATH_ADDENDUM
            + ".pdf"
        )
        os.makedirs(os.path.join("plots", "figures"), exist_ok=True)
        save_path = os.path.join("plots", "figures", save_name)
        # check if save path exists if yes add y_max_value to name in scientific notation
        if os.path.exists(save_path):
            print(f"Save path {save_path} already exists.")
            save_name = save_name.replace(".pdf", f"_{y_max_value:.2e}.pdf")
            save_path = os.path.join("plots", "figures", save_name)
        plt.savefig(save_path)
    plt.show()
