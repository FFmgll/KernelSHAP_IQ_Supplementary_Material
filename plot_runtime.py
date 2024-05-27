"""This script plots the runtime of all approximators as measured in nlp_runtime.csv"""

import os
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

COLORS = {
    "SHAP-IQ": "#ef27a6",
    "Permutation": "#7d53de",
    "SVARM-IQ": "#00b4d8",
    "KernelSHAP-IQ": "#ff6f00",
}

APPROX_PLOT_ORDER = ["Permutation", "SHAP-IQ", "SVARM-IQ", "KernelSHAP-IQ"]

if __name__ == "__main__":
    params = {
        "legend.fontsize": "x-large",
        "figure.figsize": (7, 5),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    plt.rcParams.update(params)

    # cols = budget, iteration, explainer, runtime
    df = pd.read_csv(os.path.join("results_plot", "runtime", "nlp_runtime.csv"))

    n_iterations = df["iteration"].max() + 1

    # get the mean over all iterations for each explainer and budget
    df_mean = df.groupby(["budget", "explainer"]).mean().reset_index()

    # get the std error over all iterations for each explainer and budget
    df_sem = df.groupby(["budget", "explainer"]).sem().reset_index()

    # plot the results
    fig, ax = plt.subplots()
    for explainer in APPROX_PLOT_ORDER:
        df_explainer = df_mean[df_mean["explainer"] == explainer]
        df_explainer_std = df_sem[df_sem["explainer"] == explainer]
        ax.plot(
            df_explainer["budget"],
            df_explainer["runtime"],
            label=explainer,
            color=COLORS[explainer],
            marker="o",
        )
        ax.fill_between(
            df_explainer["budget"],
            df_explainer["runtime"] - df_explainer_std["runtime"],
            df_explainer["runtime"] + df_explainer_std["runtime"],
            alpha=0.2,
            color=COLORS[explainer],
        )
    ax.set_xlabel(r"Budget / Accesses to the LM")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime of Approximators for the LM (Averaged over {n_iterations} Runs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "appendix", "nlp_runtime.pdf"))
    plt.show()
