"""This script moves and aggregates the results from the soum experiments or the normal approximation runs."""
import json
import os
import shutil

import numpy as np
import pandas as pd

if __name__ == "__main__":
    result_dir = "results"
    results_plot_dir = "results_plot"

    if not os.path.exists(result_dir):
        raise FileNotFoundError(
            f"result_dir {result_dir} does not exist. First run the experiments. See README.md"
        )

    # create results_plot folder if it does not exist
    if not os.path.exists(results_plot_dir):
        os.makedirs(results_plot_dir)

    # aggregate and move SOUM results --------------------------------------------------------------
    all_expermients_dir = {}

    # get all soum games folders containing "soum" in results
    soum_folders = [
        os.path.join(result_dir, folder)
        for folder in os.listdir(result_dir)
        if "soum" in folder
    ]

    for folder in soum_folders:
        for subfolder in os.listdir(folder):
            for interaction_index in os.listdir(os.path.join(folder, subfolder)):
                # each experiment is placed in a timestamped folder
                exp_folder = os.path.join(folder, subfolder, interaction_index)
                n_runs = len(os.listdir(exp_folder))
                if n_runs == 0:
                    continue
                # load the first run to get the experiment results
                run_folder = os.path.join(exp_folder, os.listdir(exp_folder)[0])
                first_file_name = os.path.join(run_folder, os.listdir(run_folder)[0])
                with open(first_file_name, "r") as f:
                    run_result = json.load(f)
                # get the experiment name

                first_df = pd.read_json(first_file_name)

                # get experiment_variables

                orders = [i for i in first_df.index if i > 0]
                approximators = list(first_df.columns)
                budgets = [
                    first_df.iloc[0][0][i]["budget"]
                    for i in range(len(first_df.iloc[0][0]))
                ]
                quality_metrics = [
                    "approximation_error",
                    "precision_at_10",
                    "approximation_error_at_10",
                    "kendals_tau",
                    "variance",
                ]

                # get the save paths
                soum_name = folder.split("\\")[-1]
                save_folder = os.path.join(
                    results_plot_dir, soum_name, subfolder, interaction_index
                )
                save_name = first_file_name.split("\\")[-1]
                # replace "runs-1" with "runs-{n_runs}"
                save_name = save_name.replace("runs-1", f"runs-{n_runs}")
                save_path = os.path.join(save_folder, save_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # check if save_path exists
                if os.path.exists(save_path):
                    print(f"save_path {save_path} already exists")
                    continue

                aggregated_results = {
                    approximator: {} for approximator in approximators
                }
                for order in orders:
                    aggregated_results[order] = []

                approx_errors = {approximator: {} for approximator in approximators}
                for approximator in approximators:
                    for order in orders:
                        approx_errors[approximator][order] = {}
                        for quality_metric in quality_metrics:
                            approx_errors[approximator][order][
                                quality_metric
                            ] = np.zeros((n_runs, len(budgets)))

                for run_i, run in enumerate(os.listdir(exp_folder)):
                    run_folder = os.path.join(exp_folder, run)
                    run_file = os.path.join(run_folder, os.listdir(run_folder)[0])
                    df = pd.read_json(run_file)
                    for order in orders:
                        for approximator in approximators:
                            for budget_i, budget in enumerate(budgets):
                                for quality_metric in quality_metrics:
                                    approx_error = df.loc[order][approximator][
                                        budget_i
                                    ]["mean"][quality_metric]
                                    approx_errors[approximator][order][quality_metric][
                                        run_i
                                    ][budget_i] = approx_error

                summaries = ["mean", "median", "std", "var", "q_1", "q_3"]
                for approximator in approximators:
                    for order in orders:
                        quality_metric_dict = {}
                        for quality_metric in quality_metrics:
                            raw_errors = approx_errors[approximator][order][
                                quality_metric
                            ]
                            mean = np.mean(raw_errors, axis=0)
                            median = np.median(raw_errors, axis=0)
                            std = np.std(raw_errors, axis=0)
                            var = np.var(raw_errors, axis=0)
                            q_1 = np.quantile(raw_errors, 0.25, axis=0)
                            q_3 = np.quantile(raw_errors, 0.75, axis=0)

                            errors_budget = []
                            for budget_i, budget in enumerate(budgets):
                                dict_to_add = {
                                    "mean": mean[budget_i],
                                    "median": median[budget_i],
                                    "std": std[budget_i],
                                    "var": var[budget_i],
                                    "q_1": q_1[budget_i],
                                    "q_3": q_3[budget_i],
                                }
                                errors_budget.append(dict_to_add)
                            quality_metric_dict[quality_metric] = errors_budget
                        aggregated_results[approximator][order] = quality_metric_dict

                result_to_save = {}

                # transpose qualtiy metric and summary
                for approximator in approximators:
                    result_to_save[approximator] = {}
                    for order in orders:
                        result_to_save[approximator][order] = []
                        for budget_i, budget in enumerate(budgets):
                            add_dict = {"budget": budget}
                            for summary in summaries:
                                for quality_metric in quality_metrics:
                                    to_add_element = aggregated_results[approximator][
                                        order
                                    ][quality_metric][budget_i][summary]
                                    try:
                                        add_dict[summary][
                                            quality_metric
                                        ] = to_add_element
                                    except KeyError:
                                        add_dict[summary] = {
                                            quality_metric: to_add_element
                                        }
                            result_to_save[approximator][order].append(add_dict)

                # save results
                pd.DataFrame(result_to_save).to_json(save_path)

    # move remaining files (not SOUM) --------------------------------------------------------------
    all_expermients_dir = {}

    # get all files that are not from the SOUMs "soum" in results
    run_folders = [
        os.path.join(result_dir, folder)
        for folder in os.listdir(result_dir)
        if "soum" not in folder
    ]

    # copy the files to the results_plot folder similarly to the SOUMs above but without the aggregation if they do not exist
    for folder in run_folders:
        for subfolder in os.listdir(folder):
            for interaction_index in os.listdir(os.path.join(folder, subfolder)):
                # each experiment is placed in a timestamped folder
                exp_folder = os.path.join(folder, subfolder, interaction_index)
                folder_name_without_results = os.path.join(
                    *exp_folder.split(os.sep)[1:-1]
                )

                try:
                    file_name = os.listdir(exp_folder)[0]
                except IndexError:
                    print(f"folder {exp_folder} is empty")
                    continue
                file_path = os.path.join(exp_folder, file_name)

                # create the save_path in the results_plot folder without the timestamp
                save_path = os.path.join(
                    results_plot_dir, folder_name_without_results, file_name
                )

                # check if save_path exists
                if os.path.exists(save_path):
                    print(f"save_path {save_path} already exists")
                    continue

                # create the folder if it does not exist
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # copy the file
                shutil.copyfile(file_path, save_path)
