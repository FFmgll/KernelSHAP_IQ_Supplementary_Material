"""This Module gathers game functions that can be used to test the approximation methods"""
import copy
import os
import pickle
import random
import math
import typing
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import binom
from tqdm import tqdm

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.exceptions import NotFittedError
    from transformers import pipeline
except ImportError:
    pass

from approximators.base import powerset
from .datasets_own import Adult, BikeSharing, StudentPerformance, Splice


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


class MachineLearningMetaGame:
    def __init__(
        self,
        model=None,
        input_dataset=None,
        dataset_name=None,
        random_seed=42,
        n=None,
        regression=False,
    ):
        if input_dataset is not None:
            self.x_data = input_dataset["x_data"]
            self.y_data = input_dataset["y_data"]
        else:
            if dataset_name == "adult":
                print("Selecting adult.")
                dataset = Adult(random_seed=random_seed)
            elif dataset_name == "bike":
                print("Selecting bike.")
                dataset = BikeSharing(random_seed=random_seed)
            elif dataset_name == "student":
                dataset = StudentPerformance(random_seed=random_seed)
            elif dataset_name == "splice":
                dataset = Splice(random_seed=random_seed)
            else:
                raise NotImplementedError("Only adult or bike implemented.")
            self.x_data = dataset.x_data.values
            self.y_data = dataset.y_data.values

        self.regression = regression
        if n is not None:
            self.x_data = self.x_data[:, 0:n]

        self.n_samples = len(self.x_data)
        self.replacement_values = np.mean(self.x_data, axis=0).reshape(1, -1)
        self.n = len(self.replacement_values[0])
        self.model = model
        if self.model is None:
            if not self.regression:
                self.model = RandomForestClassifier()
                self.model.fit(self.x_data, self.y_data)
            else:
                self.model = RandomForestRegressor()
                self.model.fit(self.x_data, self.y_data)
        # check if sklearn model is fitted if not fit it on the data
        try:
            model.predict(self.x_data[0:2])
        except NotFittedError:
            self.model.fit(self.x_data, self.y_data)
        self.empty_value = self.call_model(self.replacement_values, set())

    def call_model(self, x_i: np.ndarray, S: set):
        x_input = np.zeros(shape=(1, self.n))
        x_input[:, :] = self.replacement_values[:]
        x_input[:, tuple(S)] = x_i[:, tuple(S)]
        if self.regression:
            output = self.model.predict(x_input)[0]
        else:
            output = self.model.predict_proba(x_input)[0][1]
        return output


class MachineLearningGame:
    def __init__(
        self, meta_game: MachineLearningMetaGame, data_index: int = None, set_zero: bool = False
    ):
        if data_index is None:
            data_index = random.randint(0, meta_game.n_samples - 1)
        assert data_index <= meta_game.n_samples - 1, "Not enough data in this dataset."

        self.meta_game = meta_game
        self.data_point = meta_game.x_data[data_index]
        self.data_point = self.data_point.reshape(1, -1)

        self.empty_value = 0
        if set_zero:
            self.empty_value = self.meta_game.empty_value

        self.n = meta_game.n
        self.game_name = "tabular model"

    def set_call(self, S):
        output = self.meta_game.call_model(x_i=self.data_point, S=S)
        return output - self.empty_value


class NLPGame:
    """Basic NLP Game, where a sentiment analysis model given a sentence is explained.

    Players: the tokenized words in the sentence.
    Output: in range -1 to 1 the sentiment of the subset.
    """

    def __init__(self, input_text: str, label_key: str = "label", set_zero: bool = True):
        self.classifier = pipeline(model="lvwerra/distilbert-imdb", task="sentiment-analysis")
        self.tokenizer = self.classifier.tokenizer
        self.tokenized_input = np.asarray(self.tokenizer(input_text)["input_ids"][1:-1])
        self.input_sentence = self.tokenizer.decode(self.tokenized_input)
        self.n = len(self.tokenized_input)
        self._label_key = label_key
        self.game_name = "language_model"
        self.original_output = self.call(self.input_sentence)
        self.empty_value = 0
        self.original_empty_value = self.set_call(set())
        if set_zero:
            self.empty_value = self.original_empty_value

    def call(self, x):
        outputs = self.classifier(x)
        outputs = [
            output["score"] * 1 if output[self._label_key] == "POSITIVE" else output["score"] * -1
            for output in outputs
        ]
        return outputs

    def set_call_iterable(self, S):
        x_inputs = []
        for i, s in enumerate(S):
            token_subset = self.tokenized_input[s]
            x_text = self.tokenizer.decode(token_subset)
            x_inputs.append(x_text)
        return self.call(x_inputs)

    def set_call(self, S):
        token_subset = self.tokenized_input[list(S)]
        x_text = self.tokenizer.decode(token_subset)
        return self.call(x_text)[0] - self.empty_value


class NLPLookupGame:
    """Wrapper for the NLP Game to use precomputed model outputs for faster runtime in experimental
    settings."""

    def __init__(
        self, n: int, sentence_id: int = None, used_ids: set = None, set_zero: bool = True
    ):
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids
        if sentence_id is None:
            files = os.listdir(os.path.join("data", "nlp_values", str(n)))
            files = list(set(files) - used_ids)
            if len(files) == 0:
                files = os.listdir(os.path.join("data", "nlp_values", str(n)))
                self.used_ids = set()
            sentence_id = random.choice(files)
            sentence_id = int(sentence_id.split(".")[0])
        self.used_ids.add(str(sentence_id) + ".csv")
        data_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "nlp_values",
            str(n),
            str(sentence_id) + ".csv",
        )
        self.df = pd.read_csv(data_path)
        self.game_name = "language_model"
        self.n = n
        whole_data = pd.read_csv(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "simplified_imdb.csv")
        )
        self.input_sentence = str(whole_data[whole_data["id"] == sentence_id]["text"].values[0])

        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = "s_" + "_".join([str(player) for player in sorted(S)])
        return self.storage[S_id] - self.empty_value


class TabularLookUpGame:
    """Wrapper for the Machine Learning Game to use precomputed model outputs for faster runtime in
    experimental settings."""

    def __init__(
        self,
        n: int,
        data_folder: str = "adult_42",
        data_id: int = None,
        used_ids: set = None,
        set_zero: bool = True,
    ):
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids
        if data_id is None:
            files = os.listdir(os.path.join("data", data_folder, str(n)))
            files = list(set(files) - used_ids)
            if len(files) == 0:
                files = os.listdir(os.path.join("data", data_folder, str(n)))
                self.used_ids = set()
            data_id = random.choice(files)
            data_id = int(data_id.split(".")[0])
        self.used_ids.add(str(data_id) + ".csv")
        data_path = os.path.join("data", data_folder, str(n), str(data_id) + ".csv")
        self.df = pd.read_csv(data_path)
        self.game_name = "tabular_game"
        self.n = n

        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = "s_" + "_".join([str(player) for player in sorted(S)])
        return self.storage[S_id] - self.empty_value


class BaseSparseLinearModel:
    """Base class. Do not use this class but the derived one.

    Synthetic Linear Function where you know the Shapley values and interaction terms beforehand.
    To be used to create high dimensional data with ground truths.

    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """

    def __init__(
        self,
        n: int = 10,
        n_non_important_features: int = 0,
        n_interactions_per_order: typing.Dict[int, int] = None,
    ):
        assert n_non_important_features <= n, (
            f"Total number of non-important features {n_non_important_features} "
            f"must be smaller than number features {n}."
        )
        self.game_name = "sparse_linear_model"
        self.n = n
        n_important = n - n_non_important_features
        N_important = np.asarray(list(range(0, n_important)))
        self.interaction_weights = {}
        if n_interactions_per_order is not None:
            for interaction_order, n_interactions in n_interactions_per_order.items():
                interacting_features = set()
                if math.comb(n_important, interaction_order) < n_interactions:
                    raise ValueError(
                        f"The number of interaction per order {interaction_order} is {n_interactions} "
                        f"exceeding the total number of combinations "
                        f"({math.comb(n_important, interaction_order)}). Use a value smaller or equal to "
                        f"{math.comb(n_important, interaction_order)}."
                    )
                while (
                    len(interacting_features) < n_interactions
                ):  # might stall at certain parameters
                    interaction_sample = tuple(
                        sorted(np.random.choice(N_important, size=interaction_order, replace=False))
                    )
                    interacting_features.add(interaction_sample)
                interaction_weights = list(np.random.rand(len(interacting_features)))
                for interaction_feature_pair, interaction_weight in zip(
                    interacting_features, interaction_weights
                ):
                    self.interaction_weights[interaction_feature_pair] = interaction_weight
        self.N = np.asarray(list(range(0, self.n)))
        try:
            self._highest_interaction_order = max(n_interactions_per_order.keys())
        except AttributeError:
            self._highest_interaction_order = 0
        self.gamma_matrix = self._get_gamma_weights(n, s=1)

    # OLD EXACT CALCULATION
    def exact_values_old(self, gamma_matrix, min_order, max_order):
        results = {}
        for s in range(min_order, max_order + 1):
            results[s] = np.zeros(np.repeat(self.n, s))
            for subset, weight in self.interaction_weights.items():
                q = len(subset)
                for S in powerset(self.N, s, s):
                    r = len(set(subset).intersection(S))
                    results[s][S] += weight * self.coefficient_weighting(gamma_matrix, s, q, r)
        return results

    def exact_values(self, gamma_matrix=None, min_order=2, max_order=2):
        if gamma_matrix is None:
            gamma_matrix = self.gamma_matrix
        results = {}
        # pre-compute weights in matrix: order x interaction set sizes x intersection set sizes
        exact_value_weights = np.zeros((max_order + 1, self.n + 1, max_order + 1))
        pbar_budget_precomputation = (
            (max_order - min_order + 1) * len(self.allowed_interaction_sizes) * (max_order + 1)
        )
        pbar = tqdm(total=pbar_budget_precomputation, desc="Exact values: pre-computed weights")
        for s in range(min_order, max_order + 1):
            for q in self.allowed_interaction_sizes:
                for r in range(0, max_order + 1):
                    exact_value_weights[s, q, r] = self.coefficient_weighting(gamma_matrix, s, q, r)
                    pbar.update(1)
        pbar.close()

        number_of_interactions = 0
        for s in range(min_order, max_order + 1):
            number_of_interactions += binom(self.n, s)

        pbar_budget_final = (
            (max_order - min_order + 1) * len(self.interaction_weights) * number_of_interactions
        )
        pbar = tqdm(total=pbar_budget_final, desc="Exact values: Final computation")
        for s in range(min_order, max_order + 1):
            results[s] = np.zeros(np.repeat(self.n, s))
            for subset, weight in self.interaction_weights.items():
                q = len(subset)
                for S in powerset(self.N, s, s):
                    r = len(set(subset).intersection(S))
                    results[s][S] += weight * exact_value_weights[s, q, r]
                    pbar.update(1)
        pbar.close()
        return results

    def exact_values_sii(self, min_order=2, max_order=2):
        results = {}
        number_of_interactions = 0
        for s in range(min_order, max_order + 1):
            number_of_interactions += binom(self.n, s)

        pbar_budget_final = (
            (max_order - min_order + 1) * len(self.interaction_weights) * number_of_interactions
        )
        pbar = tqdm(total=pbar_budget_final, desc="Exact values: Final computation")
        for s in range(min_order, max_order + 1):
            results[s] = np.zeros(np.repeat(self.n, s))
            for subset, weight in self.interaction_weights.items():
                q = len(subset)
                if q >= s:
                    for S in powerset(self.N, s, s):
                        r = len(set(subset).intersection(S))
                        if r == s:
                            results[s][S] += weight / (q - s + 1)
                        pbar.update(1)
        pbar.close()
        return results

    def exact_values_one(self, gamma_matrix, min_order, max_order, interaction):
        results = {}
        # pre-compute weights in matrix: order x interaction set sizes x intersection set sizes
        exact_value_weights = np.zeros((max_order + 1, self.n + 1, max_order + 1))
        pbar_budget_precomputation = (
            (max_order - min_order + 1) * len(self.allowed_interaction_sizes) * (max_order + 1)
        )
        pbar = tqdm(total=pbar_budget_precomputation, desc="Exact values: pre-computed weights")
        for s in range(min_order, max_order + 1):
            for q in self.allowed_interaction_sizes:
                for r in range(0, max_order + 1):
                    exact_value_weights[s, q, r] = self.coefficient_weighting(gamma_matrix, s, q, r)
                    pbar.update(1)
        pbar.close()

        number_of_interactions = 0
        for s in range(min_order, max_order + 1):
            number_of_interactions += binom(self.n, s)

        for s in range(min_order, max_order + 1):
            results[s] = {}
            results[s][interaction] = 0
            for subset, weight in self.interaction_weights.items():
                q = len(subset)
                r = len(set(subset).intersection(interaction))
                results[s][interaction] += weight * exact_value_weights[s, q, r]
        return results

    def coefficient_weighting(self, gamma_matrix, s, q, s_cap_q):
        rslt = 0
        for t in range(q, self.n + 1):
            add = min(t - q, s - s_cap_q)
            for l in range(add + 1):
                rslt += (
                    binom(self.n - q - (s - s_cap_q), t - q - l)
                    * binom(s - s_cap_q, l)
                    * gamma_matrix[t, l + s_cap_q]
                )
        return rslt

    def call(self, x):
        output = sum(
            [
                np.prod(x[[*features]]) * interaction_weight
                for features, interaction_weight in self.interaction_weights.items()
            ]
        )
        return output

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)

    def _get_gamma_weights(self, n, s):
        weights = np.zeros((n + 1, s + 1))
        for t in range(0, n + 1):
            for k in range(max(0, s + t - n), min(s, t) + 1):
                weights[t, k] = (-1) ** (s - k) * self._kernel_m(n, t - k, s)
        return weights

    @staticmethod
    def _kernel_m(n, t, s):
        return np.math.factorial(n - t - s) * np.math.factorial(t) / np.math.factorial(n - s + 1)


class ParameterizedSparseLinearModel(BaseSparseLinearModel):
    """Synthetic Linear Function where you know the Shapley values and interaction terms beforehand.
    To be used to create high dimensional data with ground truths.

    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """

    def __init__(
        self,
        n,
        weighting_scheme,
        n_interactions,
        max_interaction_size=-1,
        min_interaction_size=1,
        n_non_important_features=0,
    ):
        self.n_non_important_features = n_non_important_features
        if max_interaction_size == -1:
            max_interaction_size = n
        max_interaction_size = min(n - n_non_important_features, max_interaction_size)
        weighting_ratios = np.zeros(n + 1)
        allowed_interaction_sizes = np.arange(min_interaction_size, max_interaction_size + 1)
        self.allowed_interaction_sizes = allowed_interaction_sizes
        for k in allowed_interaction_sizes:
            if weighting_scheme == "uniform":
                weighting_ratios[k] += 1
            if weighting_scheme == "center":
                weighting_ratios[k] += binom(n, k)
            if weighting_scheme == "tail":
                weighting_ratios[k] += 1 / binom(n, k)
        weighting_ratios /= np.sum(weighting_ratios)
        interaction_sizes = random.choices(
            allowed_interaction_sizes,
            k=n_interactions,
            weights=weighting_ratios[allowed_interaction_sizes],
        )
        n_interactions_per_order = Counter(interaction_sizes)
        for k in allowed_interaction_sizes:
            n_interactions_per_order[k] = min(
                n_interactions_per_order[k], binom(n - n_non_important_features, k)
            )
        super().__init__(
            n=n,
            n_interactions_per_order=n_interactions_per_order,
            n_non_important_features=n_non_important_features,
        )


class SimpleGame:
    """Very simple game where each feature of a linear model has a random weight and there exist one
    2nd-order interaction and one 3rd-order interaction.

    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """

    def __init__(self, n):
        self.weights = np.random.rand(n)
        self.n = n
        self.intx2 = random.random()
        self.intx3 = 0

    def call(self, x):
        return np.dot(x, self.weights) + x[1] * x[2] * self.intx2 + self.intx3 * x[1] * x[2] * x[3]

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class LinearModelWithCrosses:
    """A simple synthetic game presented in
    Sundararajan et al. (2020). The Shapley Taylor Interaction Index
    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """

    def __init__(self, n, order, variant="STI_example", c=None, l=None):
        self.n = n
        self.order = order
        if l == None:
            self.l = n
        else:
            self.l = l
        if c == None:
            self.c = random.random()
        else:
            self.c = c
        self.game_name = "linear_model_with_crosses"
        self.variant = variant

    def call(self, x):
        if self.variant == "STI_example":
            return np.sum(x) + self.c * np.prod(x[: self.l + 1])
        if self.variant == "STI_example_fix4":
            return np.sum(x) + self.c * x[0] * x[1] * x[2] * x[3] * x[4]
        if self.variant == "FSI_n_shapley_m2":
            return np.sum(x) + x[0] * x[1] + x[0] * x[2]
        if self.variant == "FSI_n_shapley_m3":
            return np.sum(x) + x[0] * x[1] + x[0] * x[2] + x[0] * x[1] * x[2]

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class MarginalUtility:
    """A simple synthetic game presented in
    Faith-Shap
    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """

    def __init__(self, n, p, example):
        self.n = n
        self.game_name = "diminishing_marginal_utility"
        self.p = p
        self.example = example

    def call(self, x):
        num_players = np.sum(x)
        if self.example == 1:
            if num_players <= 1:
                return 0
            else:
                return num_players - self.p * binom(num_players, 2)
        if self.example == 2:
            if num_players == 0:
                return 0
            if num_players == 1:
                return 3
            if num_players > 1:
                return 3 * num_players - (num_players - 2 * np.log(num_players + 1))

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class SyntheticNeuralNetwork:
    """Synthetic Neural Network
    To be used to have a function behaving like a neural network with custom controls

    Players: the input features (zero or one)
    Output: classification score between 0 and 1
    """

    def __init__(self, n, set_zero=True):
        self.n = n
        self.game_name = "synth_neural_network"
        self.weights_1 = np.random.normal(loc=0, scale=10, size=(100, self.n))
        self.bias_1 = np.random.normal(loc=0, scale=1, size=100)
        self.weights_2 = np.random.normal(loc=0, scale=0.5, size=(10, 100))
        self.bias_2 = np.random.normal(loc=0, scale=1)
        self.weights_3 = np.random.normal(loc=0, scale=0.05, size=(10, 10))
        self.bias_3 = np.random.normal(loc=0, scale=0.05)
        self.weights_4 = np.random.normal(loc=0, scale=0.05, size=(1, 10))
        self.bias_4 = np.random.normal(loc=0, scale=0.05)
        self.empty_val = 0
        if set_zero:
            self.empty_val = self.set_call({})

    def call(self, x):
        x = np.maximum(0, np.dot(self.weights_1, x) + self.bias_1)
        x = np.maximum(0, np.dot(self.weights_2, x) + self.bias_2)
        x = np.maximum(0, np.dot(self.weights_3, x) + self.bias_3)
        x = np.dot(self.weights_4, x) + self.bias_4
        y = _sigmoid(x)
        return y

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x) - self.empty_val


class ConvergenceGame:
    def __init__(
        self,
        n: int = 20,
        top_order: int = 2,
        data_id: str = None,
        approximation_method: str = "SVARM-IQ",
        set_zero: bool = True,
        used_ids: set = None,
    ):
        self.n = n
        self.game_name = "_".join(("convergence_game", str(n)))
        self.top_order = top_order
        self.set_zero = set_zero
        data_folder = os.path.join("precomputed_nlp_interactions", approximation_method)
        files = os.listdir(data_folder)
        files = [file for file in files if file.endswith(".pkl")]
        files = [file for file in files if file.split("_")[1] == str(n)]
        files = [file for file in files if file.split("order_")[1].split("_")[0] == str(top_order)]
        files_per_sentence = {}
        for file in files:
            sentence_id = file.split("_")[0]
            if sentence_id not in files_per_sentence:
                files_per_sentence[sentence_id] = []
            files_per_sentence[sentence_id].append(file)

        if used_ids is None:
            used_ids = set()

        sentences = list(files_per_sentence.keys())
        sentences = list(set(sentences) - used_ids)
        if len(sentences) == 0:
            sentences = list(files_per_sentence.keys())
            used_ids = set()

        if data_id is None:
            data_id = random.choice(sentences)

        used_ids.add(data_id)
        self.used_ids = used_ids
        self.data_id = data_id

        gt_values = None
        input_sentence = ""
        n_files = len(files_per_sentence[self.data_id])
        for file in files_per_sentence[self.data_id]:
            with open(os.path.join(data_folder, file), "rb") as pickle_file:
                data = pickle.load(pickle_file)
                if gt_values is None:
                    gt_values = copy.deepcopy(data["sii_values"])
                    input_sentence = data["sentence"]
                else:
                    for order in gt_values.keys():
                        gt_values[order] += data["sii_values"][order]
        for order in gt_values.keys():
            gt_values[order] /= n_files
        self.gt_values = gt_values
        self.input_sentence = input_sentence

        self.nlp_game = NLPGame(
            input_text=self.input_sentence,
            set_zero=self.set_zero,
        )

    def set_call(self, S):
        return self.nlp_game.set_call(S=S)

    def exact_values(self):
        return self.gt_values


class LookUpGame:
    """Wrapper for the Machine Learning Game to use precomputed model outputs for faster runtime in
    experimental settings."""

    def __init__(
        self,
        data_folder: str,
        n: int,
        data_id: typing.Union[int, str] = None,
        used_ids: set = None,
        set_zero: bool = True,
        log_output: bool = False,
        min_max_normalize: bool = False,
        random_seed: int = None,
    ):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.n = n
        self.log_output = log_output

        # to not use the same instance twice if we use the game multiple times
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids

        # get paths to the files containing the value function calls
        game_path = Path(__file__).parent.absolute()
        data_dir = os.path.join(game_path, "data")
        instances_dir = os.path.join(data_dir, data_folder, str(n))

        # randomly select a file if none was explicitly provided
        if data_id is None:
            files = os.listdir(instances_dir)
            files = list(set(files) - used_ids)
            files = sorted([file for file in files if file.endswith(".csv")])
            if len(files) == 0:
                files = os.listdir(instances_dir)
                self.used_ids = set()
            # select random file with seed
            data_id = random.choice(files)
            data_id = data_id.split(".")[0]
        self.data_id = str(data_id)
        self.game_name = "_".join(("LookUpGame", data_folder, str(n), self.data_id))

        self.used_ids.add(self.data_id + ".csv")

        # load file containing value functions into easily accessible dict
        file_path = os.path.join(instances_dir, self.data_id + ".csv")
        self.df = pd.read_csv(file_path)
        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        if min_max_normalize:
            # normalize values to [0,1]
            storage_df = pd.DataFrame.from_dict(self.storage, orient="index")
            storage_df[0] = (storage_df[0] - storage_df[0].min()) / (
                storage_df[0].max() - storage_df[0].min()
            )
            self.storage = storage_df.to_dict()[0]

        # normalize empty coalition to zero (v_0({}) = 0)
        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = "s_" + "_".join([str(player) for player in sorted(S)])
        output = self.storage[S_id] - self.empty_value
        if self.log_output:
            return float(np.log(output))
        return output

    def get_name(self):
        return self.game_name
