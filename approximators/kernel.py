"""This module contains the generalized KernelEstimator class, which is used to approximate SII using the weighted least square approach."""
import copy
import itertools
import random

import numpy as np
from scipy.special import binom, bernoulli

from approximators import BaseShapleyInteractions
from approximators.base import determine_complete_subsets, powerset


def get_weights(num_players):
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)
    return weights


class KernelSHAPIQEstimator(BaseShapleyInteractions):
    """Estimates the SI (for FSI) using the weighted least square approach"""

    def __init__(
        self,
        N,
        order,
        interaction_type="SII",
        boosting=True,
        approximator_mode="default",
        big_m=1_000_000,
    ):
        min_order = 1
        super().__init__(N, order, min_order=min_order)
        self._big_M = float(big_m)
        assert interaction_type in ("SII")
        self.interaction_type = interaction_type
        self.max_order = order
        self._bernoulli_numbers = bernoulli(self.n)
        self._boosting = boosting
        assert approximator_mode in ["full", "default", "inconsistent"]
        self._mode = approximator_mode

    def _init_kernel_weights(self, s):
        # vector that determines the kernel weights for KernelSHAPIQ
        weight_vector = np.zeros(shape=self.n + 1)
        for subset_size in range(0, self.n + 1):
            if (subset_size == s - 1) or (subset_size == self.n - s + 1):
                weight_vector[subset_size] = self._big_M * binom(self.n, subset_size)
            elif (subset_size < s) or (subset_size > self.n - s):
                weight_vector[subset_size] = self._big_M * binom(self.n, subset_size)
            else:
                weight_vector[subset_size] = binom(self.n, subset_size) / (
                    (self.n - s + 1) * binom(self.n - 2 * s, subset_size - s)
                )
        kernel_weight = weight_vector  # / np.sum(weight_vector)
        return kernel_weight

    def _init_sampling_weights(self):
        # vector that is proportional to the probability of sampling a subset of that size
        weight_vector = np.zeros(shape=self.n + 1)
        for subset_size in range(0, self.n + 1):
            if (subset_size < self.max_order) or (subset_size > self.n - self.max_order):
                # prioritize these subsets
                weight_vector[subset_size] = self._big_M
            else:
                # KernelSHAP sampling weights
                weight_vector[subset_size] = 1 / (subset_size * (self.n - subset_size))
                # weight_vector[subset_size] = binom(self.n, subset_size) / (
                #    (self.n) * binom(self.n - 2, subset_size - 1))
        sampling_weight = weight_vector / np.sum(weight_vector)
        return sampling_weight

    @staticmethod
    def get_S_weights(
        num_players,
        subsets,
        subsets_counter,
        complete_subsets,
        incomplete_subsets,
        sampling_weights,
        n_sampled_subsets,
        weights,
        order,
    ):
        subset_weights = {s: {} for s in range(1, order + 1)}
        weight_left = np.sum(sampling_weights[incomplete_subsets])

        for S in subsets:
            if len(S) in complete_subsets:
                for s in range(1, order + 1):
                    subset_weights[s][tuple(sorted(S))] = weights[s][len(S)] / binom(
                        num_players, len(S)
                    )
            if len(S) in incomplete_subsets:
                S_counter = subsets_counter[tuple(sorted(S))]
                for s in range(1, order + 1):
                    subset_weights[s][tuple(sorted(S))] = (
                        weights[s][len(S)]
                        * S_counter
                        * weight_left
                        / (n_sampled_subsets * sampling_weights[len(S)])
                    )
        return subset_weights

    @staticmethod
    def get_S_and_values_double_counting(
        sampling_budget,
        num_players,
        complete_subsets,
        incomplete_subsets,
        sampling_weights,
        N,
        pairing,
        game_fun,
    ):
        # Samples subsets and evaluate the game, but counts sampled subsets and not evaluated subsets for budget
        all_subsets = []
        for complete_subset in complete_subsets:
            combinations = itertools.combinations(N, complete_subset)
            for subset in combinations:
                subset = set(subset)
                all_subsets.append(subset)

        remaining_weight = sampling_weights[incomplete_subsets] / sum(
            sampling_weights[incomplete_subsets]
        )
        subsets_counter = {}

        n_sampled_subsets = 0
        if len(incomplete_subsets) > 0:
            sampled_subsets = set()
            while n_sampled_subsets < sampling_budget:
                subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
                ids = np.random.choice(num_players, size=subset_size, replace=False)
                sampled_subset = tuple(sorted(ids))
                if sampled_subset not in sampled_subsets:
                    sampled_subsets.add(sampled_subset)
                    subsets_counter[sampled_subset] = 1.0
                else:
                    subsets_counter[sampled_subset] += 1.0
                n_sampled_subsets += 1
                if pairing:
                    if len(sampled_subsets) < sampling_budget:
                        sampled_subset_paired = tuple(sorted(set(N) - set(ids)))
                        if sampled_subset_paired not in sampled_subsets:
                            sampled_subsets.add(sampled_subset_paired)
                            subsets_counter[sampled_subset_paired] = 1.0
                        else:
                            subsets_counter[sampled_subset_paired] += 1.0
                        n_sampled_subsets += 1
            for subset in sampled_subsets:
                all_subsets.append(set(subset))

        # game evaluation should be called separately
        game_values = [game_fun(subset) for subset in all_subsets]
        return all_subsets, subsets_counter, n_sampled_subsets, game_values

    @staticmethod
    def get_S_and_values(
        sampling_budget,
        num_players,
        complete_subsets,
        incomplete_subsets,
        sampling_weights,
        N,
        pairing,
        game_fun,
    ):
        all_subsets = []
        for complete_subset in complete_subsets:
            combinations = itertools.combinations(N, complete_subset)
            for subset in combinations:
                subset = set(subset)
                all_subsets.append(subset)

        remaining_weight = sampling_weights[incomplete_subsets] / sum(
            sampling_weights[incomplete_subsets]
        )
        subsets_counter = {}

        n_sampled_subsets = 0
        if len(incomplete_subsets) > 0:
            sampled_subsets = set()
            while len(sampled_subsets) < sampling_budget:
                subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
                ids = np.random.choice(num_players, size=subset_size, replace=False)
                sampled_subset = tuple(sorted(ids))
                if sampled_subset not in sampled_subsets:
                    sampled_subsets.add(sampled_subset)
                    subsets_counter[sampled_subset] = 1.0
                else:
                    subsets_counter[sampled_subset] += 1.0
                n_sampled_subsets += 1
                if pairing:
                    if len(sampled_subsets) < sampling_budget:
                        sampled_subset_paired = tuple(sorted(set(N) - set(ids)))
                        if sampled_subset_paired not in sampled_subsets:
                            sampled_subsets.add(sampled_subset_paired)
                            subsets_counter[sampled_subset_paired] = 1.0
                        else:
                            subsets_counter[sampled_subset_paired] += 1.0
                        n_sampled_subsets += 1
            for subset in sampled_subsets:
                all_subsets.append(set(subset))

        # game evaluation should be called separately
        game_values = [game_fun(subset) for subset in all_subsets]
        return all_subsets, subsets_counter, n_sampled_subsets, game_values

    def _get_wlsq_solution_full(self, S_list, all_S, game_values, kernel_weights, order):
        max_approx_order = self.max_order
        min_approx_order = order

        num_players: int = 0
        start_this_order: int = 0
        end_this_order: int = 0
        for s in range(min_approx_order, max_approx_order + 1):
            if s == order:
                start_this_order = num_players
                num_players += int(binom(self.n, s))
                end_this_order = num_players
            else:
                num_players += int(binom(self.n, s))

        i = 0
        player_indices = {}
        player_indices_inv = {}
        for combination in powerset(self.N, max_size=max_approx_order, min_size=min_approx_order):
            player_indices[combination] = i
            player_indices_inv[i] = combination
            i += 1

        N_arr = np.arange(0, self.n)
        W = np.zeros(shape=game_values.shape, dtype=float)
        new_S = np.zeros(shape=(len(S_list), num_players), dtype=float)
        higher_order_indicator = np.zeros(shape=game_values.shape, dtype=float)

        for i, S in enumerate(all_S):
            if np.sum(S) >= order and np.sum(S) <= self.n - order:
                higher_order_indicator[i] = 1
            S_as_list = N_arr[S]
            W[i] = kernel_weights[tuple(sorted(S_as_list))]
            for combination in powerset(self.N, min_approx_order, max_approx_order):
                index = player_indices[combination]
                D, S_set = set(combination), set(S_as_list)
                size_S_cap_T = len(S_set.intersection(D))
                size_interaction_S = len(D)
                new_S_value = self._get_bernoulli_weights(
                    size_interaction_S=size_interaction_S, size_S_cap_T=size_S_cap_T
                )
                new_S[i, index] = new_S_value

        A = new_S
        B = game_values
        W_sqrt = np.sqrt(np.diag(W))
        W_diag = np.diag(W)

        # print(np.sum(0==higher_order_indicator))
        if order == 1:
            # solution to weighted least square are shapley interactions
            Aw = np.dot(W_sqrt, A)
            Bw = np.dot(B, W_sqrt)
            phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)
        else:
            AWA = np.dot(A.T, np.dot(W_diag, A))
            AWA_inverse = np.linalg.inv(AWA)
            final_weights = np.dot(AWA_inverse, np.dot(A.T, W_diag))
            true_weights = self._get_true_weights_all(
                all_S[higher_order_indicator == 0],
                N_arr,
                min_approx_order,
                max_approx_order,
                num_players,
            )
            phi_const = np.dot(true_weights.T, B[higher_order_indicator == 0])
            phi_kernel = np.dot(
                final_weights[:, higher_order_indicator == 1], B[higher_order_indicator == 1]
            )
            phi = phi_const + phi_kernel

        # only currently approximated order
        phi_current = phi[start_this_order:end_this_order]
        approximation = np.dot(A[:, start_this_order:end_this_order], phi_current)
        # approximation = np.dot(A,phi)
        # test = self._get_true_weights(all_S,N_arr,order)
        return phi_current, approximation

    def _get_wlsq_solution_inconsistent(self, S_list, all_S, game_values, kernel_weights):
        num_players: int = 0
        for s in range(1, self.max_order + 1):
            num_players += int(binom(self.n, s))

        i = 0
        player_indices = {}
        player_indices_inv = {}
        for combination in powerset(self.N, max_size=self.max_order, min_size=1):
            player_indices[combination] = i
            player_indices_inv[i] = combination
            i += 1

        N_arr = np.arange(0, self.n)
        W = np.zeros(shape=game_values.shape, dtype=float)
        new_S = np.zeros(shape=(len(S_list), num_players), dtype=float)

        for i, S in enumerate(all_S):
            S_as_list = N_arr[S]
            W[i] = kernel_weights[tuple(sorted(S_as_list))]
            for combination in powerset(self.N, 1, self.max_order):
                index = player_indices[combination]
                D, S_set = set(combination), set(S_as_list)
                size_S_cap_T = len(S_set.intersection(D))
                size_interaction_S = len(D)
                new_S_value = self._get_bernoulli_weights(
                    size_interaction_S=size_interaction_S, size_S_cap_T=size_S_cap_T
                )
                new_S[i, index] = new_S_value

        A = new_S
        B = game_values
        W_sqrt = np.sqrt(np.diag(W))
        W_diag = np.diag(W)
        # solution to weighted least square are shapley interactions
        Aw = np.dot(W_sqrt, A)
        Bw = np.dot(B, W_sqrt)
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)

        shapley_interactions = {}
        shapley_approximations = {}
        # Approximation only for maximum order, no boosting possible
        shapley_approximations[self.max_order] = np.dot(A, phi)

        counter = 0
        for s in range(1, self.max_order + 1):
            n_interactions = int(binom(self.n, s))
            shapley_interactions[s] = phi[counter : counter + n_interactions]
            counter += n_interactions

        return shapley_interactions, shapley_approximations

    def _get_wlsq_solution(self, S_list, all_S, game_values, kernel_weights):
        N_arr = np.arange(0, self.n)
        gt_indicator_dict = {}
        A_dict = {}
        W_dict = {}
        final_weights_dict = {}
        AWA_inverse_dict = {}

        for current_order in range(1, self.max_order + 1):
            num_players: int = int(binom(self.n, current_order))
            i = 0
            player_indices = {}
            player_indices_inv = {}
            for combination in powerset(self.N, max_size=current_order, min_size=current_order):
                player_indices[combination] = i
                player_indices_inv[i] = combination
                i += 1

            gt_indicator = np.zeros(shape=game_values.shape, dtype=bool)
            W = np.zeros(shape=game_values.shape, dtype=float)
            new_S = np.zeros(shape=(len(S_list), num_players), dtype=float)
            for i, S in enumerate(all_S):
                if np.sum(S) < current_order or np.sum(S) > self.n - current_order:
                    gt_indicator[i] = True
                S_as_list = N_arr[S]
                W[i] = kernel_weights[current_order][tuple(sorted(S_as_list))]
                for combination in powerset(self.N, current_order, current_order):
                    index = player_indices[combination]
                    D, S_set = set(combination), set(S_as_list)
                    size_S_cap_T = len(S_set.intersection(D))
                    size_interaction_S = len(D)
                    new_S_value = self._get_bernoulli_weights(
                        size_interaction_S=size_interaction_S, size_S_cap_T=size_S_cap_T
                    )
                    new_S[i, index] = new_S_value

            A_dict[current_order] = new_S
            W_dict[current_order] = W
            AW = np.dot(new_S.T, np.diag(W))
            AWA_inverse_dict[current_order] = np.linalg.inv(np.dot(AW, new_S))
            final_weights_dict[current_order] = np.dot(AWA_inverse_dict[current_order], AW)
            gt_indicator_dict[current_order] = gt_indicator

        shapley_interactions = {}
        shapley_approximations = {}

        # B = copy.copy(game_values)
        B_dict = {}
        B_dict[1] = copy.copy(game_values)

        counter = 0
        for s in range(1, self.max_order + 1):
            n_interactions = int(binom(self.n, s))
            # Subset handling for const.
            B = copy.copy(B_dict[s])
            if s == 1:
                const = 0
                phi = np.dot(final_weights_dict[s], B)
            elif s == 2:
                # Compute cardinalities 1<=t<=n-1 with kernel, empty set and full set with gt-weights
                gt_weights_prev = self._get_true_weights(all_S[gt_indicator_dict[s - 1]], N_arr, s)
                const = np.dot(gt_weights_prev.T, B[gt_indicator_dict[s - 1]])
                phi = np.dot(final_weights_dict[s], B)
            else:
                # Compute cardinalities order<=t<=n-order with kernel, remaining with gt-weights
                gt_weights = self._get_true_weights(all_S[gt_indicator_dict[s]], N_arr, s)
                const = np.dot(gt_weights.T, B[gt_indicator_dict[s]])
                phi = np.dot(
                    final_weights_dict[s][:, gt_indicator_dict[s] == False],
                    B[gt_indicator_dict[s] == False],
                )

            shapley_interactions[s] = phi + const
            shapley_approximations[s] = np.dot(A_dict[s], shapley_interactions[s])

            if self._boosting:
                B_dict[s + 1] = B_dict[s] - shapley_approximations[s]
                # B -= shapley_approximations[s]
            else:
                B_dict[s+1] = copy.copy(B_dict[s])

            counter += n_interactions

        return shapley_interactions, shapley_approximations

    def _get_true_weights_all(
        self, constant_subsets, N_arr, min_approx_order, max_approx_order, num_players
    ):
        constant_weights = np.zeros((np.shape(constant_subsets)[0], num_players), dtype=float)
        for i, T in enumerate(constant_subsets):
            T_set = N_arr[T]
            T_size = len(T_set)
            for j, S in enumerate(
                powerset(self.N, max_size=max_approx_order, min_size=min_approx_order)
            ):
                order_set = len(S)
                T_cap_S = len(set(T_set).intersection(set(S)))
                constant_weights[i, j] = (-1) ** (order_set - T_cap_S) / (
                    (self.n - order_set + 1) * binom(self.n - order_set, T_size - T_cap_S)
                )
        return constant_weights

    def _get_true_weights(self, constant_subsets, N_arr, order):
        constant_weights = np.zeros(
            (np.shape(constant_subsets)[0], int(binom(self.n, order))), dtype=float
        )
        for i, T in enumerate(constant_subsets):
            T_set = N_arr[T]
            T_size = len(T_set)
            for j, S in enumerate(powerset(self.N, max_size=order, min_size=order)):
                T_cap_S = len(set(T_set).intersection(set(S)))
                constant_weights[i, j] = (-1) ** (order - T_cap_S) / (
                    (self.n - order + 1) * binom(self.n - order, T_size - T_cap_S)
                )
        return constant_weights

    def approximate_with_budget(self, game_fun, budget, pairing: bool = True):
        # weights for subset sampling
        sampling_weights = self._init_sampling_weights()
        # dictionary for kernel weights
        kernel_weights = {}
        for s in range(1, self.max_order + 1):
            kernel_weights[s] = self._init_kernel_weights(s=s)
        # determine complete and incomplete subsets and sampling budget
        complete_subsets, incomplete_subsets, sampling_budget = determine_complete_subsets(
            budget=budget, n=self.n, s=0, q=sampling_weights
        )
        # generate and sample subsets and game evaluations
        S_list, subsets_counter, n_sampled_subsets, game_values = self.get_S_and_values(
            sampling_budget,
            self.n,
            complete_subsets,
            incomplete_subsets,
            sampling_weights,
            self.N,
            pairing,
            game_fun,
        )
        # compute subset weights for least square optimization
        subset_weights = self.get_S_weights(
            self.n,
            S_list,
            subsets_counter,
            complete_subsets,
            incomplete_subsets,
            sampling_weights,
            n_sampled_subsets,
            kernel_weights,
            self.max_order,
        )

        # Approximate...............
        # print(game_fun(self.N))
        # transform s and v into np.ndarrays
        all_S = np.zeros(shape=(len(S_list), self.n), dtype=bool)
        for i, subset in enumerate(S_list):
            if len(subset) == 0:
                continue
            subset = np.asarray(list(subset))
            all_S[i, subset] = 1
        game_values = np.asarray(game_values)

        shapley_interactions = {}
        shapley_approximations = {}
        # Baseline value
        empty_value = game_fun({})
        shapley_interactions[0] = empty_value
        game_values -= empty_value

        if self._mode == "default":
            shapley_interactions, shapley_approximations = self._get_wlsq_solution(
                S_list=S_list, all_S=all_S, game_values=game_values, kernel_weights=subset_weights
            )
        if self._mode == "full":
            for s in range(1, self.max_order + 1):
                # Regression higher order include lower order
                shapley_interactions[s], shapley_approximations[s] = self._get_wlsq_solution_full(
                    S_list=S_list,
                    all_S=all_S,
                    game_values=game_values,
                    kernel_weights=subset_weights[s],
                    order=s,
                )

                # Boosting - required for most methods, improves performance
                # print("boosting")
                if self._boosting:
                    game_values -= shapley_approximations[s]

        if self._mode == "inconsistent":
            shapley_interactions, shapley_approximations = self._get_wlsq_solution_inconsistent(
                S_list=S_list,
                all_S=all_S,
                game_values=game_values,
                kernel_weights=subset_weights[1],
            )

        result = self.init_results()
        result[0] = np.array([empty_value])
        for s in range(1, self.max_order + 1):
            for i, combination in enumerate(powerset(self.N, max_size=s, min_size=s)):
                result[s][combination] = shapley_interactions[s][i]

        return copy.deepcopy(self._smooth_with_epsilon(result))

    def compute_exact_values(self, game_fun):
        S_list = []
        game_values = []
        kernel_weights = {}

        sampling_weight = self._init_sampling_weights()
        # scale sampling weights to kernel_weights
        kernel_size_weights = np.zeros(self.n + 1)
        for i in range(1, self.n):
            kernel_size_weights[i] = sampling_weight[i] / binom(self.n, i)

        for T in powerset(self.N, 1, self.n - 1):
            S_list.append(set(T))
            game_values.append(game_fun(T))
            kernel_weights[T] = kernel_size_weights[len(T)]

        empty_value = game_fun({})
        full_value = game_fun(self.N)
        S_list.append(set())
        S_list.append(self.N)
        game_values.append(empty_value)
        game_values.append(full_value)
        kernel_weights[()] = self._big_M
        kernel_weights[tuple(self.N)] = self._big_M

        # transform s and v into np.ndarrays
        all_S = np.zeros(shape=(len(S_list), self.n), dtype=bool)
        for i, subset in enumerate(S_list):
            if len(subset) == 0:
                continue
            subset = np.asarray(list(subset))
            all_S[i, subset] = 1
        game_values = np.asarray(game_values)
        game_values = game_values - empty_value

        num_players: int = 0
        for s in range(1, self.s_0 + 1):
            num_players += int(binom(self.n, s))

        i = 0
        player_indices = {}
        player_indices_inv = {}
        for combination in powerset(self.N, max_size=self.s_0, min_size=1):
            player_indices[combination] = i
            player_indices_inv[i] = combination
            i += 1

        N_arr = np.arange(0, self.n)
        W = np.zeros(shape=game_values.shape, dtype=float)
        new_S = np.zeros(shape=(len(S_list), num_players), dtype=bool)
        for i, S in enumerate(all_S):
            S = N_arr[S]
            W[i] = kernel_weights[tuple(S)]
            for s in range(1, self.s_0 + 1):
                for combination in itertools.combinations(S, s):
                    index = player_indices[combination]
                    new_S[i, index] = 1

        A = new_S
        B = game_values
        W = np.sqrt(np.diag(W))
        Aw = np.dot(W, A)
        Bw = np.dot(B, W)
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)

        # result = np.zeros(np.repeat(self.n, self.s_0), dtype=float)
        result = self.init_results()

        for i in range(len(phi)):
            combination = player_indices_inv[i]
            result[len(combination)][combination] = phi[i]

        return copy.deepcopy(self._smooth_with_epsilon(result))

    def _get_bernoulli_weights(self, size_S_cap_T, size_interaction_S):
        # size_S_cap_T: size of intersection S \cap T
        # size_interaction_S: size of interaction |S|
        weight = 0
        for l in range(1, size_S_cap_T + 1):
            weight += binom(size_S_cap_T, l) * self._bernoulli_numbers[size_interaction_S - l]
        return weight

    def _sampling_convergence_test(self, game_fun, budget, pairing=True):
        sampling_weights = self._init_sampling_weights()
        incomplete_subsets = list(range(self.max_order, self.n - self.max_order + 1))
        complete_subsets = list(range(0, self.max_order)) + list(
            range(self.n - self.max_order + 1, self.n + 1)
        )
        sampling_budget = budget

        # dictionary for kernel weights
        kernel_weights = {}
        for s in range(1, self.max_order + 1):
            kernel_weights[s] = self._init_kernel_weights(s=s)
        # determine complete and incomplete subsets and sampling budget
        # generate and sample subsets and game evaluations
        (
            S_list,
            subsets_counter,
            n_sampled_subsets,
            game_values,
        ) = self.get_S_and_values_double_counting(
            sampling_budget,
            self.n,
            complete_subsets,
            incomplete_subsets,
            sampling_weights,
            self.N,
            pairing,
            game_fun,
        )
        # compute subset weights for least square optimization
        subset_weights = self.get_S_weights(
            self.n,
            S_list,
            subsets_counter,
            complete_subsets,
            incomplete_subsets,
            sampling_weights,
            n_sampled_subsets,
            kernel_weights,
            self.max_order,
        )
        print("n sampled subsets:", n_sampled_subsets)
        gt_expectation = {key: 0 for key in range(1, self.max_order + 1)}
        for T in powerset(self.N, self.max_order, self.n - self.max_order):
            game_eval = game_fun(T)
            for s in range(1, self.max_order + 1):
                gt_expectation[s] += kernel_weights[s][len(T)] / binom(self.n, len(T)) * game_eval

        est_expectation = {key: 0 for key in range(1, self.max_order + 1)}
        for i, T in enumerate(S_list):
            if len(T) >= self.max_order and len(T) <= self.n - self.max_order:
                for s in range(1, self.max_order + 1):
                    est_expectation[s] += subset_weights[s][tuple(sorted(T))] * game_values[i]

        mse = {}
        for s in range(1, self.max_order + 1):
            mse[s] = (est_expectation[s] - gt_expectation[s]) ** 2
        return gt_expectation, est_expectation, mse
