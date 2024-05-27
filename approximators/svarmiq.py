import copy
import itertools
import math
import random

import numpy as np

from approximators import BaseShapleyInteractions
from approximators.base import determine_complete_subsets, powerset


class SvarmIQ(BaseShapleyInteractions):
    """Estimates the SI (for SII, STI) using the Stratified SVARM sampling approach"""

    def __init__(
        self,
        N,
        order,
        interaction_type="SII",
        top_order: bool = False,
        sample_strat="uniform",
        replacement=True,
        naive_sampling=False,
        dynamic=False,
    ):
        min_order = order if top_order else 1
        super().__init__(N, order, min_order)
        self.interaction_type = interaction_type
        self.orders = list(range(min_order, order + 1))
        self.consumed_budget = 0
        self.strata_estimates = {}
        self.strata_counts = {}
        assert sample_strat in ("uniform", "ksh")
        self.naive_sampling = naive_sampling
        if self.naive_sampling:
            sample_strat = "uniform"
            replacement = True
        self.sample_strat = sample_strat
        self.seen_coalitions = set()
        self.replacement = replacement
        self.max_order = order
        self.smart_factor = 0.5  # the ratio of different samples to be collected in order to switch from saving samples to sampling remaining coalitions
        self.dynamic = dynamic  # set true to adjust the size probability for sampling without replacement with each sampled coalition, otherwise it remains static
        self._big_M = 1000000
        # self.subset_list = None
        # if not replacement:
        # self.subset_list = [tuple(sorted(subset)) for subset in powerset(self.N, min_size=1, max_size=self.n)]

        self.strata_weights = {}
        if self.interaction_type == "SII":
            for k in self.orders:
                self.strata_weights[k] = [
                    1 / ((self.n - k + 1) * math.comb(self.n - k, l))
                    for l in range(0, self.n - k + 1)
                ]
        elif self.interaction_type == "STI":
            for k in self.orders:
                self.strata_weights[k] = [
                    k / (self.n * math.comb(self.n - 1, l)) for l in range(0, self.n - k + 1)
                ]
        elif self.interaction_type == "FSI":
            for k in self.orders:
                self.strata_weights[k] = [
                    (math.factorial(2 * k - 1) / math.pow(math.factorial(k - 1), 2))
                    * (
                        math.factorial(self.n - l - 1)
                        * math.factorial(l + k - 1)
                        / math.factorial(self.n + k - 1)
                    )
                    for l in range(0, self.n - k + 1)
                ]
        elif self.interaction_type == "BHI":
            for k in self.orders:
                self.strata_weights[k] = [math.pow(2, self.n - k) for l in range(0, self.n - k + 1)]

    def _init_sampling_weights(self):
        weight_vector = np.zeros(shape=self.n - 1)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0])) / sum(weight_vector)
        return sampling_weight

    def _init_sampling_weights_size(self):
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

    def approximate_with_budget(self, game, budget):
        self.initialize()

        # initializes probability distribution for sampled coalition sizes
        weight_vector = self._init_sampling_weights()

        # determine sizes for border strata in complete_subsets which are to be computed exactly
        # the remaining to be estimated strata sizes are contained in incomplete_subsets
        if self.naive_sampling:
            complete_subsets = [0, self.n]
            incomplete_subsets = [
                size for size in range(0, self.n + 1) if size not in complete_subsets
            ]
            remaining_budget = budget - 2
        else:
            complete_subsets, incomplete_subsets, remaining_budget = determine_complete_subsets(
                s=0, n=self.n, budget=budget, q=weight_vector
            )
            if 0 not in complete_subsets:
                complete_subsets = [0] + complete_subsets
            if self.n not in complete_subsets:
                complete_subsets = complete_subsets + [self.n]
            # TODO check whether remaining_budget considers 2 token spent for 0 and n size coalitions -> if no adjust remaining budget by - 2

        if self.sample_strat == "ksh":
            probs = copy.deepcopy(weight_vector)
            for complete_subset in complete_subsets:
                probs[complete_subset] = 0.0
            probs = probs / sum(probs)
        else:
            probs = [1.0 for _ in range(0, len(incomplete_subsets))]
            probs = (
                [0 for _ in range(0, int(len(complete_subsets) / 2))]
                + probs
                + [0 for _ in range(0, int(len(complete_subsets) / 2))]
            )
            probs = np.asarray(probs) / sum(probs)

        self.original_probs = copy.deepcopy(probs)

        self.exact_calculation(game, complete_subsets)
        self.consumed_budget += budget - remaining_budget

        # list containing a list for each size s between 0 and n
        # each lists stores the so far sampled coalitions of size as sets
        # sizes in complete_subsets are ignored
        self.sampled_lists = [[] for s in range(0, self.n + 1)]
        self.switch = [False for s in range(0, self.n + 1)]
        self.left_to_sample_lists = [[] for s in range(0, self.n + 1)]

        while budget > self.consumed_budget and sum(probs) > 0:
            self.sample_and_update(game, probs)
            self.consumed_budget += 1

        estimates = self.getEstimates()
        results = self._turn_estimates_into_results(estimates)

        return copy.deepcopy(results)

    # initializes the estimates and the counters for each strata
    # one estimate and counter for each (S,l,W)
    # S: subset of the player set having size of the given orders
    # l: between 0 and n-k
    # W: a subset of S
    def initialize(self):
        self.consumed_budget = 0
        self.strata_estimates = {}
        self.strata_counts = {}
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                self.strata_estimates[tuple(subset)] = [{} for _ in range(0, self.n - k + 1)]
                self.strata_counts[tuple(subset)] = [{} for _ in range(0, self.n - k + 1)]
                for l in range(0, self.n - k + 1):
                    for w in range(0, k + 1):
                        subsets_W = list(itertools.combinations(subset, w))
                        for subset_W in subsets_W:
                            self.strata_estimates[tuple(subset)][l][tuple(subset_W)] = 0
                            self.strata_counts[tuple(subset)][l][tuple(subset_W)] = 0

    # calculates the border strata exactly by evaluating all coalitions of a size contained in complete_subsets
    def exact_calculation(self, game, complete_subsets):
        # iterate over all sizes l that are to be sampled exhaustively
        for l in complete_subsets:
            all_samples = list(itertools.combinations(self.N, l))
            # iterate over coalitions of particular size l
            for sample in all_samples:
                # access the value function only once for each such coalition
                val = game(sample)
                # iterate over all interaction orders k
                for k in self.orders:
                    subsets = list(itertools.combinations(self.N, k))
                    # iterate over all estimates for order k
                    for subset in subsets:
                        subset_W = set(sample).intersection(set(subset))
                        subset_W = tuple(sorted(subset_W))
                        w = len(subset_W)
                        self.strata_estimates[tuple(subset)][l - w][subset_W] += val * (
                            1 / math.comb(self.n - k, l - w)
                        )

    # sample a coalition for each stratum, afterwards all counters are > 0
    def warmup(self, game, budget):
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                for w in range(0, k + 1):
                    subsets_W = list(itertools.combinations(subset, w))
                    for subset_W in subsets_W:
                        for l in range(max(2 - w, 0), min(self.n - 2 - w, self.n - k) + 1):
                            if budget > self.consumed_budget:
                                available_players = list(set(self.N).copy().difference(subset))
                                coalition = set(
                                    np.random.choice(available_players, l, replace=False)
                                )
                                self.strata_estimates[tuple(subset)][l][tuple(subset_W)] = game(
                                    coalition
                                )
                                self.strata_counts[tuple(subset)][l][tuple(subset_W)] = 1
                                self.consumed_budget += 1

    # sample a set and updates the strata accordingly
    def sample_and_update(self, game, probs):
        # first check whether in case of sampling without replacement all coalitions are already sampled once
        # if so then no update is needed
        if (not self.replacement) and sum(probs) == 0:
            return

        coalition, size = self._sample_coalition(probs)
        coalition = set(coalition)
        val = game(coalition)
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                subset_W = coalition.intersection(set(subset))
                subset_W = tuple(sorted(subset_W))
                w = len(subset_W)
                avg_old = self.strata_estimates[tuple(subset)][size - w][subset_W]
                count_old = self.strata_counts[tuple(subset)][size - w][subset_W]
                self.strata_estimates[tuple(subset)][size - w][subset_W] = (
                    avg_old * count_old + val
                ) / (count_old + 1)
                self.strata_counts[tuple(subset)][size - w][subset_W] += 1

    # aggregates all strata estimates to estimates, one for each considered coalition, and returns them
    def getEstimates(self):
        estimates = {}
        for k in self.orders:
            subsets = list(itertools.combinations(self.N, k))
            for subset in subsets:
                estimates[tuple(subset)] = 0
                for l in range(0, self.n - k + 1):
                    factor = math.comb(self.n - k, l) * self.strata_weights[k][l]
                    for w in range(0, k + 1):
                        subsets_W = list(itertools.combinations(subset, w))
                        for subset_W in subsets_W:
                            strata_estimate = self.strata_estimates[tuple(subset)][l][
                                tuple(subset_W)
                            ]
                            if (k - w) % 2 == 0:
                                estimates[tuple(subset)] += factor * strata_estimate
                            else:
                                estimates[tuple(subset)] -= factor * strata_estimate
        return estimates

    def _turn_estimates_into_results(self, estimates: dict[tuple, float]) -> dict[int, np.ndarray]:
        results: dict[int, np.ndarray] = self.init_results()
        for coalition, estimate in estimates.items():
            order = len(coalition)
            results[order][coalition] = estimate
        return results

    def _sample_coalition(self, probs):
        # sample a coalition size and coalition depending on with replacement or without
        if self.replacement:
            size = int(np.random.choice(range(0, self.n + 1), 1, p=probs))
            coalition = set(np.random.choice(list(self.N), size, replace=False))
            return coalition, size
        else:
            # draw size s according to probability distribution
            size = int(np.random.choice(range(0, self.n + 1), 1, p=probs))

            # update switch state and list if number of samled coalitions is at least the smart_factor ratio of all coalitions of that size
            if not self.switch[size] and len(
                self.sampled_lists[size]
            ) >= self.smart_factor * math.comb(self.n, size):
                for subset in list(itertools.combinations([i for i in range(self.n)], size)):
                    if set(subset) not in self.sampled_lists[size]:
                        self.left_to_sample_lists[size].append(set(subset))
                self.switch[size] = True

            # draw coalition from list of remaining ones or exhaustively check whether it has been sampled before
            if self.switch[size]:
                coalition = self._sample_from_remaining(size)
            else:
                coalition = self._sample_exhaustively(size)

            # add the new coalition to the list and the update sampling distribution probs
            self.sampled_lists[size].append(set(coalition))
            self._update_probs(size, probs)
            return coalition, size

    # sample a coalition of fixed size from the list of reaming coalitions and removes it from the list
    def _sample_from_remaining(self, size):
        index = random.randint(0, len(self.left_to_sample_lists[size]) - 1)
        return self.left_to_sample_lists[size].pop(index)

    # sample a coalition of fixed size from the whole space and check whether it ahs been sampled before
    def _sample_exhaustively(self, size):
        while True:
            coalition = np.random.choice(list(self.N), size, replace=False)
            if not self._alreadySampled(coalition):
                return coalition

    # returns a boolean whether the coalition has been sampled before by searching it in its size list
    def _alreadySampled(self, coalition):
        return set(coalition) in self.sampled_lists[len(coalition)]

    # updates the sampling distribution probs after a coalition of given size has been sampled
    def _update_probs(self, size, probs):
        probs_sum = sum(probs)
        s = size
        stratum_size = math.comb(self.n, s)
        old_prob_s = probs[s]
        if self.dynamic:
            sampled = len(self.sampled_lists[s])
            if sampled == stratum_size:
                probs[s] = 0
                if probs_sum != old_prob_s:
                    new_sum = sum(probs)
                    for i in range(0, self.n + 1):
                        probs[i] /= new_sum
            else:
                probs[s] = self.original_probs[s] - (
                    (self.original_probs[s] * sampled) / stratum_size
                )
                sum_others = sum(probs) - probs[s]
                if probs_sum != old_prob_s:
                    for i in range(0, s):
                        probs[i] *= (1 - probs[s]) / sum_others
                    for i in range(s + 1, self.n - 1):
                        probs[i] *= (1 - probs[s]) / sum_others
                else:
                    probs[s] = 1
        else:
            # adjust sample distribution, all coalitions of size s could have been sampled, set to zero, rescale the others
            if len(self.sampled_lists[s]) == stratum_size:
                # check whether this was the last size, if so then set all probabilities to zero, sampling is finished
                probs[s] = 0
                if probs_sum != old_prob_s:
                    new_sum = sum(probs)
                    for i in range(0, self.n + 1):
                        probs[i] /= new_sum
