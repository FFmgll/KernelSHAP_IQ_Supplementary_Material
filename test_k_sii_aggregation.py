"""This module contains different aggregation techniques for k-SII."""
import copy
import itertools
import random

import numpy as np
import math
from scipy.special import binom, bernoulli

from approximators import BaseShapleyInteractions, SHAPIQEstimator
from approximators.base import determine_complete_subsets, powerset
from games import ParameterizedSparseLinearModel
import time


def measure_execution_time(func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def _aggregate_k_sii_recursively(N,k,sii_vector,pos):
    k_sii_dict = {}
    bernoulli_numbers = bernoulli(k)    # all subsets S with 1 <= |S| <= n
    #initialize Shapley values
    k_sii_dict[1] = sii_vector[:len(N)]
    for l in range(1,k+1):
        num_interactions = 0
        for i in range(1,l+1):
            num_interactions += int(binom(len(N),i))
        if l==1:
            #Shapley value
            k_sii_dict[1] = sii_vector[:num_interactions]
        else:
            k_sii_values = np.zeros(num_interactions)
            #initialize previous k-SII for lower-order interactions
            k_sii_values[:len(k_sii_dict[l-1])] = copy.copy(k_sii_dict[l-1])
            for S_pos,S in enumerate(powerset(N,1,l)):
                subset_size = len(S)
                if subset_size==l:
                    #Top-order
                    k_sii_values[S_pos] = sii_vector[S_pos]
                else:
                    bernoulli_weight = bernoulli_numbers[l-subset_size]
                    #lower-order recursion
                    for tilde_S in powerset(set(N)-set(S),l-subset_size,l-subset_size):
                        sii_effect = sii_vector[pos[tuple(sorted(S+tilde_S))]]
                        k_sii_values[S_pos] += bernoulli_weight*sii_effect
            k_sii_dict[l] = k_sii_values
    return k_sii_dict[k]


def _aggregate_k_sii_explicitly(N,k,sii_vector,pos):
    k_sii_values = np.zeros_like(sii_vector)
    bernoulli_numbers = bernoulli(k)    # all subsets S with 1 <= |S| <= n
    for i,S in enumerate(powerset(N, min_size=1, max_size=k)):
        S_effect = sii_vector[i]
        subset_size = len(S)
        # go over all subsets S_tilde of length |S| + 1, ..., n that contain S
        for S_tilde in powerset(N, min_size=subset_size+1, max_size=k):
            if not set(S).issubset(S_tilde):
                continue
            # get the effect of T
            S_tilde_effect = sii_vector[pos[S_tilde]]
            # normalization with bernoulli numbers
            S_effect += bernoulli_numbers[len(S_tilde) - subset_size] * S_tilde_effect
        k_sii_values[i] = S_effect
    return k_sii_values

def _aggregate_k_sii_matrix(N,k,sii_vector,pos):
    num_interactions = len(sii_vector)
    bernoulli_numbers = bernoulli(k)
    convert_matrix = np.zeros((num_interactions,num_interactions))
    for i,S in enumerate(powerset(N,1,k)):
        for j,S_tilde in enumerate(powerset(N,1,k)):
            if set(S).intersection(set(S_tilde)) == set(S):
                convert_matrix[i,j] = bernoulli_numbers[len(S_tilde)-len(S)]

    k_sii_values = np.dot(convert_matrix,sii_vector)
    return k_sii_values


def _aggregate_k_sii(mode,max_players,max_order):
    for n in range(1, max_players + 1):
        N = range(n)
        k_max = min(n,max_order)
        for k in range(1, k_max + 1):
            game = ParameterizedSparseLinearModel(n=n, weighting_scheme="uniform", n_interactions=1000)
            gt_sii_matrix = game.exact_values_sii(1, k)

            num_interactions = 0
            for l in range(1, k + 1):
                num_interactions += int(binom(n, l))

            gt_sii_vector = np.zeros(num_interactions)

            interaction_position = {}
            for i, S in enumerate(powerset(N, 1, k)):
                gt_sii_vector[i] = gt_sii_matrix[len(S)][S]
                interaction_position[S] = i

            if mode == "all":
                k_sii_values_explicit = _aggregate_k_sii_explicitly(N, k, gt_sii_vector, interaction_position)
                k_sii_values_recursively = _aggregate_k_sii_recursively(N, k, gt_sii_vector, interaction_position)
                k_sii_values_matrix = _aggregate_k_sii_matrix(N, k, gt_sii_vector, interaction_position)
                mse = np.sum((k_sii_values_recursively - k_sii_values_explicit) ** 2 + (
                        k_sii_values_recursively - k_sii_values_matrix) ** 2)
                assert mse < 10e-8, "Aggregation does not match"
            if mode == "explicit":
                k_sii_values_explicit = _aggregate_k_sii_explicitly(N, k, gt_sii_vector, interaction_position)
            if mode == "recursion":
                k_sii_values_recursively = _aggregate_k_sii_recursively(N, k, gt_sii_vector, interaction_position)
            start_time = time.time()
            if mode == "matrix":
                k_sii_values_matrix = _aggregate_k_sii_matrix(N, k, gt_sii_vector, interaction_position)


if __name__ == "__main__":
    max_players = 10
    max_order = 5
    #compare method's results
    _aggregate_k_sii("all", max_players, max_order)
    #compare run-time
    runtimes = {}
    modes = ["recursion","explicit","matrix"]
    for mode in modes:
        start_time = time.time()
        _aggregate_k_sii(mode,max_players,max_order)
        end_time = time.time()
        runtimes[mode] = end_time-start_time

    for mode in modes:
        print(mode,runtimes[mode])