"""This module contains the empirical validation of the presented conjectures."""
import copy
import itertools
import random

import numpy as np
import math
from scipy.special import binom, bernoulli

from approximators import BaseShapleyInteractions
from approximators.base import determine_complete_subsets, powerset
from games import ParameterizedSparseLinearModel


def _get_kernel_weights(d, k, mu_inf):
    # vector that determines the kernel weights for KernelSHAPIQ
    weight_vector = np.zeros(shape=d + 1)
    for subset_size in range(0, d + 1):
        if (subset_size < k) or (subset_size > d - k):
            weight_vector[subset_size] = mu_inf
        else:
            weight_vector[subset_size] = 1 / binom(d - 2 * k, subset_size - k)
    return weight_vector

def _get_bernoulli_weights(bernoulli_numbers,size_S_cap_T, size_interaction_S):
    # size_S_cap_T: size of intersection S \cap T
    # size_interaction_S: size of interaction |S|
    weight = 0
    for l in range(1, size_S_cap_T + 1):
        weight += binom(size_S_cap_T, l) * bernoulli_numbers[size_interaction_S - l]
    return weight

def _gt_inverse(D,d,k,n_interactions):
    #computes exact inverse as mentioned in conjecture
    gt_inverse = np.zeros((n_interactions,n_interactions))
    for i, S1 in enumerate(powerset(D,k,k)):
        for j, S2 in enumerate(powerset(D,k,k)):
            S1_cap_S2 = len(set(S1).intersection(S2))
            gt_inverse[i,j] = (-1)**(k-S1_cap_S2)/((d-k+1)*binom(d-k,k-S1_cap_S2))
    return gt_inverse

def _empirical_inverse(D, n_subsets, n_interactions):
    #Compute A_k as the empirical inverse
    bernoulli_numbers = bernoulli(d)
    mu_inf = 10000000
    kernel_weights = _get_kernel_weights(d, k, mu_inf)
    X = np.zeros((n_subsets, n_interactions))
    W = np.zeros((n_subsets, n_subsets))
    for i, T in enumerate(powerset(D)):
        t = len(T)
        W[i, i] = kernel_weights[t]
        for j, S in enumerate(powerset(D, k, k)):
            T_cap_S = len(set(T).intersection(set(S)))
            X[i, j] = _get_bernoulli_weights(bernoulli_numbers, T_cap_S, k)
    inverse = np.linalg.inv(np.dot(X.T, np.dot(W, X)))
    return inverse, X, W

def _validate_conjecture_inverse(d,k):
    #compute inverse and validate conjecture
    assert d >= 2*k, "It must hold d>=2k"
    D = range(0,d)
    n_interactions = int(binom(d,k))
    n_subsets = 2**d
    emp_inverse, X, W = _empirical_inverse(D,n_subsets,n_interactions)
    gt_inverse = _gt_inverse(D,d,k,n_interactions)
    mse = np.average((emp_inverse-gt_inverse)**2)
    return emp_inverse, mse

def sii_weight(d,t,k):
    #SII weight m(t)
    return math.factorial(d-t-k)*math.factorial(t)/(math.factorial(d-k+1))

def _validate_conjecture_sii(d,k):
    D = range(0, d)
    n_interactions = int(binom(d, k))
    n_subsets = 2 ** d
    #Compute empirical inverse and X and W, independent of game
    emp_inverse, X, W = _empirical_inverse(D, n_subsets, n_interactions)
    n_games = 10
    mse_errors = np.zeros(n_games)
    for l in range(n_games):
        # Initialize random SOUMs
        game = ParameterizedSparseLinearModel(n=d, weighting_scheme="uniform", n_interactions=1000)
        #Empirical SII
        game_values = np.zeros(n_subsets)
        y_plus_indicator = np.zeros(n_subsets)
        Q_gt_sii_weight = np.zeros((n_interactions,n_subsets))
        for i,T in enumerate(powerset(D)):
            t = len(T)
            game_values[i] = game.set_call(T)
            if t >= k and t <= d-k:
                y_plus_indicator[i] = 1
            else:
                for j,S in enumerate(powerset(D,k,k)):
                    T_cap_S = len(set(T).intersection(S))
                    #GT SII weights, cf. representation in Fumagalli et al. (2023)
                    Q_gt_sii_weight[j,i] = (-1)**(k-T_cap_S)*sii_weight(d,t-T_cap_S,k)
        y_plus = y_plus_indicator*game_values
        y_minus = (1-y_plus_indicator)*game_values

        Q_y_minus = np.dot(Q_gt_sii_weight,y_minus)
        wls_solution = np.dot(emp_inverse,np.dot(X.T,np.dot(W,y_plus)))
        emp_sii = Q_y_minus + wls_solution
        # GT SII values
        gt_sii_matrix = game.exact_values_sii(k, k)[k]
        gt_sii = np.zeros(n_interactions)
        for i,S in enumerate(powerset(D,k,k)):
            gt_sii[i] = gt_sii_matrix[tuple(sorted(S))]
        #Compare MSE
        mse_errors[l] = np.average((emp_sii-gt_sii)**2)
    return np.average(mse_errors)

if __name__ == "__main__":
    d_max = 12
    for d in range(2,d_max):
        k_max = int(d/2) #Conjectures hold for d>=2k
        for k in range(1,k_max+1):
            # Validate conjecture for inverse
            emp_inverse, mse_inverse = _validate_conjecture_inverse(d,k)
            assert mse_inverse < 10e-10, "Inverse error"
            #Validate conjecture for SII
            mse_sii = _validate_conjecture_sii(d,k)
            assert mse_sii < 10e-10, "SII error"
