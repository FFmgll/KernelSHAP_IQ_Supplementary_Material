"""This module contains the synthetic games presented by Tsai et al. 2023."""
from scipy.special import binom
from approximators import (
    SHAPIQEstimator,
    RegressionEstimator
)
import numpy as np

class fsi_example_1:
    #Example 1 in FSI Paper
    def __init__(self,n=11,p=0.1):
        self.n = n
        self.p = p
    def set_call(self,T):
        if len(T)<=1:
            return 0
        else:
            return len(T)-self.p*binom(len(T),2)


class fsi_example_2:
    #Example 2 in FSI Paper
    def __init__(self,n=11):
        self.n=n

    def set_call(self,T):
        if len(T) == 0:
            return 0
        elif len(T) == 1:
            return 3
        else:
            return 3*len(T)-(len(T)+2*np.log(len(T)+1))


if __name__ == "__main__":
    n = 10
    N = range(n)
    p=0.1
    ex1 = fsi_example_1(n,p)
    ex2 = fsi_example_2(n)

    order = 2

    fsi = RegressionEstimator(N,order,interaction_type="FSI")
    sii = SHAPIQEstimator(N,order,interaction_type="SII",top_order=False)


    ex1_sii_values = sii.compute_interactions_complete(ex1.set_call)
    ex1_k_sii_values = sii.transform_interactions_in_n_shapley(ex1_sii_values)
    ex1_fsi_values = fsi.compute_exact_values(ex1.set_call)



    ex2_sii_values = sii.compute_interactions_complete(ex2.set_call)
    ex2_k_sii_values = sii.transform_interactions_in_n_shapley(ex2_sii_values)
    ex2_fsi_values = fsi.compute_exact_values(ex2.set_call)

