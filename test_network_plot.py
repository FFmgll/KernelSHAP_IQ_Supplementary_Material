"""This script tests the network plot function."""
import numpy as np
from matplotlib import pyplot as plt

from approximators.plot import network_plot

if __name__ == "__main__":
    example_nsii_values = {
        1: np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        2: np.repeat(0.05, 25).reshape(5, 5),
    }
    example_nsii_values[2][(0, 3)] = 0.3
    example_feature_names = ["0", "1", "2", "3", "4"]

    fig_network, axis_network = network_plot(
        first_order_values=example_nsii_values[1],
        second_order_values=example_nsii_values[2],
        feature_names=example_feature_names,
    )
    title_network: str = "network plot"
    axis_network.set_title(title_network)
    plt.show()
