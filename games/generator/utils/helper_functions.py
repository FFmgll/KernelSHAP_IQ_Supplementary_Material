from itertools import chain, combinations
import pandas as pd
import numpy as np
from scipy import stats


def powerset(iterable):
    s = list(iterable)
    return chain(*[combinations(s, r) for r in range(len(s) + 1)])


def pack_binary_row(binary_row):
    packed_value = 0
    for i, bit in enumerate(binary_row):
        packed_value |= int(bit) << i
    return packed_value


def unpack_binary_value(packed_value, num_bits):
    return [(packed_value >> i) & 1 for i in range(num_bits)]


def compute_mean_mode(data):
    # Check if the input is a DataFrame
    if isinstance(data, pd.DataFrame):
        results = []
        for column in data.columns:
            # If the data type is numeric, compute mean
            if np.issubdtype(data[column].dtype, np.number):
                results.append(data[column].mean())
            # Otherwise, compute mode
            else:
                mode_val = data[column].mode()[0]
                results.append(mode_val)
        return np.array(results)

    # Check if the input is a NumPy array
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Only 2D arrays are supported.")

        results = []
        for i in range(data.shape[1]):
            column = data[:, i]
            # If the data type is numeric, compute mean
            if np.issubdtype(column.dtype, np.number):
                results.append(np.mean(column))
            # Otherwise, compute mode
            else:
                mode_val = stats.mode(column)[0][0]
                results.append(mode_val)
        return np.array(results)

    else:
        raise TypeError("Input must be a Pandas DataFrame or a NumPy array.")


def identity_function(_, y_pred):
    return y_pred

# packed_binary = np.array([pack_binary_row(row) for row in self.precomputed_game])
# print(packed_binary)
#
# num_bits = n
# print(pack_binary_row([0,1,1]))
# print(unpack_binary_value(6,n))
# # Unpack each packed binary value into the original binary rows
# unpacked_binary_array = np.array([unpack_binary_value(val, num_bits) for val in packed_binary])
# print(unpacked_binary_array)
# self.precomputed_game[:, -1] = normalized_value_array
