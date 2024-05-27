import os

import numpy as np
import pandas as pd

from typing import List, Union
from ..utils.helper_functions import pack_binary_row
from .player import create_generic_players, names_or_indices_to_players, str_to_players


class PrecomputedGame:
    """
    A class for handling games with precomputed values, allowing for quick querying of game outcomes
    based on player coalitions.

    Attributes:
        players (List[Player]): List of players in the game.
        num_players (int): Number of players in the game.
        memory_efficient (bool): Flag to indicate if the game values are stored in a memory-efficient format.
        precomputed_game (Union[np.ndarray, Dict[tuple, float]]): The precomputed game values.
    """

    def __init__(self,
                 player_names_or_indices: Union[List[str], List[int]],
                 path_to_file: str,
                 filename: str,
                 memory_efficient: bool = True
                 ):
        """
        Initializes the PrecomputedGame class.

        Args:
            player_names_or_indices (Union[List[str], List[int]]): List of names given to players in the
            game.
            path_to_file (str): Path to the directory containing the precomputed game file.
            filename (str): Name of the file containing the precomputed game values.
            memory_efficient (bool): If True, expects the game values to be in a memory-efficient format.
        """
        self.players = create_generic_players(player_names_or_indices)
        self.num_players = len(self.players)
        self.memory_efficient = memory_efficient
        self.precomputed_game = None
        self.load_game(path_to_file, filename)

    def __call__(self, S: List[List]) -> np.ndarray:
        """
        Queries the value of each coalition in S from the precomputed game.

        Args:
            S (List[List]): A list of subsets of players.

        Returns:
            np.ndarray: An array of values for each coalition in S.

        Raises:
            ValueError: If the precomputed game has not been loaded.
        """

        # To be able to also use strings to query, this maps str/int/Player to the players:
        S = [sorted(names_or_indices_to_players(self.players, lst)) for lst in S]

        if self.precomputed_game is not None:
            value_list = []
            for s in S:
                value = self._query_loaded_game(s)
                value_list.append(value)
            return np.array(value_list)
        else:
            raise ValueError("No precomputed game found")

    def load_game(self,
                  path_to_file: str,
                  filename: str
                  ) -> None:
        """
        Loads the precomputed game from a file.

        Args:
            path_to_file (str): Path to the directory containing the precomputed game file.
            filename (str): Name of the file containing the precomputed game values.
        """
        if self.memory_efficient:
            self.precomputed_game = np.load(os.path.join(path_to_file, filename + ".npy"))

            # Load column names
            # with open(os.path.join(path_to_file, filename + ".txt"), "r") as f:
            #     loaded_column_names = [line.strip() for line in f.readlines()]
            # TODO - what do I do with this?
        else:
            self.precomputed_game = pd.read_csv(os.path.join(path_to_file, filename + ".csv"))
            self.precomputed_game['coalition'] = self.precomputed_game['coalition'].apply(
                lambda str: str_to_players(str, self.players))
            self.precomputed_game = dict(zip(self.precomputed_game['coalition'], self.precomputed_game['value']))

    def _query_loaded_game(self, s: List) -> float:
        """
        Queries the value of a specific coalition from the precomputed game.

        Args:
            s (List): A subset of players.

        Returns:
            float: The value of the specified coalition.
        """
        if self.memory_efficient:
            binary_repr = np.zeros(self.num_players)
            s = [self.players.index(element) for element in s]
            binary_repr[s] = 1
            value_array_idx = pack_binary_row(binary_repr)
            return self.precomputed_game[value_array_idx]
        else:
            return self.precomputed_game[tuple(s)]

    def save_config(self):
        raise NotImplementedError

    def load_config(self):
        raise NotImplementedError
