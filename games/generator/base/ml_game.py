import os

import numpy as np
import pandas as pd

from typing import Any, Callable, List, Optional, Union
from .base_game import BaseGame
from ..utils.helper_functions import powerset, pack_binary_row
from .player import names_or_indices_to_players, players_to_str, str_to_players


class MLGame(BaseGame):
    """
    A class representing a machine learning game, used for evaluating different coalitions of players
    based on their contributions to a predictive model's performance.

    Attributes:
        X (Any): The feature set used for the model.
        y (Any): The target variable.
        model (Any): The predictive model.
        player_names_or_indices (Union[List[str], List[int]]): A list of names or indices for players in the game.
        imputation_strategy (Callable): A strategy function for data imputation.
        loss_function (Optional[Callable]): A function to calculate the loss.
        get_null_set_predictions (Optional[Callable]): A function to get predictions for an empty set.
        impute_params (Optional[Any]): Parameters for the imputer function.
        lower_is_better (bool): Indicator whether lower values are better for the loss function.
        memory_efficient (bool): If True, uses a memory-efficient approach for storing the game values.
        precomputed_game (Optional[Union[np.ndarray, pd.DataFrame, dict]]): Precomputed game values.
        num_players (int): Number of players in the game.
        value_empty_set (Optional[float]): Value of the empty set, if applicable.
    """

    def __init__(
        self,
        X: Any,
        y: Any,
        model: Any,
        player_names_or_indices: Union[List[str], List[int]],
        imputation_strategy: Callable,
        loss_function: Optional[Callable] = None,
        get_null_set_predictions: Optional[Callable] = None,
        impute_params: Optional[Any] = None,
        lower_is_better: bool = True,
        memory_efficient: bool = True,
    ):
        super(MLGame, self).__init__(player_names_or_indices, X)
        self.precomputed_game = None
        self.imputation_strategy = imputation_strategy
        self.X = X
        self.y = y
        self.model = model
        self.memory_efficient = memory_efficient

        if loss_function is not None:
            self.loss_function = loss_function

        if imputation_strategy is not None and callable(imputation_strategy):
            self.imputer = imputation_strategy

        if get_null_set_predictions is not None:
            self.get_null_set_predictions = get_null_set_predictions

        if get_null_set_predictions and loss_function is not None:
            self.value_empty_set = self._compute_value([])

        # TODO - Write validator for impute_params
        self.impute_params = impute_params
        self.lower_is_better = lower_is_better
        self.num_players = len(self.players)

    def __call__(self, S: List[List]) -> np.ndarray:
        """
        Computes the value for each subset of players.

        Args:
            S (List[List]): A list of subsets of players.

        Returns:
            np.ndarray: An array of values for each subset.
        """

        # To be able to also use names or indices to query, this maps str/int/Player to the player objects
        # Players are then sorted according to player.number
        S = [sorted(names_or_indices_to_players(self.players, lst)) for lst in S]

        value_list = []
        if self.precomputed_game is None:
            for s in S:
                value = self._compute_value(s)
                # TODO - check
                value = value - self.value_empty_set
                if self.lower_is_better:
                    value = value * -1
                value_list.append(value)
        else:
            for s in S:
                value = self._query_loaded_game(s)
                value_list.append(value)
        return np.array(value_list)

    def _compute_value(self, s: List) -> float:
        """
        Computes the value of a given subset of players.

        Args:
            s (List): A subset of players.

        Returns:
            float: The computed value for the subset.
        """
        y = None
        if len(s) > 0:
            is_player_present = all(elem in self.players for elem in s)
            if is_player_present:
                pred_imputed_X, y = self.imputer(
                    s=s, model=self.model, X=self.X, y=self.y, impute_params=self.impute_params
                )
            else:
                raise ValueError("Players in this coalition were not provided during instantiation")
        else:
            pred_imputed_X = self.get_null_set_predictions(self.y)
        if y is None:
            y = self.y
        value = self.loss_function(y, pred_imputed_X)
        return value

    def precompute(self) -> None:
        """
        Precomputes the game values for all subsets of players and stores them for efficient querying.
        """
        assert self.players == sorted(self.players, key=(lambda player: player.number))
        S = list(powerset(self.players))
        value_array = self.__call__(S)
        if self.memory_efficient:
            binary_matrix = np.zeros((2**self.num_players, self.num_players))
            self.precomputed_game = np.zeros((2**self.num_players,))
            for row, subset in enumerate(S):
                subset = [self.players.index(element) for element in subset]
                for item in subset:
                    binary_matrix[row, item] = 1
                packed_binary = pack_binary_row(binary_matrix[row, :])
                self.precomputed_game[packed_binary] = value_array[row]
        else:
            self.precomputed_game = pd.DataFrame(
                columns=["coalition", "value"], index=range(2**self.num_players)
            )
            row_count = self.precomputed_game.shape[0]
            for row in range(row_count):
                self.precomputed_game.iloc[row, 0] = S[row]
                self.precomputed_game.iloc[row, 1] = value_array[row]
            self.precomputed_game = dict(
                zip(self.precomputed_game["coalition"], self.precomputed_game["value"])
            )

    def save_game(self, filename: str, output_dir: str) -> None:
        """
        Saves the precomputed game to a file.

        Args:
            filename (str): The name of the file to save the game.
            output_dir (str): The directory where the file will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.memory_efficient:
            if self.precomputed_game is not None:
                np.save(os.path.join(output_dir, filename + ".npy"), self.precomputed_game)

            if self.players is not None:
                column_names = [str(player) for player in self.players]
                with open(os.path.join(output_dir, filename + ".txt"), "w") as f:
                    for name in column_names:
                        f.write(name + "\n")
        else:
            precomputed_game_df = pd.DataFrame(
                list(self.precomputed_game.items()), columns=["coalition", "value"]
            )
            precomputed_game_df["coalition"] = precomputed_game_df["coalition"].apply(
                players_to_str
            )
            precomputed_game_df.to_csv(os.path.join(output_dir, filename + ".csv"), index=False)

    def load_game(self, filename: str, path_to_file: str) -> None:
        """
        Loads a precomputed game from a file.

        Args:
            filename (str): The name of the file containing the game.
            path_to_file (str): The path to the file.
        """
        if self.memory_efficient:
            self.precomputed_game = np.load(os.path.join(path_to_file, filename + ".npy"))

            # Load column names
            # with open(os.path.join(path_to_file, filename + ".txt"), "r") as f:
            #     loaded_column_names = [line.strip() for line in f.readlines()]
            # TODO - what do I do with this?
        else:
            self.precomputed_game = pd.read_csv(os.path.join(path_to_file, filename + ".csv"))
            self.precomputed_game["coalition"] = self.precomputed_game["coalition"].apply(
                lambda str: str_to_players(str, self.players)
            )
            self.precomputed_game = dict(
                zip(self.precomputed_game["coalition"], self.precomputed_game["value"])
            )

    def _query_loaded_game(self, s: List) -> float:
        """
        Queries the value of a given subset from the precomputed game.

        Args:
            s (List): A subset of players.

        Returns:
            float: The value of the given subset.
        """
        is_player_present = True
        if len(s) > 0:
            is_player_present = all(elem in self.players for elem in s)
        if is_player_present:
            if self.memory_efficient:
                binary_repr = np.zeros(self.num_players)
                s = [self.players.index(element) for element in s]
                binary_repr[s] = 1
                value_array_idx = pack_binary_row(binary_repr)
                return self.precomputed_game[value_array_idx]
            else:
                assert s == sorted(s, key=(lambda player: player.number))
                return self.precomputed_game[tuple(s)]
        else:
            raise ValueError("Players in this coalition were not provided during instantiation")

    def _remove_and_refit(
        self, s: List, model: Any, X: Any, y: Any, impute_params: Optional[Any] = None
    ) -> None:
        raise NotImplementedError

    def _impute_marginals(
        self,
        S: List[List],
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        impute_params: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError

    def _impute_conditionals(
        self,
        S: List[List],
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        impute_params: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError

    def save_config(self):
        raise NotImplementedError

    def load_config(self):
        raise NotImplementedError
